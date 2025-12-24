"""Main client for driver node to coordinate PS cluster."""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from pyspark_ps.communication.protocol import (
    MessageType,
    PSMessage,
    ServerInfo,
    CheckpointInfo,
)
from pyspark_ps.communication.rpc_handler import RPCClient, MultiServerClient
from pyspark_ps.communication.serialization import Serializer
from pyspark_ps.server.ps_server import PSServer
from pyspark_ps.client.barrier import BarrierCoordinator
from pyspark_ps.utils.config import PSConfig
from pyspark_ps.utils.logging import get_logger


class PSMainClient:
    """
    Main Parameter Server Client running on Spark DRIVER node.
    
    Responsibilities:
    - Server lifecycle management (start/shutdown)
    - Barrier synchronization for all worker clients
    - Coordinate server-side decay operations
    - Global state management
    - S3 persistence operations
    
    NOTE: This client runs ONLY on the driver, not on executors.
    """
    
    def __init__(self, spark_context, config: PSConfig):
        """
        Initialize main client on driver node.
        
        Must be created before any worker clients.
        
        Args:
            spark_context: Spark context (can be None for local testing)
            config: PS configuration
        """
        self.spark_context = spark_context
        self.config = config
        self.client_id = f"main_{uuid.uuid4().hex[:8]}"
        
        self.logger = get_logger("main_client")
        self.serializer = Serializer(
            compression=config.compression,
            compression_algorithm=config.compression_algorithm
        )
        
        # Server management
        self._servers: List[PSServer] = []
        self._server_info: List[ServerInfo] = []
        self._rpc_client = RPCClient(timeout=config.timeout_seconds)
        self._multi_client: Optional[MultiServerClient] = None
        self._executor = ThreadPoolExecutor(max_workers=config.num_servers * 2)
        
        # Barrier coordination
        self._barrier_coordinator: Optional[BarrierCoordinator] = None
        
        # State
        self._started = False
        self._step_count = 0
    
    def start_servers(self) -> List[ServerInfo]:
        """
        Start PS servers on the driver node.
        
        All servers run as threads on the driver. Workers (on executors)
        connect to the driver's IP address to access the PS.
        
        Returns:
            List of server endpoints for worker clients
        """
        if self._started:
            return self._server_info
        
        self.logger.info(f"Starting {self.config.num_servers} PS servers on driver")
        
        # Get driver IP for worker connections
        driver_host = self._get_driver_host()
        self.logger.info(f"Driver host for worker connections: {driver_host}")
        
        # Start all servers on driver as threads
        self._servers = PSServer.discover_and_start(
            None,  # Use local mode - all servers on driver
            self.config
        )
        
        # Wait for servers to be ready
        time.sleep(0.5)
        
        # Get server info and update host to driver's external IP
        # so workers on executors can connect
        self._server_info = []
        for server in self._servers:
            info = server.get_server_info()
            # Replace localhost with driver's external IP
            updated_info = ServerInfo(
                server_id=info.server_id,
                host=driver_host,
                port=info.port,
                status=info.status,
                shard_range=info.shard_range,
                metadata=info.metadata,
            )
            self._server_info.append(updated_info)
        
        # Initialize multi-server client
        server_addrs = [(s.host, s.port) for s in self._server_info]
        self._multi_client = MultiServerClient(
            server_addrs,
            timeout=self.config.timeout_seconds
        )
        
        # Initialize barrier coordinator
        self._barrier_coordinator = BarrierCoordinator(
            self._server_info,
            timeout_seconds=self.config.timeout_seconds
        )
        
        self._started = True
        self.logger.info(f"Started {len(self._server_info)} PS servers")
        
        return self._server_info
    
    def _get_driver_host(self) -> str:
        """
        Get the driver's host address for worker connections.
        
        Priority:
        1. Spark driver host from config
        2. External IP via socket connection
        3. Local hostname
        4. Fallback to 127.0.0.1
        """
        import socket
        
        # Try to get from Spark config
        if self.spark_context is not None:
            try:
                driver_host = self.spark_context.getConf().get("spark.driver.host")
                if driver_host:
                    self.logger.info(f"Using driver host from Spark config: {driver_host}")
                    return driver_host
            except Exception:
                pass
        
        # Try to get external IP by connecting to an external address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            # Connect to a public DNS (doesn't actually send data)
            s.connect(("8.8.8.8", 80))
            external_ip = s.getsockname()[0]
            s.close()
            self.logger.info(f"Using external IP: {external_ip}")
            return external_ip
        except Exception:
            pass
        
        # Try local hostname
        try:
            hostname = socket.gethostname()
            host_ip = socket.gethostbyname(hostname)
            if host_ip and not host_ip.startswith("127."):
                self.logger.info(f"Using hostname IP: {host_ip}")
                return host_ip
        except Exception:
            pass
        
        # Fallback
        self.logger.warning("Could not determine external IP, using 127.0.0.1")
        return "127.0.0.1"
    
    def _is_local_mode(self) -> bool:
        """Check if Spark is in local mode."""
        if self.spark_context is None:
            return True
        
        try:
            master = self.spark_context.master
            return master.startswith("local")
        except Exception:
            return True
    
    def _verify_servers(self, server_infos: List[ServerInfo], max_retries: int = 5):
        """Verify all servers are reachable."""
        for retry in range(max_retries):
            all_ok = True
            for server_info in server_infos:
                try:
                    message = PSMessage(
                        msg_type=MessageType.GET_STATS,
                        client_id=self.client_id,
                        payload=b""
                    )
                    response = self._rpc_client.call(
                        server_info.host,
                        server_info.port,
                        message
                    )
                    if response.msg_type == MessageType.RESPONSE_ERROR:
                        all_ok = False
                except Exception as e:
                    self.logger.warning(
                        f"Server {server_info.server_id} at {server_info.host}:{server_info.port} "
                        f"not reachable (attempt {retry + 1}): {e}"
                    )
                    all_ok = False
            
            if all_ok:
                self.logger.info("All PS servers verified and reachable")
                return
            
            time.sleep(1.0)
        
        self.logger.warning("Some PS servers may not be fully ready")
    
    def shutdown_servers(self, grace_period_seconds: float = 30):
        """
        Gracefully shutdown all server nodes.
        
        1. Broadcast shutdown signal to all servers
        2. Wait for pending operations to complete
        3. Collect final statistics
        
        Args:
            grace_period_seconds: Time to wait for graceful shutdown
        """
        if not self._started:
            return
        
        self.logger.info("Shutting down PS servers")
        
        # Send shutdown to all servers
        for server_info in self._server_info:
            try:
                message = PSMessage(
                    msg_type=MessageType.SHUTDOWN,
                    client_id=self.client_id,
                    payload=b""
                )
                self._rpc_client.call(
                    server_info.host,
                    server_info.port,
                    message
                )
            except Exception as e:
                self.logger.warning(f"Error sending shutdown to server {server_info.server_id}: {e}")
        
        # For local servers, wait for shutdown
        for server in self._servers:
            try:
                server.shutdown(grace_period_seconds)
            except Exception as e:
                self.logger.warning(f"Error shutting down server: {e}")
        
        # Cleanup
        if self._multi_client:
            self._multi_client.close()
        if self._barrier_coordinator:
            self._barrier_coordinator.close()
        self._rpc_client.close()
        self._executor.shutdown(wait=True)
        
        self._started = False
        self._servers.clear()
        self._server_info.clear()
        
        self.logger.info("PS servers shutdown complete")
    
    def get_server_info(self) -> List[ServerInfo]:
        """Return server endpoints for worker client initialization."""
        return self._server_info.copy()
    
    # Barrier Synchronization
    
    def create_barrier(self, name: str, num_workers: int):
        """
        Create a named barrier for worker synchronization.
        
        Must be called before workers attempt to enter barrier.
        
        Args:
            name: Barrier name
            num_workers: Number of workers that must enter
        """
        if not self._barrier_coordinator:
            raise RuntimeError("Servers not started")
        
        self._barrier_coordinator.create(name, num_workers)
        self.logger.debug(f"Created barrier '{name}' for {num_workers} workers")
    
    def wait_barrier(self, name: str, timeout_seconds: float = 300) -> bool:
        """
        Wait for all workers to reach the barrier.
        
        Blocks until all num_workers have entered.
        
        Args:
            name: Barrier name
            timeout_seconds: Timeout
            
        Returns:
            True if all workers entered, False if timeout
        """
        if not self._barrier_coordinator:
            raise RuntimeError("Servers not started")
        
        return self._barrier_coordinator.wait(name, timeout_seconds)
    
    def release_barrier(self, name: str):
        """
        Release all workers waiting at the barrier.
        
        Workers will proceed after this call.
        
        Args:
            name: Barrier name
        """
        if not self._barrier_coordinator:
            raise RuntimeError("Servers not started")
        
        self._barrier_coordinator.release(name)
        self.logger.debug(f"Released barrier '{name}'")
    
    # Decay Operations
    
    def decay_embeddings(
        self,
        method: str = "multiply",
        factor: float = 0.99,
        min_count: int = 5
    ) -> Dict[str, Any]:
        """
        Coordinate decay operation across all servers.
        
        Args:
            method: "multiply" or "prune"
            factor: For multiply method, the decay factor
            min_count: For prune method, minimum count threshold
            
        Returns:
            Dict with operation results
        """
        if not self._started:
            raise RuntimeError("Servers not started")
        
        self.logger.info(f"Executing decay: method={method}, factor={factor}, min_count={min_count}")
        
        results = []
        
        for server_info in self._server_info:
            message = PSMessage(
                msg_type=MessageType.DECAY_EMBEDDINGS,
                client_id=self.client_id,
                payload=self.serializer.serialize({
                    "method": method,
                    "factor": factor,
                    "min_count": min_count,
                })
            )
            
            try:
                response = self._rpc_client.call(
                    server_info.host,
                    server_info.port,
                    message
                )
                
                if response.msg_type == MessageType.RESPONSE_DATA:
                    result = self.serializer.deserialize(response.payload)
                    results.append({
                        "server_id": server_info.server_id,
                        **result
                    })
                    
            except Exception as e:
                self.logger.error(f"Decay failed on server {server_info.server_id}: {e}")
                results.append({
                    "server_id": server_info.server_id,
                    "error": str(e)
                })
        
        return {"results": results, "method": method}
    
    # Model Initialization
    
    def init_weights(
        self,
        weight_shapes: Dict[str, Tuple[int, ...]],
        init_strategy: str = "normal",
        init_scale: float = 0.01
    ):
        """
        Initialize weight tensors on all servers.
        
        Args:
            weight_shapes: Dict mapping layer name to shape
            init_strategy: Initialization strategy
            init_scale: Scale for initialization
        """
        if not self._started:
            raise RuntimeError("Servers not started")
        
        self.logger.info(f"Initializing {len(weight_shapes)} weight tensors")
        
        for server_info in self._server_info:
            message = PSMessage(
                msg_type=MessageType.INIT_WEIGHTS,
                client_id=self.client_id,
                payload=self.serializer.serialize({
                    "shapes": {k: list(v) for k, v in weight_shapes.items()},
                    "init_strategy": init_strategy,
                    "init_scale": init_scale,
                })
            )
            
            response = self._rpc_client.call(
                server_info.host,
                server_info.port,
                message
            )
            
            if response.msg_type == MessageType.RESPONSE_ERROR:
                error = self.serializer.deserialize(response.payload)
                raise RuntimeError(f"Failed to init weights on server {server_info.server_id}: {error}")
    
    # Monitoring
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Collect statistics from all servers.
        
        Returns:
            Dict with embedding counts, update stats, memory usage, etc.
        """
        if not self._started:
            return {"error": "Servers not started"}
        
        stats = {
            "servers": [],
            "total_embeddings": 0,
            "total_weights": 0,
            "total_memory_bytes": 0,
        }
        
        for server_info in self._server_info:
            try:
                message = PSMessage(
                    msg_type=MessageType.GET_STATS,
                    client_id=self.client_id,
                    payload=b""
                )
                
                response = self._rpc_client.call(
                    server_info.host,
                    server_info.port,
                    message
                )
                
                if response.msg_type == MessageType.RESPONSE_DATA:
                    server_stats = self.serializer.deserialize(response.payload)
                    stats["servers"].append({
                        "server_id": server_info.server_id,
                        **server_stats
                    })
                    
                    emb_stats = server_stats.get("embedding_stats", {})
                    weight_stats = server_stats.get("weight_stats", {})
                    
                    stats["total_embeddings"] += emb_stats.get("num_embeddings", 0)
                    stats["total_weights"] += weight_stats.get("num_layers", 0)
                    stats["total_memory_bytes"] += (
                        emb_stats.get("memory_bytes", 0) +
                        weight_stats.get("memory_bytes", 0)
                    )
                    
            except Exception as e:
                stats["servers"].append({
                    "server_id": server_info.server_id,
                    "error": str(e)
                })
        
        return stats
    
    # S3 Persistence
    
    def save_to_s3(
        self,
        s3_path: str,
        save_model: bool = True,
        save_embeddings: bool = True,
        save_optimizer_states: bool = True
    ):
        """
        Save model weights and embeddings to S3.
        
        Args:
            s3_path: S3 URI (e.g., "s3://bucket/path/checkpoint_001")
            save_model: Whether to save dense model weights
            save_embeddings: Whether to save sparse embeddings
            save_optimizer_states: Whether to save optimizer states
        """
        if not self._started:
            raise RuntimeError("Servers not started")
        
        self.logger.info(f"Saving checkpoint to {s3_path}")
        
        # Import S3 backend
        from pyspark_ps.storage.s3_backend import S3Backend
        from pyspark_ps.storage.checkpoint import CheckpointManager
        
        s3_backend = S3Backend(self.config)
        checkpoint_manager = CheckpointManager(s3_backend)
        
        # Collect state from all servers
        server_states = []
        for server_info in self._server_info:
            message = PSMessage(
                msg_type=MessageType.SAVE_CHECKPOINT,
                client_id=self.client_id,
                payload=self.serializer.serialize({
                    "save_model": save_model,
                    "save_embeddings": save_embeddings,
                    "save_optimizer_states": save_optimizer_states,
                })
            )
            
            response = self._rpc_client.call(
                server_info.host,
                server_info.port,
                message
            )
            
            if response.msg_type == MessageType.RESPONSE_DATA:
                state = self.serializer.deserialize(response.payload)
                server_states.append({
                    "server_id": server_info.server_id,
                    "state": state
                })
        
        # Save to S3
        checkpoint_manager.save(
            s3_path=s3_path,
            server_states=server_states,
            config=self.config.to_dict(),
            metadata={
                "timestamp": time.time(),
                "num_servers": len(self._server_info),
            }
        )
        
        self.logger.info(f"Checkpoint saved to {s3_path}")
    
    def load_from_s3(
        self,
        s3_path: str,
        load_model: bool = True,
        load_embeddings: bool = True,
        load_optimizer_states: bool = True
    ):
        """
        Load model weights and embeddings from S3.
        
        Args:
            s3_path: S3 URI to load from
            load_model: Whether to load dense model weights
            load_embeddings: Whether to load sparse embeddings
            load_optimizer_states: Whether to restore optimizer states
        """
        if not self._started:
            raise RuntimeError("Servers not started")
        
        self.logger.info(f"Loading checkpoint from {s3_path}")
        
        # Import S3 backend
        from pyspark_ps.storage.s3_backend import S3Backend
        from pyspark_ps.storage.checkpoint import CheckpointManager
        
        s3_backend = S3Backend(self.config)
        checkpoint_manager = CheckpointManager(s3_backend)
        
        # Load from S3
        checkpoint_data = checkpoint_manager.load(s3_path)
        server_states = checkpoint_data.get("server_states", [])
        
        # Distribute state to servers
        # This handles re-sharding if number of servers changed
        for server_info in self._server_info:
            # Find matching state or redistribute
            matching_state = None
            for ss in server_states:
                if ss["server_id"] == server_info.server_id:
                    matching_state = ss["state"]
                    break
            
            if matching_state is None:
                # Re-sharding needed - for now, skip
                self.logger.warning(
                    f"No matching state for server {server_info.server_id}, skipping"
                )
                continue
            
            message = PSMessage(
                msg_type=MessageType.LOAD_CHECKPOINT,
                client_id=self.client_id,
                payload=self.serializer.serialize(matching_state)
            )
            
            response = self._rpc_client.call(
                server_info.host,
                server_info.port,
                message
            )
            
            if response.msg_type == MessageType.RESPONSE_ERROR:
                error = self.serializer.deserialize(response.payload)
                raise RuntimeError(
                    f"Failed to load checkpoint on server {server_info.server_id}: {error}"
                )
        
        self.logger.info(f"Checkpoint loaded from {s3_path}")
    
    def list_checkpoints(self, s3_prefix: str) -> List[CheckpointInfo]:
        """
        List available checkpoints under an S3 prefix.
        
        Args:
            s3_prefix: S3 prefix to search
            
        Returns:
            List of CheckpointInfo with path, timestamp, etc.
        """
        from pyspark_ps.storage.s3_backend import S3Backend
        from pyspark_ps.storage.checkpoint import CheckpointManager
        
        s3_backend = S3Backend(self.config)
        checkpoint_manager = CheckpointManager(s3_backend)
        
        return checkpoint_manager.list_checkpoints(s3_prefix)
    
    def delete_checkpoint(self, s3_path: str):
        """Delete a checkpoint from S3."""
        from pyspark_ps.storage.s3_backend import S3Backend
        from pyspark_ps.storage.checkpoint import CheckpointManager
        
        s3_backend = S3Backend(self.config)
        checkpoint_manager = CheckpointManager(s3_backend)
        checkpoint_manager.delete(s3_path)
    
    def checkpoint(self, path: str):
        """
        Save checkpoint (convenience method).
        
        Args:
            path: Local or S3 path
        """
        if path.startswith("s3://"):
            self.save_to_s3(path)
        else:
            # Local checkpoint
            from pyspark_ps.storage.checkpoint import LocalCheckpointManager
            manager = LocalCheckpointManager()
            
            # Collect states
            server_states = []
            for server_info in self._server_info:
                message = PSMessage(
                    msg_type=MessageType.SAVE_CHECKPOINT,
                    client_id=self.client_id,
                    payload=self.serializer.serialize({})
                )
                
                response = self._rpc_client.call(
                    server_info.host,
                    server_info.port,
                    message
                )
                
                if response.msg_type == MessageType.RESPONSE_DATA:
                    state = self.serializer.deserialize(response.payload)
                    server_states.append({
                        "server_id": server_info.server_id,
                        "state": state
                    })
            
            manager.save(path, server_states, self.config.to_dict())
    
    def restore(self, path: str):
        """
        Restore from checkpoint (convenience method).
        
        Args:
            path: Local or S3 path
        """
        if path.startswith("s3://"):
            self.load_from_s3(path)
        else:
            # Local checkpoint
            from pyspark_ps.storage.checkpoint import LocalCheckpointManager
            manager = LocalCheckpointManager()
            
            checkpoint_data = manager.load(path)
            server_states = checkpoint_data.get("server_states", [])
            
            for server_info in self._server_info:
                for ss in server_states:
                    if ss["server_id"] == server_info.server_id:
                        message = PSMessage(
                            msg_type=MessageType.LOAD_CHECKPOINT,
                            client_id=self.client_id,
                            payload=self.serializer.serialize(ss["state"])
                        )
                        
                        self._rpc_client.call(
                            server_info.host,
                            server_info.port,
                            message
                        )
                        break
    
    def step(self):
        """Increment step counter (for auto-decay, etc.)."""
        self._step_count += 1
        
        # Auto decay if configured
        if self.config.auto_decay:
            if self._step_count % self.config.decay_interval_steps == 0:
                self.decay_embeddings(
                    method="multiply",
                    factor=self.config.decay_factor
                )
    
    def __enter__(self):
        self.start_servers()
        return self
    
    def __exit__(self, *args):
        self.shutdown_servers()

