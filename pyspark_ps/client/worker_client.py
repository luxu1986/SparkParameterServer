"""Worker client for executors to communicate with PS servers."""

import uuid
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from pyspark_ps.communication.protocol import MessageType, PSMessage, ServerInfo
from pyspark_ps.communication.rpc_handler import RPCClient
from pyspark_ps.communication.serialization import Serializer
from pyspark_ps.utils.config import PSConfig
from pyspark_ps.utils.sharding import ConsistentHashRing
from pyspark_ps.utils.logging import get_logger


class PSWorkerClient:
    """
    Worker Parameter Server Client running on Spark EXECUTOR nodes.
    
    Responsibilities:
    - Pull model weights and embeddings from servers
    - Push gradients to servers
    - Participate in barrier synchronization
    
    NOTE: This client runs on executors, NOT on driver.
    One instance per Spark partition/task.
    """
    
    def __init__(
        self,
        server_info: List[ServerInfo],
        config: PSConfig,
        client_id: Optional[str] = None
    ):
        """
        Initialize worker client.
        
        Args:
            server_info: List of server endpoints (broadcast from driver)
            config: PS configuration (broadcast from driver)
            client_id: Unique client ID (auto-generated if not provided)
        """
        self.server_info = server_info
        self.config = config
        self.client_id = client_id or str(uuid.uuid4())
        
        self.logger = get_logger(f"worker_client_{self.client_id[:8]}")
        
        # Server mapping
        self._servers = {s.server_id: s for s in server_info}
        self._server_addrs = [(s.host, s.port) for s in server_info]
        
        # Sharding
        self._hash_ring = ConsistentHashRing(
            len(server_info),
            virtual_nodes=config.virtual_nodes_per_server
        )
        
        # Communication
        self._rpc_client = RPCClient(
            timeout=config.timeout_seconds,
            max_message_size=config.max_message_size
        )
        self.serializer = Serializer(
            compression=config.compression,
            compression_algorithm=config.compression_algorithm
        )
        
        # Thread pool for parallel server communication
        self._executor = ThreadPoolExecutor(max_workers=len(server_info) * 2)
        
        self._closed = False
    
    def enter_barrier(self, name: str, timeout_seconds: Optional[float] = None):
        """
        Enter a named barrier and wait for release.
        
        Blocks until main client releases the barrier.
        Used for synchronization during decay, checkpointing, etc.
        
        Args:
            name: Barrier name
            timeout_seconds: Timeout (uses config default if not specified)
        """
        timeout = timeout_seconds or self.config.timeout_seconds
        coordinator = self._servers[0]
        
        import time
        start_time = time.time()
        
        # Enter barrier
        message = PSMessage(
            msg_type=MessageType.BARRIER_ENTER,
            client_id=self.client_id,
            payload=self.serializer.serialize({"name": name})
        )
        
        response = self._rpc_client.call(
            coordinator.host,
            coordinator.port,
            message
        )
        
        if response.msg_type == MessageType.RESPONSE_ERROR:
            error = self.serializer.deserialize(response.payload)
            raise RuntimeError(f"Failed to enter barrier: {error}")
        
        # Poll for release
        poll_interval = 0.1
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Barrier '{name}' timeout after {elapsed:.1f}s")
            
            # Check status
            status_msg = PSMessage(
                msg_type=MessageType.BARRIER_STATUS,
                client_id=self.client_id,
                payload=self.serializer.serialize({"name": name})
            )
            
            status_resp = self._rpc_client.call(
                coordinator.host,
                coordinator.port,
                status_msg
            )
            
            if status_resp.msg_type == MessageType.RESPONSE_DATA:
                status = self.serializer.deserialize(status_resp.payload)
                if status.get("released", False) or not status.get("exists", True):
                    # Barrier released or doesn't exist anymore (already processed)
                    return
            
            time.sleep(min(poll_interval, timeout - elapsed))
            poll_interval = min(poll_interval * 1.5, 1.0)
    
    def pull_model(self, layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Pull model weights from servers.
        
        Since weights are replicated, we can pull from any server.
        
        Args:
            layer_names: Optional list of layer names to pull (None = all)
            
        Returns:
            Dict mapping layer names to weight arrays
        """
        if self._closed:
            raise RuntimeError("Client is closed")
        
        # Pull from first server (weights are replicated)
        server = self._servers[0]
        
        message = PSMessage(
            msg_type=MessageType.PULL_MODEL,
            client_id=self.client_id,
            payload=self.serializer.serialize({"layer_names": layer_names})
        )
        
        response = self._rpc_client.call(server.host, server.port, message)
        
        if response.msg_type == MessageType.RESPONSE_ERROR:
            error = self.serializer.deserialize(response.payload)
            raise RuntimeError(f"Failed to pull model: {error}")
        
        return self.serializer.deserialize(response.payload)
    
    def pull_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """
        Pull embeddings for specific tokens.
        
        Routes requests to appropriate servers based on token hash.
        For missing tokens, servers initialize with configurable strategy.
        
        Args:
            token_ids: List of integer token IDs
            
        Returns:
            np.ndarray of shape (len(token_ids), embedding_dim)
        """
        if self._closed:
            raise RuntimeError("Client is closed")
        
        if not token_ids:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        
        # Partition tokens by server
        server_tokens = self._hash_ring.get_servers_batch(token_ids)
        
        # Prepare requests for each server
        requests = []
        server_order = []
        
        for server_id, tokens in server_tokens.items():
            if not tokens:
                continue
            
            server = self._servers[server_id]
            message = PSMessage(
                msg_type=MessageType.PULL_EMBEDDINGS,
                client_id=self.client_id,
                payload=self.serializer.serialize({
                    "token_ids": tokens,
                    "create_if_missing": True
                })
            )
            requests.append((server.host, server.port, message))
            server_order.append((server_id, tokens))
        
        # Execute requests in parallel
        responses = self._rpc_client.call_batch(requests, self._executor)
        
        # Aggregate results in original order
        token_to_embedding = {}
        
        for (server_id, tokens), response in zip(server_order, responses):
            if response.msg_type == MessageType.RESPONSE_ERROR:
                error = self.serializer.deserialize(response.payload)
                raise RuntimeError(f"Failed to pull embeddings from server {server_id}: {error}")
            
            data = self.serializer.deserialize(response.payload)
            embeddings = data["embeddings"]
            found_ids = data["token_ids"]
            
            for i, tid in enumerate(found_ids):
                token_to_embedding[tid] = embeddings[i]
        
        # Reconstruct in original order
        result = np.zeros((len(token_ids), self.config.embedding_dim), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            if tid in token_to_embedding:
                result[i] = token_to_embedding[tid]
        
        return result
    
    def push_gradients(
        self,
        weight_gradients: Dict[str, np.ndarray],
        embedding_gradients: Dict[int, np.ndarray],
        batch_loss: float = None,
        batch_size: int = 1
    ):
        """
        Push gradients to server nodes.
        
        Args:
            weight_gradients: Dense gradients for model layers
            embedding_gradients: Sparse gradients for embeddings (token_id -> gradient)
            batch_loss: Optional batch loss for real-time tracking
            batch_size: Batch size for loss weighting
        """
        if self._closed:
            raise RuntimeError("Client is closed")
        
        # Push weights to first server (they're replicated)
        if weight_gradients:
            self.push_weight_gradients(weight_gradients, batch_loss, batch_size)
        
        # Push embeddings to appropriate servers
        if embedding_gradients:
            self.push_embedding_gradients(embedding_gradients)
    
    def push_weight_gradients(
        self,
        weight_gradients: Dict[str, np.ndarray],
        batch_loss: float = None,
        batch_size: int = 1
    ):
        """
        Push weight gradients to servers.
        
        Weights are replicated, so we push to all servers.
        Each server aggregates and applies updates.
        
        Args:
            weight_gradients: Dict mapping layer name to gradient array
            batch_loss: Optional batch loss for real-time tracking
            batch_size: Batch size for loss weighting
        """
        if self._closed:
            raise RuntimeError("Client is closed")
        
        if not weight_gradients:
            return
        
        # Build payload with gradients and optional loss
        payload = {
            "weight_gradients": weight_gradients,
            "batch_loss": batch_loss,
            "batch_size": batch_size,
        }
        
        # For simplicity, push to all servers
        # In production, could aggregate first or use AllReduce
        message = PSMessage(
            msg_type=MessageType.PUSH_GRADIENTS,  # Use combined message type
            client_id=self.client_id,
            payload=self.serializer.serialize(payload)
        )
        
        requests = [
            (server.host, server.port, message)
            for server in self._servers.values()
        ]
        
        responses = self._rpc_client.call_batch(requests, self._executor)
        
        # Check for errors
        for i, response in enumerate(responses):
            if response.msg_type == MessageType.RESPONSE_ERROR:
                error = self.serializer.deserialize(response.payload)
                self.logger.warning(f"Error pushing weight gradients to server {i}: {error}")
    
    def push_embedding_gradients(self, embedding_gradients: Dict[int, np.ndarray]):
        """
        Push embedding gradients to appropriate servers.
        
        Routes gradients to servers based on token hash.
        
        Args:
            embedding_gradients: Dict mapping token_id to gradient array
        """
        if self._closed:
            raise RuntimeError("Client is closed")
        
        if not embedding_gradients:
            return
        
        # Partition gradients by server
        server_grads: Dict[int, Dict[int, np.ndarray]] = {
            i: {} for i in range(len(self._servers))
        }
        
        for token_id, gradient in embedding_gradients.items():
            server_id = self._hash_ring.get_server(token_id)
            server_grads[server_id][token_id] = gradient
        
        # Prepare requests
        requests = []
        server_order = []
        
        for server_id, grads in server_grads.items():
            if not grads:
                continue
            
            server = self._servers[server_id]
            message = PSMessage(
                msg_type=MessageType.PUSH_EMBEDDING_GRADS,
                client_id=self.client_id,
                payload=self.serializer.serialize({
                    "gradients": grads,
                    "increment_count": True
                })
            )
            requests.append((server.host, server.port, message))
            server_order.append(server_id)
        
        # Execute in parallel
        responses = self._rpc_client.call_batch(requests, self._executor)
        
        # Check for errors
        for server_id, response in zip(server_order, responses):
            if response.msg_type == MessageType.RESPONSE_ERROR:
                error = self.serializer.deserialize(response.payload)
                self.logger.warning(f"Error pushing embedding gradients to server {server_id}: {error}")
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings across all servers."""
        total = 0
        for server in self._servers.values():
            message = PSMessage(
                msg_type=MessageType.GET_STATS,
                client_id=self.client_id,
                payload=b""
            )
            
            try:
                response = self._rpc_client.call(server.host, server.port, message)
                if response.msg_type == MessageType.RESPONSE_DATA:
                    stats = self.serializer.deserialize(response.payload)
                    total += stats.get("embedding_stats", {}).get("num_embeddings", 0)
            except Exception as e:
                self.logger.warning(f"Failed to get stats from server {server.server_id}: {e}")
        
        return total
    
    def close(self):
        """Release client resources."""
        if self._closed:
            return
        
        self._closed = True
        self._executor.shutdown(wait=True)
        self._rpc_client.close()
        
        self.logger.debug(f"Worker client {self.client_id[:8]} closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

