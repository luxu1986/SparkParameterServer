"""Parameter Server node implementation."""

import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from pyspark_ps.communication.protocol import (
    MessageType,
    PSMessage,
    ServerInfo,
)
from pyspark_ps.communication.rpc_handler import RPCServer
from pyspark_ps.communication.serialization import Serializer
from pyspark_ps.server.shard_manager import ShardManager
from pyspark_ps.utils.config import PSConfig
from pyspark_ps.utils.logging import get_logger


class PSServer:
    """
    Parameter Server node running as a service.
    
    Features:
    - Sharded storage for embeddings and weights
    - High-performance optimizer updates
    - Update counting for embeddings
    - Graceful lifecycle management
    - Barrier coordination support
    """
    
    def __init__(
        self,
        server_id: int,
        total_servers: int,
        config: PSConfig,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize PS server.
        
        Args:
            server_id: This server's ID (0 to total_servers-1)
            total_servers: Total number of PS servers
            config: PS configuration
            host: Host to bind to (default: auto-detect)
            port: Port to listen on (default: config.server_port_base + server_id)
        """
        self.server_id = server_id
        self.total_servers = total_servers
        self.config = config
        
        # Network settings
        self.host = host or self._get_host()
        self.port = port or (config.server_port_base + server_id)
        
        self.logger = get_logger(f"ps_server_{server_id}")
        self.serializer = Serializer(
            compression=config.compression,
            compression_algorithm=config.compression_algorithm
        )
        
        # Shard manager handles storage
        self._shard_manager = ShardManager(server_id, total_servers, config)
        
        # RPC server
        self._rpc_server: Optional[RPCServer] = None
        
        # Barrier state
        self._barriers: Dict[str, Dict[str, Any]] = {}
        self._barrier_lock = threading.Lock()
        
        # State
        self._running = False
        self._initialized = False
        self._shutdown_event = threading.Event()
        
        # Statistics
        self._stats = {
            "requests_handled": 0,
            "start_time": None,
            "errors": 0,
        }
        
        # Loss tracking (aggregated from worker gradient pushes)
        self._loss_tracker_lock = threading.Lock()
        self._loss_stats = {
            "total_loss": 0.0,
            "total_samples": 0,
            "batch_count": 0,
            "ema_loss": None,
            "ema_alpha": 0.01,
        }
    
    def _report_loss(self, loss: float, batch_size: int = 1):
        """Record a batch loss for tracking."""
        with self._loss_tracker_lock:
            # Update EMA
            alpha = self._loss_stats["ema_alpha"]
            if self._loss_stats["ema_loss"] is None:
                self._loss_stats["ema_loss"] = loss
            else:
                self._loss_stats["ema_loss"] = alpha * loss + (1 - alpha) * self._loss_stats["ema_loss"]
            
            # Update totals
            self._loss_stats["total_loss"] += loss * batch_size
            self._loss_stats["total_samples"] += batch_size
            self._loss_stats["batch_count"] += 1
    
    def get_loss_stats(self) -> Dict[str, Any]:
        """Get current loss statistics."""
        with self._loss_tracker_lock:
            total_samples = self._loss_stats["total_samples"]
            return {
                "ema_loss": self._loss_stats["ema_loss"],
                "avg_loss": self._loss_stats["total_loss"] / max(total_samples, 1),
                "total_samples": total_samples,
                "batch_count": self._loss_stats["batch_count"],
            }
    
    def _get_host(self) -> str:
        """Get the local host address."""
        try:
            # Get external-facing IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host = s.getsockname()[0]
            s.close()
            return host
        except Exception:
            return "127.0.0.1"
    
    @classmethod
    def discover_and_start(
        cls,
        spark_context,
        config: PSConfig
    ) -> List['PSServer']:
        """
        Discover available nodes and start PS servers.
        
        This is typically called from the driver to start servers
        on executor nodes.
        
        Args:
            spark_context: Spark context
            config: PS configuration
            
        Returns:
            List of server instances
        """
        # For local testing, start servers in threads
        servers = []
        
        for i in range(config.num_servers):
            server = cls(
                server_id=i,
                total_servers=config.num_servers,
                config=config
            )
            server.start()
            servers.append(server)
        
        return servers
    
    def start(self):
        """Start the PS server."""
        if self._running:
            return
        
        self.logger.info(f"Starting PS server {self.server_id} on {self.host}:{self.port}")
        
        # Create RPC server
        self._rpc_server = RPCServer(
            host=self.host,
            port=self.port,
            num_workers=self.config.num_worker_threads,
            max_message_size=self.config.max_message_size
        )
        
        # Register handlers
        self._register_handlers()
        
        # Start RPC server
        self._rpc_server.start()
        
        self._running = True
        self._stats["start_time"] = time.time()
        
        self.logger.info(f"PS server {self.server_id} started")
    
    def _register_handlers(self):
        """Register message handlers."""
        handlers = {
            MessageType.PULL_MODEL: self._handle_pull_model,
            MessageType.PULL_EMBEDDINGS: self._handle_pull_embeddings,
            MessageType.PUSH_WEIGHT_GRADS: self._handle_push_weight_grads,
            MessageType.PUSH_EMBEDDING_GRADS: self._handle_push_embedding_grads,
            MessageType.PUSH_GRADIENTS: self._handle_push_gradients,
            MessageType.BARRIER_ENTER: self._handle_barrier_enter,
            MessageType.BARRIER_CREATE: self._handle_barrier_create,
            MessageType.BARRIER_RELEASE: self._handle_barrier_release,
            MessageType.BARRIER_STATUS: self._handle_barrier_status,
            MessageType.DECAY_EMBEDDINGS: self._handle_decay,
            MessageType.INIT_EMBEDDINGS: self._handle_init_embeddings,
            MessageType.INIT_WEIGHTS: self._handle_init_weights,
            MessageType.GET_STATS: self._handle_get_stats,
            MessageType.PING: self._handle_ping,
            MessageType.SHUTDOWN: self._handle_shutdown,
            MessageType.SAVE_CHECKPOINT: self._handle_save_checkpoint,
            MessageType.LOAD_CHECKPOINT: self._handle_load_checkpoint,
        }
        
        for msg_type, handler in handlers.items():
            self._rpc_server.register_handler(msg_type, handler)
    
    def shutdown(self, grace_period_seconds: float = 30):
        """
        Gracefully shutdown the server.
        
        Args:
            grace_period_seconds: Time to wait for pending operations
        """
        if not self._running:
            return
        
        self.logger.info(f"Shutting down PS server {self.server_id}")
        
        self._shutdown_event.set()
        
        # Stop RPC server
        if self._rpc_server:
            self._rpc_server.stop(timeout=grace_period_seconds)
        
        self._running = False
        
        self.logger.info(f"PS server {self.server_id} shutdown complete")
    
    def get_server_info(self) -> ServerInfo:
        """Get server information for clients."""
        return ServerInfo(
            server_id=self.server_id,
            host=self.host,
            port=self.port,
            status="running" if self._running else "stopped",
            metadata={
                "embedding_count": self._shard_manager.get_embedding_count(),
                "weight_count": self._shard_manager.get_weight_count(),
            }
        )
    
    # Handler implementations
    
    def _handle_pull_model(self, message: PSMessage) -> PSMessage:
        """Handle pull model request."""
        try:
            self._stats["requests_handled"] += 1
            
            request = self.serializer.deserialize(message.payload)
            layer_names = request.get("layer_names")
            
            weights = self._shard_manager.get_weights(layer_names)
            
            response_data = self.serializer.serialize(weights)
            return message.create_response(
                MessageType.RESPONSE_DATA,
                payload=response_data
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"Error in pull_model: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_pull_embeddings(self, message: PSMessage) -> PSMessage:
        """Handle pull embeddings request."""
        try:
            self._stats["requests_handled"] += 1
            
            request = self.serializer.deserialize(message.payload)
            token_ids = request.get("token_ids", [])
            create_if_missing = request.get("create_if_missing", True)
            
            embeddings, found_ids = self._shard_manager.get_embeddings(
                token_ids, create_if_missing
            )
            
            response_data = self.serializer.serialize({
                "embeddings": embeddings,
                "token_ids": found_ids
            })
            
            return message.create_response(
                MessageType.RESPONSE_DATA,
                payload=response_data
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"Error in pull_embeddings: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_push_weight_grads(self, message: PSMessage) -> PSMessage:
        """Handle push weight gradients request."""
        try:
            self._stats["requests_handled"] += 1
            
            gradients = self.serializer.deserialize(message.payload)
            self._shard_manager.update_weights(gradients)
            
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"Error in push_weight_grads: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_push_embedding_grads(self, message: PSMessage) -> PSMessage:
        """Handle push embedding gradients request."""
        try:
            self._stats["requests_handled"] += 1
            
            request = self.serializer.deserialize(message.payload)
            gradients = request.get("gradients", {})
            increment_count = request.get("increment_count", True)
            
            self._shard_manager.update_embeddings(gradients, increment_count)
            
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"Error in push_embedding_grads: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_push_gradients(self, message: PSMessage) -> PSMessage:
        """Handle combined push gradients request."""
        try:
            self._stats["requests_handled"] += 1
            
            request = self.serializer.deserialize(message.payload)
            
            weight_grads = request.get("weight_gradients", {})
            embedding_grads = request.get("embedding_gradients", {})
            
            # Track loss if provided
            batch_loss = request.get("batch_loss")
            batch_size = request.get("batch_size", 1)
            if batch_loss is not None:
                self._report_loss(batch_loss, batch_size)
            
            if weight_grads:
                self._shard_manager.update_weights(weight_grads)
            
            if embedding_grads:
                self._shard_manager.update_embeddings(embedding_grads)
            
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"Error in push_gradients: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_barrier_create(self, message: PSMessage) -> PSMessage:
        """Handle barrier creation request."""
        try:
            request = self.serializer.deserialize(message.payload)
            name = request["name"]
            num_workers = request["num_workers"]
            
            with self._barrier_lock:
                self._barriers[name] = {
                    "num_workers": num_workers,
                    "entered": set(),
                    "released": False,
                    "created_at": time.time(),
                }
            
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_barrier_enter(self, message: PSMessage) -> PSMessage:
        """Handle barrier enter request."""
        try:
            request = self.serializer.deserialize(message.payload)
            name = request["name"]
            client_id = message.client_id
            
            with self._barrier_lock:
                if name not in self._barriers:
                    return message.create_response(
                        MessageType.RESPONSE_ERROR,
                        payload=self.serializer.serialize({"error": f"Barrier {name} not found"})
                    )
                
                barrier = self._barriers[name]
                barrier["entered"].add(client_id)
                
                all_entered = len(barrier["entered"]) >= barrier["num_workers"]
                released = barrier["released"]
                
            return message.create_response(
                MessageType.RESPONSE_DATA,
                payload=self.serializer.serialize({
                    "all_entered": all_entered,
                    "released": released,
                    "num_entered": len(barrier["entered"]),
                })
            )
            
        except Exception as e:
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_barrier_release(self, message: PSMessage) -> PSMessage:
        """Handle barrier release request."""
        try:
            request = self.serializer.deserialize(message.payload)
            name = request["name"]
            
            with self._barrier_lock:
                if name in self._barriers:
                    self._barriers[name]["released"] = True
                    # Clean up old barriers
                    del self._barriers[name]
            
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_barrier_status(self, message: PSMessage) -> PSMessage:
        """Handle barrier status request."""
        try:
            request = self.serializer.deserialize(message.payload)
            name = request["name"]
            
            with self._barrier_lock:
                if name not in self._barriers:
                    return message.create_response(
                        MessageType.RESPONSE_DATA,
                        payload=self.serializer.serialize({"exists": False})
                    )
                
                barrier = self._barriers[name]
                
            return message.create_response(
                MessageType.RESPONSE_DATA,
                payload=self.serializer.serialize({
                    "exists": True,
                    "num_workers": barrier["num_workers"],
                    "num_entered": len(barrier["entered"]),
                    "released": barrier["released"],
                })
            )
            
        except Exception as e:
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_decay(self, message: PSMessage) -> PSMessage:
        """Handle decay operation."""
        try:
            self._stats["requests_handled"] += 1
            
            request = self.serializer.deserialize(message.payload)
            method = request.get("method", "multiply")
            
            if method == "multiply":
                factor = request.get("factor", 0.99)
                self._shard_manager.decay_embeddings(factor)
                result = {"method": "multiply", "factor": factor}
                
            elif method == "prune":
                min_count = request.get("min_count", 5)
                pruned = self._shard_manager.prune_embeddings(min_count)
                result = {"method": "prune", "min_count": min_count, "pruned": pruned}
                
            else:
                return message.create_response(
                    MessageType.RESPONSE_ERROR,
                    payload=self.serializer.serialize({"error": f"Unknown method: {method}"})
                )
            
            return message.create_response(
                MessageType.RESPONSE_DATA,
                payload=self.serializer.serialize(result)
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(f"Error in decay: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_init_embeddings(self, message: PSMessage) -> PSMessage:
        """Handle embedding initialization."""
        try:
            # Embeddings are initialized on-demand, so this is a no-op
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_init_weights(self, message: PSMessage) -> PSMessage:
        """Handle weight initialization."""
        try:
            request = self.serializer.deserialize(message.payload)
            weight_shapes = request.get("shapes", {})
            init_strategy = request.get("init_strategy", "normal")
            init_scale = request.get("init_scale", 0.01)
            
            # Convert shape lists to tuples
            weight_shapes = {k: tuple(v) for k, v in weight_shapes.items()}
            
            self._shard_manager.init_weights(weight_shapes, init_strategy, init_scale)
            
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            self.logger.error(f"Error in init_weights: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_get_stats(self, message: PSMessage) -> PSMessage:
        """Handle stats request."""
        try:
            stats = self._shard_manager.get_stats()
            stats.update({
                "server_stats": self._stats.copy(),
                "uptime": time.time() - self._stats["start_time"] if self._stats["start_time"] else 0,
                "loss_stats": self.get_loss_stats(),
            })
            
            return message.create_response(
                MessageType.RESPONSE_DATA,
                payload=self.serializer.serialize(stats)
            )
            
        except Exception as e:
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_ping(self, message: PSMessage) -> PSMessage:
        """Handle ping request."""
        return message.create_response(
            MessageType.RESPONSE_OK,
            payload=self.serializer.serialize({
                "server_id": self.server_id,
                "status": "running",
                "timestamp": time.time(),
            })
        )
    
    def _handle_shutdown(self, message: PSMessage) -> PSMessage:
        """Handle shutdown request."""
        # Schedule shutdown in background
        threading.Thread(
            target=lambda: self.shutdown(grace_period_seconds=5),
            daemon=True
        ).start()
        
        return message.create_response(MessageType.RESPONSE_OK)
    
    def _handle_save_checkpoint(self, message: PSMessage) -> PSMessage:
        """Handle checkpoint save request."""
        try:
            request = self.serializer.deserialize(message.payload)
            
            # Get state from shard manager
            state = self._shard_manager.get_state()
            
            return message.create_response(
                MessageType.RESPONSE_DATA,
                payload=self.serializer.serialize(state)
            )
            
        except Exception as e:
            self.logger.error(f"Error in save_checkpoint: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    def _handle_load_checkpoint(self, message: PSMessage) -> PSMessage:
        """Handle checkpoint load request."""
        try:
            state = self.serializer.deserialize(message.payload)
            self._shard_manager.set_state(state)
            
            return message.create_response(MessageType.RESPONSE_OK)
            
        except Exception as e:
            self.logger.error(f"Error in load_checkpoint: {e}")
            return message.create_response(
                MessageType.RESPONSE_ERROR,
                payload=self.serializer.serialize({"error": str(e)})
            )
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
    
    @property
    def shard_manager(self) -> ShardManager:
        """Access the shard manager."""
        return self._shard_manager

