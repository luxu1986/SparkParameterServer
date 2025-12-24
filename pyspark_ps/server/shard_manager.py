"""Shard manager for distributed data across PS servers."""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from pyspark_ps.utils.sharding import ConsistentHashRing
from pyspark_ps.server.embedding_store import EmbeddingStore
from pyspark_ps.server.weight_store import WeightStore
from pyspark_ps.utils.config import PSConfig


class ShardManager:
    """
    Manages data sharding across multiple PS servers.
    
    Responsibilities:
    - Determine which server owns which tokens
    - Route requests to appropriate shards
    - Handle rebalancing when servers change
    """
    
    def __init__(
        self,
        server_id: int,
        total_servers: int,
        config: PSConfig
    ):
        """
        Initialize shard manager.
        
        Args:
            server_id: This server's ID (0 to total_servers-1)
            total_servers: Total number of PS servers
            config: PS configuration
        """
        self.server_id = server_id
        self.total_servers = total_servers
        self.config = config
        
        # Consistent hashing for token assignment
        self._hash_ring = ConsistentHashRing(
            total_servers,
            virtual_nodes=config.virtual_nodes_per_server
        )
        
        # Local stores
        self._embedding_store = EmbeddingStore(
            embedding_dim=config.embedding_dim,
            init_strategy=config.embedding_init,
            init_scale=config.embedding_init_scale,
            optimizer_name=config.embedding_optimizer,
            optimizer_config=config.get_optimizer_config(config.embedding_optimizer),
            max_embeddings=config.max_embeddings_per_shard
        )
        
        self._weight_store = WeightStore(
            optimizer_name=config.weight_optimizer,
            optimizer_config=config.get_optimizer_config(config.weight_optimizer)
        )
    
    def owns_token(self, token_id: int) -> bool:
        """Check if this server owns the given token."""
        return self._hash_ring.get_server(token_id) == self.server_id
    
    def get_owner(self, token_id: int) -> int:
        """Get the server ID that owns a token."""
        return self._hash_ring.get_server(token_id)
    
    def filter_owned_tokens(self, token_ids: List[int]) -> List[int]:
        """Filter list to only tokens owned by this server."""
        return [tid for tid in token_ids if self.owns_token(tid)]
    
    def partition_tokens(self, token_ids: List[int]) -> Dict[int, List[int]]:
        """
        Partition tokens by owning server.
        
        Returns:
            Dict mapping server_id -> list of tokens
        """
        return self._hash_ring.get_servers_batch(token_ids)
    
    # Embedding operations
    
    def get_embeddings(
        self,
        token_ids: List[int],
        create_if_missing: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get embeddings for tokens owned by this server.
        
        Args:
            token_ids: List of token IDs (will be filtered to owned tokens)
            create_if_missing: Create new embeddings for missing tokens
            
        Returns:
            Tuple of (embeddings array, list of found token IDs)
        """
        owned_tokens = self.filter_owned_tokens(token_ids)
        return self._embedding_store.get_batch(owned_tokens, create_if_missing)
    
    def update_embeddings(
        self,
        gradients: Dict[int, np.ndarray],
        increment_count: bool = True
    ):
        """
        Update embeddings with gradients.
        
        Args:
            gradients: Dict mapping token_id -> gradient (filtered to owned)
            increment_count: Whether to increment update counts
        """
        owned_grads = {
            tid: grad for tid, grad in gradients.items()
            if self.owns_token(tid)
        }
        self._embedding_store.update_batch(owned_grads, increment_count)
    
    # Weight operations
    
    def init_weights(
        self,
        weight_shapes: Dict[str, Tuple[int, ...]],
        init_strategy: str = "normal",
        init_scale: float = 0.01
    ):
        """
        Initialize weight tensors.
        
        For now, all servers maintain all weights (replicated).
        Could be extended for weight sharding.
        """
        for name, shape in weight_shapes.items():
            self._weight_store.init_weights(name, shape, init_strategy, init_scale)
    
    def get_weights(self, layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get weight tensors."""
        if layer_names is None:
            return self._weight_store.get_all()
        return self._weight_store.get_batch(layer_names)
    
    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """Update weights with gradients."""
        self._weight_store.update_batch(gradients)
    
    # Decay and pruning
    
    def decay_embeddings(self, factor: float, decay_optimizer: bool = False):
        """Apply decay to embeddings."""
        self._embedding_store.decay(factor, decay_optimizer)
    
    def prune_embeddings(self, min_count: int) -> int:
        """Prune infrequently accessed embeddings."""
        return self._embedding_store.prune(min_count)
    
    # Statistics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get shard statistics."""
        return {
            "server_id": self.server_id,
            "total_servers": self.total_servers,
            "embedding_stats": self._embedding_store.get_stats(),
            "weight_stats": self._weight_store.get_stats(),
        }
    
    def get_embedding_count(self) -> int:
        """Get number of embeddings in this shard."""
        return len(self._embedding_store)
    
    def get_weight_count(self) -> int:
        """Get number of weight tensors."""
        return len(self._weight_store)
    
    # State management
    
    def get_state(self) -> Dict[str, Any]:
        """Get full shard state for checkpointing."""
        return {
            "server_id": self.server_id,
            "total_servers": self.total_servers,
            "embedding_state": self._embedding_store.get_state(),
            "weight_state": self._weight_store.get_state(),
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore shard state from checkpoint."""
        self._embedding_store.set_state(state["embedding_state"])
        self._weight_store.set_state(state["weight_state"])
    
    def clear(self):
        """Clear all data."""
        self._embedding_store.clear()
        self._weight_store.clear()
    
    @property
    def embedding_store(self) -> EmbeddingStore:
        """Access underlying embedding store."""
        return self._embedding_store
    
    @property
    def weight_store(self) -> WeightStore:
        """Access underlying weight store."""
        return self._weight_store

