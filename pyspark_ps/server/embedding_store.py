"""Sparse embedding storage for Parameter Server."""

import threading
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from pyspark_ps.server.update_counter import UpdateCounter
from pyspark_ps.optimizers import create_optimizer, BaseOptimizer


class EmbeddingStore:
    """
    Thread-safe storage for sparse embeddings.
    
    Features:
    - Dynamic allocation of embeddings
    - Per-embedding optimizer states
    - Update counting for decay/pruning
    - Configurable initialization strategies
    """
    
    def __init__(
        self,
        embedding_dim: int,
        init_strategy: str = "normal",
        init_scale: float = 0.01,
        optimizer_name: str = "adagrad",
        optimizer_config: Optional[Dict[str, Any]] = None,
        max_embeddings: Optional[int] = None
    ):
        """
        Initialize embedding store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            init_strategy: Initialization strategy ("zeros", "random", "normal")
            init_scale: Scale for random/normal initialization
            optimizer_name: Name of optimizer to use
            optimizer_config: Optimizer configuration
            max_embeddings: Maximum number of embeddings (None = unlimited)
        """
        self.embedding_dim = embedding_dim
        self.init_strategy = init_strategy
        self.init_scale = init_scale
        self.max_embeddings = max_embeddings
        
        # Storage
        self._embeddings: Dict[int, np.ndarray] = {}
        self._lock = threading.RLock()
        
        # Update tracking
        self._update_counter = UpdateCounter(default_count=1)
        
        # Optimizer
        optimizer_config = optimizer_config or {}
        self._optimizer = create_optimizer(optimizer_name, **optimizer_config)
        
        # Statistics
        self._stats = {
            "total_lookups": 0,
            "total_updates": 0,
            "total_misses": 0,
            "total_creates": 0,
        }
    
    def _init_embedding(self) -> np.ndarray:
        """Create a new embedding with the configured initialization."""
        if self.init_strategy == "zeros":
            return np.zeros(self.embedding_dim, dtype=np.float32)
        elif self.init_strategy == "random":
            return (np.random.random(self.embedding_dim).astype(np.float32) - 0.5) * 2 * self.init_scale
        elif self.init_strategy == "normal":
            return np.random.randn(self.embedding_dim).astype(np.float32) * self.init_scale
        else:
            raise ValueError(f"Unknown init strategy: {self.init_strategy}")
    
    def get(self, token_id: int, create_if_missing: bool = True) -> Optional[np.ndarray]:
        """
        Get embedding for a token.
        
        Args:
            token_id: Token identifier
            create_if_missing: Create new embedding if not found
            
        Returns:
            Embedding array or None if not found and create_if_missing=False
        """
        with self._lock:
            self._stats["total_lookups"] += 1
            
            if token_id in self._embeddings:
                return self._embeddings[token_id].copy()
            
            self._stats["total_misses"] += 1
            
            if not create_if_missing:
                return None
            
            # Check capacity
            if self.max_embeddings and len(self._embeddings) >= self.max_embeddings:
                # Could implement LRU eviction here
                return None
            
            # Create new embedding
            embedding = self._init_embedding()
            self._embeddings[token_id] = embedding
            self._update_counter.set_count(token_id, 1)
            self._stats["total_creates"] += 1
            
            return embedding.copy()
    
    def get_batch(
        self,
        token_ids: List[int],
        create_if_missing: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get embeddings for multiple tokens.
        
        Args:
            token_ids: List of token identifiers
            create_if_missing: Create new embeddings if not found
            
        Returns:
            Tuple of (embeddings array, list of found token_ids)
        """
        with self._lock:
            self._stats["total_lookups"] += len(token_ids)
            
            embeddings = []
            found_ids = []
            
            for token_id in token_ids:
                if token_id in self._embeddings:
                    embeddings.append(self._embeddings[token_id].copy())
                    found_ids.append(token_id)
                elif create_if_missing:
                    if self.max_embeddings and len(self._embeddings) >= self.max_embeddings:
                        self._stats["total_misses"] += 1
                        continue
                    
                    embedding = self._init_embedding()
                    self._embeddings[token_id] = embedding
                    self._update_counter.set_count(token_id, 1)
                    embeddings.append(embedding.copy())
                    found_ids.append(token_id)
                    self._stats["total_creates"] += 1
                else:
                    self._stats["total_misses"] += 1
            
            if embeddings:
                return np.stack(embeddings), found_ids
            else:
                return np.array([]).reshape(0, self.embedding_dim), []
    
    def update(
        self,
        token_id: int,
        gradient: np.ndarray,
        increment_count: bool = True
    ):
        """
        Update embedding with gradient.
        
        Args:
            token_id: Token identifier
            gradient: Gradient array
            increment_count: Whether to increment update count
        """
        with self._lock:
            self._stats["total_updates"] += 1
            
            if token_id not in self._embeddings:
                # Create embedding if it doesn't exist
                self._embeddings[token_id] = self._init_embedding()
                self._update_counter.set_count(token_id, 1)
                self._stats["total_creates"] += 1
            
            # Apply optimizer update
            param_id = f"emb_{token_id}"
            self._embeddings[token_id] = self._optimizer.update(
                param_id,
                self._embeddings[token_id],
                gradient
            )
            
            if increment_count:
                self._update_counter.increment(token_id)
    
    def update_batch(
        self,
        gradients: Dict[int, np.ndarray],
        increment_count: bool = True
    ):
        """
        Update multiple embeddings with gradients.
        
        Args:
            gradients: Dict mapping token_id -> gradient
            increment_count: Whether to increment update counts
        """
        with self._lock:
            self._stats["total_updates"] += len(gradients)
            
            for token_id, gradient in gradients.items():
                if token_id not in self._embeddings:
                    self._embeddings[token_id] = self._init_embedding()
                    self._update_counter.set_count(token_id, 1)
                    self._stats["total_creates"] += 1
                
                param_id = f"emb_{token_id}"
                self._embeddings[token_id] = self._optimizer.update(
                    param_id,
                    self._embeddings[token_id],
                    gradient
                )
            
            if increment_count:
                self._update_counter.increment_batch(list(gradients.keys()))
    
    def set(self, token_id: int, embedding: np.ndarray):
        """
        Set embedding directly (bypassing optimizer).
        
        Args:
            token_id: Token identifier
            embedding: Embedding array
        """
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Embedding shape mismatch: expected ({self.embedding_dim},), "
                f"got {embedding.shape}"
            )
        
        with self._lock:
            self._embeddings[token_id] = embedding.astype(np.float32).copy()
            if token_id not in self._update_counter:
                self._update_counter.set_count(token_id, 1)
    
    def remove(self, token_id: int):
        """Remove an embedding."""
        with self._lock:
            if token_id in self._embeddings:
                del self._embeddings[token_id]
                self._update_counter.remove(token_id)
                self._optimizer.remove_state(f"emb_{token_id}")
    
    def remove_batch(self, token_ids: List[int]):
        """Remove multiple embeddings."""
        with self._lock:
            for token_id in token_ids:
                if token_id in self._embeddings:
                    del self._embeddings[token_id]
                    self._optimizer.remove_state(f"emb_{token_id}")
            self._update_counter.remove_batch(token_ids)
    
    def decay(self, factor: float, decay_optimizer: bool = False):
        """
        Apply multiplicative decay to all embeddings.
        
        Args:
            factor: Decay factor (0 < factor <= 1)
            decay_optimizer: Whether to decay optimizer states too
        """
        if not 0 < factor <= 1:
            raise ValueError("Decay factor must be in (0, 1]")
        
        with self._lock:
            for token_id in self._embeddings:
                self._embeddings[token_id] *= factor
            
            self._update_counter.decay(factor)
            
            if decay_optimizer and hasattr(self._optimizer, 'decay_accumulator'):
                self._optimizer.decay_accumulator(factor)
    
    def prune(self, min_count: int) -> int:
        """
        Remove embeddings with update count below threshold.
        
        Args:
            min_count: Minimum update count threshold
            
        Returns:
            Number of pruned embeddings
        """
        with self._lock:
            pruned_ids = self._update_counter.prune(min_count)
            
            for token_id in pruned_ids:
                if token_id in self._embeddings:
                    del self._embeddings[token_id]
                    self._optimizer.remove_state(f"emb_{token_id}")
            
            return len(pruned_ids)
    
    def get_update_count(self, token_id: int) -> float:
        """Get update count for a token."""
        return self._update_counter.get_count(token_id)
    
    def get_all_token_ids(self) -> Set[int]:
        """Get all stored token IDs."""
        with self._lock:
            return set(self._embeddings.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            counter_stats = self._update_counter.get_stats()
            
            return {
                "num_embeddings": len(self._embeddings),
                "embedding_dim": self.embedding_dim,
                "memory_bytes": len(self._embeddings) * self.embedding_dim * 4,
                **self._stats,
                "update_counter": counter_stats,
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state for checkpointing."""
        with self._lock:
            return {
                "embeddings": {k: v.copy() for k, v in self._embeddings.items()},
                "update_counts": self._update_counter.get_all_counts(),
                "optimizer_state": self._optimizer.get_state(),
                "config": {
                    "embedding_dim": self.embedding_dim,
                    "init_strategy": self.init_strategy,
                    "init_scale": self.init_scale,
                },
            }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        with self._lock:
            self._embeddings = {k: v.copy() for k, v in state["embeddings"].items()}
            self._update_counter.set_all_counts(state["update_counts"])
            self._optimizer.set_state(state["optimizer_state"])
    
    def clear(self):
        """Clear all embeddings and reset state."""
        with self._lock:
            self._embeddings.clear()
            self._update_counter.clear()
            self._optimizer.reset()
            self._stats = {k: 0 for k in self._stats}
    
    def __len__(self) -> int:
        """Return number of stored embeddings."""
        with self._lock:
            return len(self._embeddings)
    
    def __contains__(self, token_id: int) -> bool:
        """Check if token has a stored embedding."""
        with self._lock:
            return token_id in self._embeddings

