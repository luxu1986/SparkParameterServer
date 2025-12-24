"""Update counter for tracking embedding access patterns."""

import threading
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import numpy as np


class UpdateCounter:
    """
    Thread-safe counter for tracking embedding update frequencies.
    
    Used for:
    - Pruning infrequently accessed embeddings
    - Decay operations
    - Statistics and monitoring
    """
    
    def __init__(self, default_count: int = 0):
        """
        Initialize update counter.
        
        Args:
            default_count: Default count for new entries
        """
        self._counts: Dict[int, float] = {}
        self._lock = threading.RLock()
        self._default_count = default_count
        
        # Statistics
        self._total_updates = 0
        self._total_accesses = 0
    
    def increment(self, token_id: int, amount: int = 1):
        """
        Increment count for a token.
        
        Args:
            token_id: Token identifier
            amount: Amount to increment
        """
        with self._lock:
            if token_id not in self._counts:
                self._counts[token_id] = self._default_count
            self._counts[token_id] += amount
            self._total_updates += 1
    
    def increment_batch(self, token_ids: List[int], amount: int = 1):
        """
        Increment counts for multiple tokens.
        
        Args:
            token_ids: List of token identifiers
            amount: Amount to increment for each
        """
        with self._lock:
            for token_id in token_ids:
                if token_id not in self._counts:
                    self._counts[token_id] = self._default_count
                self._counts[token_id] += amount
            self._total_updates += len(token_ids)
    
    def get_count(self, token_id: int) -> float:
        """
        Get count for a token.
        
        Args:
            token_id: Token identifier
            
        Returns:
            Count value (0 if not found)
        """
        with self._lock:
            self._total_accesses += 1
            return self._counts.get(token_id, 0)
    
    def get_counts(self, token_ids: List[int]) -> Dict[int, float]:
        """
        Get counts for multiple tokens.
        
        Args:
            token_ids: List of token identifiers
            
        Returns:
            Dict mapping token_id -> count
        """
        with self._lock:
            self._total_accesses += len(token_ids)
            return {tid: self._counts.get(tid, 0) for tid in token_ids}
    
    def set_count(self, token_id: int, count: float):
        """Set count for a token."""
        with self._lock:
            self._counts[token_id] = count
    
    def remove(self, token_id: int):
        """Remove a token from tracking."""
        with self._lock:
            if token_id in self._counts:
                del self._counts[token_id]
    
    def remove_batch(self, token_ids: List[int]):
        """Remove multiple tokens from tracking."""
        with self._lock:
            for token_id in token_ids:
                if token_id in self._counts:
                    del self._counts[token_id]
    
    def decay(self, factor: float):
        """
        Apply multiplicative decay to all counts.
        
        Args:
            factor: Decay factor (0 < factor <= 1)
        """
        if not 0 < factor <= 1:
            raise ValueError("Decay factor must be in (0, 1]")
        
        with self._lock:
            for token_id in self._counts:
                self._counts[token_id] *= factor
    
    def prune(self, min_count: float) -> List[int]:
        """
        Remove entries with count below threshold.
        
        Args:
            min_count: Minimum count threshold
            
        Returns:
            List of pruned token IDs
        """
        with self._lock:
            pruned = [
                tid for tid, count in self._counts.items()
                if count < min_count
            ]
            for tid in pruned:
                del self._counts[tid]
            return pruned
    
    def get_below_threshold(self, threshold: float) -> List[int]:
        """Get token IDs with count below threshold (without removing)."""
        with self._lock:
            return [
                tid for tid, count in self._counts.items()
                if count < threshold
            ]
    
    def get_above_threshold(self, threshold: float) -> List[int]:
        """Get token IDs with count at or above threshold."""
        with self._lock:
            return [
                tid for tid, count in self._counts.items()
                if count >= threshold
            ]
    
    def get_top_k(self, k: int) -> List[Tuple[int, float]]:
        """
        Get top k tokens by count.
        
        Args:
            k: Number of top entries to return
            
        Returns:
            List of (token_id, count) tuples, sorted by count descending
        """
        with self._lock:
            items = list(self._counts.items())
            items.sort(key=lambda x: x[1], reverse=True)
            return items[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the counter.
        
        Returns:
            Dict with count statistics
        """
        with self._lock:
            if not self._counts:
                return {
                    "num_entries": 0,
                    "total_updates": self._total_updates,
                    "total_accesses": self._total_accesses,
                    "min_count": 0,
                    "max_count": 0,
                    "mean_count": 0,
                    "median_count": 0,
                }
            
            counts = list(self._counts.values())
            counts_arr = np.array(counts)
            
            return {
                "num_entries": len(self._counts),
                "total_updates": self._total_updates,
                "total_accesses": self._total_accesses,
                "min_count": float(np.min(counts_arr)),
                "max_count": float(np.max(counts_arr)),
                "mean_count": float(np.mean(counts_arr)),
                "median_count": float(np.median(counts_arr)),
                "std_count": float(np.std(counts_arr)),
            }
    
    def clear(self):
        """Clear all counts."""
        with self._lock:
            self._counts.clear()
            self._total_updates = 0
            self._total_accesses = 0
    
    def get_all_counts(self) -> Dict[int, float]:
        """Get a copy of all counts."""
        with self._lock:
            return self._counts.copy()
    
    def set_all_counts(self, counts: Dict[int, float]):
        """Set all counts from a dictionary."""
        with self._lock:
            self._counts = counts.copy()
    
    def __len__(self) -> int:
        """Return number of tracked entries."""
        with self._lock:
            return len(self._counts)
    
    def __contains__(self, token_id: int) -> bool:
        """Check if token is tracked."""
        with self._lock:
            return token_id in self._counts

