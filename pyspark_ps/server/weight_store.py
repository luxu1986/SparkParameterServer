"""Dense weight storage for Parameter Server."""

import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from pyspark_ps.optimizers import create_optimizer, BaseOptimizer


class WeightStore:
    """
    Thread-safe storage for dense model weights.
    
    Features:
    - Named weight tensors of arbitrary shape
    - Per-layer optimizers
    - Gradient aggregation support
    - Versioning for staleness tracking
    """
    
    def __init__(
        self,
        optimizer_name: str = "adam",
        optimizer_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize weight store.
        
        Args:
            optimizer_name: Name of optimizer to use
            optimizer_config: Optimizer configuration
        """
        self._weights: Dict[str, np.ndarray] = {}
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._lock = threading.RLock()
        
        # Optimizer
        optimizer_config = optimizer_config or {}
        self._optimizer = create_optimizer(optimizer_name, **optimizer_config)
        
        # Version tracking for staleness
        self._version = 0
        self._layer_versions: Dict[str, int] = {}
        
        # Gradient accumulation
        self._accumulated_grads: Dict[str, np.ndarray] = {}
        self._accumulation_counts: Dict[str, int] = {}
        
        # Statistics
        self._stats = {
            "total_pulls": 0,
            "total_pushes": 0,
            "total_updates": 0,
        }
    
    def init_weights(
        self,
        name: str,
        shape: Tuple[int, ...],
        init_strategy: str = "normal",
        init_scale: float = 0.01,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize a weight tensor.
        
        Args:
            name: Layer/weight name
            shape: Tensor shape
            init_strategy: Initialization strategy
            init_scale: Scale for initialization
            dtype: Data type
        """
        with self._lock:
            if init_strategy == "zeros":
                weights = np.zeros(shape, dtype=dtype)
            elif init_strategy == "ones":
                weights = np.ones(shape, dtype=dtype)
            elif init_strategy == "random":
                weights = (np.random.random(shape).astype(dtype) - 0.5) * 2 * init_scale
            elif init_strategy == "normal":
                weights = np.random.randn(*shape).astype(dtype) * init_scale
            elif init_strategy == "xavier":
                # Xavier/Glorot initialization
                fan_in = shape[0] if len(shape) > 0 else 1
                fan_out = shape[1] if len(shape) > 1 else 1
                std = np.sqrt(2.0 / (fan_in + fan_out))
                weights = np.random.randn(*shape).astype(dtype) * std
            elif init_strategy == "he":
                # He initialization
                fan_in = shape[0] if len(shape) > 0 else 1
                std = np.sqrt(2.0 / fan_in)
                weights = np.random.randn(*shape).astype(dtype) * std
            else:
                raise ValueError(f"Unknown init strategy: {init_strategy}")
            
            self._weights[name] = weights
            self._shapes[name] = shape
            self._layer_versions[name] = 0
    
    def set_weights(self, name: str, weights: np.ndarray):
        """
        Set weight tensor directly.
        
        Args:
            name: Layer/weight name
            weights: Weight array
        """
        with self._lock:
            self._weights[name] = weights.astype(np.float32).copy()
            self._shapes[name] = weights.shape
            self._layer_versions[name] = self._layer_versions.get(name, 0) + 1
            self._version += 1
    
    def get(self, name: str) -> Optional[np.ndarray]:
        """
        Get weight tensor.
        
        Args:
            name: Layer/weight name
            
        Returns:
            Weight array or None if not found
        """
        with self._lock:
            self._stats["total_pulls"] += 1
            
            if name not in self._weights:
                return None
            
            return self._weights[name].copy()
    
    def get_batch(self, names: List[str]) -> Dict[str, np.ndarray]:
        """
        Get multiple weight tensors.
        
        Args:
            names: List of layer/weight names
            
        Returns:
            Dict mapping name -> weight array
        """
        with self._lock:
            self._stats["total_pulls"] += len(names)
            
            return {
                name: self._weights[name].copy()
                for name in names
                if name in self._weights
            }
    
    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all weight tensors."""
        with self._lock:
            self._stats["total_pulls"] += len(self._weights)
            return {name: w.copy() for name, w in self._weights.items()}
    
    def update(self, name: str, gradient: np.ndarray):
        """
        Update weights with gradient.
        
        Args:
            name: Layer/weight name
            gradient: Gradient array
        """
        with self._lock:
            self._stats["total_updates"] += 1
            
            if name not in self._weights:
                raise KeyError(f"Weight tensor not found: {name}")
            
            self._weights[name] = self._optimizer.update(
                name,
                self._weights[name],
                gradient
            )
            
            self._layer_versions[name] += 1
            self._version += 1
    
    def update_batch(self, gradients: Dict[str, np.ndarray]):
        """
        Update multiple weight tensors.
        
        Args:
            gradients: Dict mapping name -> gradient
        """
        with self._lock:
            self._stats["total_updates"] += len(gradients)
            
            for name, gradient in gradients.items():
                if name not in self._weights:
                    continue
                
                self._weights[name] = self._optimizer.update(
                    name,
                    self._weights[name],
                    gradient
                )
                self._layer_versions[name] += 1
            
            self._version += 1
    
    def accumulate_gradient(self, name: str, gradient: np.ndarray):
        """
        Accumulate gradient for later application.
        
        Useful for gradient aggregation across multiple workers.
        
        Args:
            name: Layer/weight name
            gradient: Gradient to accumulate
        """
        with self._lock:
            self._stats["total_pushes"] += 1
            
            if name not in self._accumulated_grads:
                self._accumulated_grads[name] = np.zeros_like(gradient)
                self._accumulation_counts[name] = 0
            
            self._accumulated_grads[name] += gradient
            self._accumulation_counts[name] += 1
    
    def apply_accumulated_gradients(self, average: bool = True):
        """
        Apply accumulated gradients and reset accumulators.
        
        Args:
            average: Whether to average gradients before applying
        """
        with self._lock:
            for name, grad in self._accumulated_grads.items():
                if name not in self._weights:
                    continue
                
                if average and self._accumulation_counts[name] > 0:
                    grad = grad / self._accumulation_counts[name]
                
                self._weights[name] = self._optimizer.update(
                    name,
                    self._weights[name],
                    grad
                )
                self._layer_versions[name] += 1
            
            self._version += 1
            self._accumulated_grads.clear()
            self._accumulation_counts.clear()
    
    def get_version(self) -> int:
        """Get global version number."""
        with self._lock:
            return self._version
    
    def get_layer_version(self, name: str) -> int:
        """Get version number for a specific layer."""
        with self._lock:
            return self._layer_versions.get(name, -1)
    
    def get_layer_names(self) -> List[str]:
        """Get list of all layer names."""
        with self._lock:
            return list(self._weights.keys())
    
    def get_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes of all weight tensors."""
        with self._lock:
            return self._shapes.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_params = sum(np.prod(s) for s in self._shapes.values())
            total_bytes = sum(w.nbytes for w in self._weights.values())
            
            return {
                "num_layers": len(self._weights),
                "total_parameters": int(total_params),
                "memory_bytes": total_bytes,
                "version": self._version,
                **self._stats,
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state for checkpointing."""
        with self._lock:
            return {
                "weights": {name: w.copy() for name, w in self._weights.items()},
                "shapes": self._shapes.copy(),
                "version": self._version,
                "layer_versions": self._layer_versions.copy(),
                "optimizer_state": self._optimizer.get_state(),
            }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        with self._lock:
            self._weights = {name: w.copy() for name, w in state["weights"].items()}
            self._shapes = state["shapes"].copy()
            self._version = state["version"]
            self._layer_versions = state["layer_versions"].copy()
            self._optimizer.set_state(state["optimizer_state"])
    
    def clear(self):
        """Clear all weights and reset state."""
        with self._lock:
            self._weights.clear()
            self._shapes.clear()
            self._version = 0
            self._layer_versions.clear()
            self._accumulated_grads.clear()
            self._accumulation_counts.clear()
            self._optimizer.reset()
            self._stats = {k: 0 for k in self._stats}
    
    def remove(self, name: str):
        """Remove a weight tensor."""
        with self._lock:
            if name in self._weights:
                del self._weights[name]
                del self._shapes[name]
                if name in self._layer_versions:
                    del self._layer_versions[name]
                self._optimizer.remove_state(name)
    
    def __len__(self) -> int:
        """Return number of weight tensors."""
        with self._lock:
            return len(self._weights)
    
    def __contains__(self, name: str) -> bool:
        """Check if weight tensor exists."""
        with self._lock:
            return name in self._weights

