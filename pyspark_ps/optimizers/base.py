"""Base optimizer interface for Parameter Server."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseOptimizer(ABC):
    """
    Base class for all optimizers.
    
    Optimizers are responsible for updating parameters based on gradients.
    They maintain internal state (like momentum buffers) that persists
    across updates.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for updates
        """
        self.learning_rate = learning_rate
        self._state: Dict[str, Any] = {}
        self._step_count = 0
    
    @abstractmethod
    def update(
        self,
        param_id: str,
        params: np.ndarray,
        grads: np.ndarray
    ) -> np.ndarray:
        """
        Apply gradient update and return new parameters.
        
        Args:
            param_id: Unique identifier for the parameter
            params: Current parameter values
            grads: Gradient values
            
        Returns:
            Updated parameter values
        """
        pass
    
    def update_batch(
        self,
        param_ids: list,
        params_list: list,
        grads_list: list
    ) -> list:
        """
        Apply gradient updates to a batch of parameters.
        
        Default implementation calls update() for each parameter.
        Subclasses can override for vectorized batch updates.
        
        Args:
            param_ids: List of parameter identifiers
            params_list: List of parameter arrays
            grads_list: List of gradient arrays
            
        Returns:
            List of updated parameter arrays
        """
        return [
            self.update(pid, params, grads)
            for pid, params, grads in zip(param_ids, params_list, grads_list)
        ]
    
    def step(self):
        """Increment step counter (for schedulers, etc.)."""
        self._step_count += 1
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Return optimizer state for checkpointing.
        
        Returns:
            Dictionary containing all state needed to resume training
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]):
        """
        Restore optimizer state from checkpoint.
        
        Args:
            state: State dictionary from get_state()
        """
        pass
    
    def reset(self):
        """Reset optimizer state."""
        self._state.clear()
        self._step_count = 0
    
    def remove_state(self, param_id: str):
        """Remove state for a specific parameter (e.g., after pruning)."""
        keys_to_remove = [k for k in self._state if k.startswith(f"{param_id}_")]
        for k in keys_to_remove:
            del self._state[k]
    
    def _get_or_create_state(
        self,
        param_id: str,
        state_name: str,
        shape: tuple,
        dtype: np.dtype = np.float32,
        init_value: float = 0.0
    ) -> np.ndarray:
        """
        Get or create a state buffer for a parameter.
        
        Args:
            param_id: Parameter identifier
            state_name: Name of the state (e.g., "momentum", "velocity")
            shape: Shape of the buffer
            dtype: Data type
            init_value: Initial value
            
        Returns:
            State buffer array
        """
        key = f"{param_id}_{state_name}"
        if key not in self._state:
            self._state[key] = np.full(shape, init_value, dtype=dtype)
        return self._state[key]


class OptimizerGroup:
    """
    Group of optimizers for different parameter types.
    
    Allows using different optimizers for different parameter groups
    (e.g., different learning rates for embeddings vs. dense weights).
    """
    
    def __init__(self):
        self._optimizers: Dict[str, BaseOptimizer] = {}
        self._param_to_group: Dict[str, str] = {}
    
    def add_group(self, name: str, optimizer: BaseOptimizer):
        """Add an optimizer for a parameter group."""
        self._optimizers[name] = optimizer
    
    def assign_param(self, param_id: str, group: str):
        """Assign a parameter to a group."""
        if group not in self._optimizers:
            raise ValueError(f"Unknown optimizer group: {group}")
        self._param_to_group[param_id] = group
    
    def update(
        self,
        param_id: str,
        params: np.ndarray,
        grads: np.ndarray,
        default_group: str = "default"
    ) -> np.ndarray:
        """Update parameter using its assigned optimizer."""
        group = self._param_to_group.get(param_id, default_group)
        optimizer = self._optimizers.get(group)
        
        if optimizer is None:
            raise ValueError(f"No optimizer for group: {group}")
        
        return optimizer.update(param_id, params, grads)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state of all optimizers."""
        return {
            name: opt.get_state()
            for name, opt in self._optimizers.items()
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set state of all optimizers."""
        for name, opt_state in state.items():
            if name in self._optimizers:
                self._optimizers[name].set_state(opt_state)

