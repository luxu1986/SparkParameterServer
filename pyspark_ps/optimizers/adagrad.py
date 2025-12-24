"""Adagrad optimizer for sparse gradients."""

from typing import Any, Dict
import numpy as np

from pyspark_ps.optimizers.base import BaseOptimizer


class AdagradOptimizer(BaseOptimizer):
    """
    Adagrad optimizer (Adaptive Gradient Algorithm).
    
    Particularly effective for sparse gradients and embedding training.
    Maintains a sum of squared gradients to adapt the learning rate
    per-parameter.
    
    Reference: "Adaptive Subgradient Methods for Online Learning and 
               Stochastic Optimization" - Duchi et al., 2011
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 1e-10,
        initial_accumulator: float = 0.1,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adagrad optimizer.
        
        Args:
            learning_rate: Learning rate (larger values work well due to adaptation)
            epsilon: Small constant for numerical stability
            initial_accumulator: Initial value for squared gradient accumulator
                                (helps with early iterations)
            weight_decay: L2 regularization factor
        """
        super().__init__(learning_rate)
        
        self.epsilon = epsilon
        self.initial_accumulator = initial_accumulator
        self.weight_decay = weight_decay
    
    def update(
        self,
        param_id: str,
        params: np.ndarray,
        grads: np.ndarray
    ) -> np.ndarray:
        """
        Apply Adagrad update.
        
        Update rule:
            If weight_decay > 0:
                grads = grads + weight_decay * params
            
            accumulator += grads^2
            params = params - lr * grads / (sqrt(accumulator) + eps)
        """
        # Apply weight decay to gradients
        if self.weight_decay != 0:
            grads = grads + self.weight_decay * params
        
        # Get or create accumulator
        accumulator = self._get_or_create_state(
            param_id, "accumulator",
            params.shape, params.dtype,
            self.initial_accumulator
        )
        
        # Update accumulator with squared gradients
        accumulator += np.square(grads)
        
        # Compute adaptive learning rate and update
        std = np.sqrt(accumulator) + self.epsilon
        update = self.learning_rate * grads / std
        
        return params - update
    
    def update_batch(
        self,
        param_ids: list,
        params_list: list,
        grads_list: list
    ) -> list:
        """
        Batch update optimized for sparse embeddings.
        
        When dealing with many small embeddings (common in sparse models),
        this batched version can be more cache-friendly.
        """
        if not params_list:
            return []
        
        results = []
        for param_id, params, grads in zip(param_ids, params_list, grads_list):
            results.append(self.update(param_id, params, grads))
        
        return results
    
    def update_sparse(
        self,
        param_id: str,
        params: np.ndarray,
        grad_indices: np.ndarray,
        grad_values: np.ndarray
    ) -> np.ndarray:
        """
        Sparse gradient update for embedding tables.
        
        Only updates the rows specified by grad_indices, which is
        more efficient for large embedding tables with sparse updates.
        
        Args:
            param_id: Parameter identifier
            params: Full embedding table
            grad_indices: Indices of rows with gradients
            grad_values: Gradient values for those rows
            
        Returns:
            Updated embedding table
        """
        # Get or create accumulator
        accumulator = self._get_or_create_state(
            param_id, "accumulator",
            params.shape, params.dtype,
            self.initial_accumulator
        )
        
        # Apply weight decay if needed
        if self.weight_decay != 0:
            grad_values = grad_values + self.weight_decay * params[grad_indices]
        
        # Update only affected rows of accumulator
        accumulator[grad_indices] += np.square(grad_values)
        
        # Compute updates
        std = np.sqrt(accumulator[grad_indices]) + self.epsilon
        updates = self.learning_rate * grad_values / std
        
        # Apply updates
        params[grad_indices] -= updates
        
        return params
    
    def get_state(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "type": "adagrad",
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "initial_accumulator": self.initial_accumulator,
            "weight_decay": self.weight_decay,
            "step_count": self._step_count,
            "state": {k: v.copy() for k, v in self._state.items()},
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore optimizer state."""
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.initial_accumulator = state.get("initial_accumulator", self.initial_accumulator)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self._step_count = state.get("step_count", 0)
        
        if "state" in state:
            self._state = {k: v.copy() for k, v in state["state"].items()}
    
    def decay_accumulator(self, factor: float = 0.9):
        """
        Decay the accumulator values.
        
        Can help prevent learning rates from becoming too small
        over time, especially in long-running training.
        
        Args:
            factor: Multiplicative decay factor (0 < factor <= 1)
        """
        for key in self._state:
            if key.endswith("_accumulator"):
                self._state[key] *= factor

