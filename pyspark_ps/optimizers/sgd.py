"""SGD optimizer with momentum and weight decay."""

from typing import Any, Dict
import numpy as np

from pyspark_ps.optimizers.base import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent with momentum and optional weight decay.
    
    Supports:
    - Classical momentum
    - Nesterov accelerated gradient
    - L2 weight decay (decoupled)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        dampening: float = 0.0
    ):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 = no momentum)
            weight_decay: L2 regularization factor
            nesterov: Use Nesterov momentum
            dampening: Dampening for momentum
        """
        super().__init__(learning_rate)
        
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires momentum > 0 and dampening = 0"
            )
    
    def update(
        self,
        param_id: str,
        params: np.ndarray,
        grads: np.ndarray
    ) -> np.ndarray:
        """
        Apply SGD update.
        
        Update rule:
            If momentum = 0:
                params = params - lr * (grads + weight_decay * params)
            Else:
                velocity = momentum * velocity + (1 - dampening) * grads
                If nesterov:
                    update = grads + momentum * velocity
                Else:
                    update = velocity
                params = params - lr * (update + weight_decay * params)
        """
        # Apply weight decay (decoupled)
        if self.weight_decay != 0:
            grads = grads + self.weight_decay * params
        
        if self.momentum == 0:
            # Simple SGD without momentum
            return params - self.learning_rate * grads
        
        # Get or create velocity buffer
        velocity = self._get_or_create_state(
            param_id, "velocity", params.shape, params.dtype, 0.0
        )
        
        # Update velocity
        velocity[:] = self.momentum * velocity + (1 - self.dampening) * grads
        
        # Compute update direction
        if self.nesterov:
            update = grads + self.momentum * velocity
        else:
            update = velocity
        
        return params - self.learning_rate * update
    
    def update_batch(
        self,
        param_ids: list,
        params_list: list,
        grads_list: list
    ) -> list:
        """
        Vectorized batch update for SGD.
        
        When all parameters have the same shape, this is more efficient
        than individual updates.
        """
        if not params_list:
            return []
        
        # Check if we can vectorize (all same shape)
        shapes = [p.shape for p in params_list]
        if len(set(shapes)) == 1 and self.momentum == 0:
            # Stack and update together
            params = np.stack(params_list)
            grads = np.stack(grads_list)
            
            if self.weight_decay != 0:
                grads = grads + self.weight_decay * params
            
            updated = params - self.learning_rate * grads
            return [updated[i] for i in range(len(params_list))]
        
        # Fall back to individual updates
        return super().update_batch(param_ids, params_list, grads_list)
    
    def get_state(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "type": "sgd",
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
            "dampening": self.dampening,
            "step_count": self._step_count,
            "state": {k: v.copy() for k, v in self._state.items()},
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore optimizer state."""
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.momentum = state.get("momentum", self.momentum)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.nesterov = state.get("nesterov", self.nesterov)
        self.dampening = state.get("dampening", self.dampening)
        self._step_count = state.get("step_count", 0)
        
        if "state" in state:
            self._state = {k: v.copy() for k, v in state["state"].items()}

