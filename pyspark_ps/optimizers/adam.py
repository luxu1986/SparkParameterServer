"""Adam optimizer with bias correction."""

from typing import Any, Dict
import numpy as np

from pyspark_ps.optimizers.base import BaseOptimizer


class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Maintains exponential moving averages of gradients (m) and 
    squared gradients (v), with bias correction for the initial steps.
    
    Reference: "Adam: A Method for Stochastic Optimization"
               Kingma & Ba, 2014
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            weight_decay: Decoupled weight decay (AdamW)
            amsgrad: Use AMSGrad variant
        """
        super().__init__(learning_rate)
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Per-parameter step counters for bias correction
        self._param_steps: Dict[str, int] = {}
    
    def update(
        self,
        param_id: str,
        params: np.ndarray,
        grads: np.ndarray
    ) -> np.ndarray:
        """
        Apply Adam update.
        
        Update rule:
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads^2
            
            m_hat = m / (1 - beta1^t)
            v_hat = v / (1 - beta2^t)
            
            If AMSGrad:
                v_max = max(v_max, v_hat)
                params = params - lr * m_hat / (sqrt(v_max) + eps)
            Else:
                params = params - lr * m_hat / (sqrt(v_hat) + eps)
            
            If weight_decay > 0:
                params = params - lr * weight_decay * params
        """
        # Get or create moment buffers
        m = self._get_or_create_state(param_id, "m", params.shape, params.dtype, 0.0)
        v = self._get_or_create_state(param_id, "v", params.shape, params.dtype, 0.0)
        
        # Increment step for this parameter
        if param_id not in self._param_steps:
            self._param_steps[param_id] = 0
        self._param_steps[param_id] += 1
        t = self._param_steps[param_id]
        
        # Update biased first moment estimate
        m[:] = self.beta1 * m + (1 - self.beta1) * grads
        
        # Update biased second moment estimate
        v[:] = self.beta2 * v + (1 - self.beta2) * np.square(grads)
        
        # Bias correction
        bias_correction1 = 1 - self.beta1 ** t
        bias_correction2 = 1 - self.beta2 ** t
        
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        
        if self.amsgrad:
            # AMSGrad: use maximum of all past v_hat
            v_max = self._get_or_create_state(
                param_id, "v_max", params.shape, params.dtype, 0.0
            )
            np.maximum(v_max, v_hat, out=v_max)
            denom = np.sqrt(v_max) + self.epsilon
        else:
            denom = np.sqrt(v_hat) + self.epsilon
        
        # Apply update
        update = self.learning_rate * m_hat / denom
        
        # Decoupled weight decay (AdamW style)
        if self.weight_decay != 0:
            update = update + self.learning_rate * self.weight_decay * params
        
        return params - update
    
    def get_state(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "type": "adam",
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
            "step_count": self._step_count,
            "param_steps": self._param_steps.copy(),
            "state": {k: v.copy() for k, v in self._state.items()},
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore optimizer state."""
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.amsgrad = state.get("amsgrad", self.amsgrad)
        self._step_count = state.get("step_count", 0)
        self._param_steps = state.get("param_steps", {}).copy()
        
        if "state" in state:
            self._state = {k: v.copy() for k, v in state["state"].items()}
    
    def remove_state(self, param_id: str):
        """Remove state for a specific parameter."""
        super().remove_state(param_id)
        if param_id in self._param_steps:
            del self._param_steps[param_id]

