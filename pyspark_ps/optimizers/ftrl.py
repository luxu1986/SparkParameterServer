"""FTRL-Proximal optimizer for sparse features."""

from typing import Any, Dict
import numpy as np

from pyspark_ps.optimizers.base import BaseOptimizer


class FTRLOptimizer(BaseOptimizer):
    """
    FTRL-Proximal optimizer (Follow The Regularized Leader).
    
    Optimal for sparse features in online learning settings.
    Combines L1 and L2 regularization with adaptive learning rates.
    Particularly effective for large-scale sparse linear models.
    
    Reference: "Ad Click Prediction: a View from the Trenches"
               McMahan et al., 2013
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0,
        learning_rate_power: float = -0.5
    ):
        """
        Initialize FTRL optimizer.
        
        Args:
            alpha: Learning rate (roughly equivalent to 1/learning_rate in SGD)
            beta: Learning rate scale parameter
            l1: L1 regularization strength (induces sparsity)
            l2: L2 regularization strength
            learning_rate_power: Power for learning rate decay (-0.5 is standard)
        """
        super().__init__(alpha)  # alpha is our "learning rate"
        
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.learning_rate_power = learning_rate_power
    
    def update(
        self,
        param_id: str,
        params: np.ndarray,
        grads: np.ndarray
    ) -> np.ndarray:
        """
        Apply FTRL update.
        
        FTRL maintains:
            z: Sum of gradients - learning_rate * params (accumulated)
            n: Sum of squared gradients (for adaptive learning rate)
        
        Update rule:
            1. Update n: n += grads^2
            2. Update z: z += grads - (sqrt(n) - sqrt(n_prev)) / alpha * params
            3. Compute new params with L1/L2 regularization:
               If |z| <= l1:
                   params = 0  (sparse!)
               Else:
                   params = -(z - sign(z) * l1) / (l2 + (beta + sqrt(n)) / alpha)
        """
        # Get or create state
        z = self._get_or_create_state(param_id, "z", params.shape, params.dtype, 0.0)
        n = self._get_or_create_state(param_id, "n", params.shape, params.dtype, 0.0)
        
        # Store previous n for computing sigma
        n_prev = n.copy()
        
        # Update n (sum of squared gradients)
        n += np.square(grads)
        
        # Compute sigma (learning rate adjustment)
        sigma = (np.power(n, -self.learning_rate_power) - 
                 np.power(n_prev, -self.learning_rate_power)) / self.alpha
        
        # Update z
        z += grads - sigma * params
        
        # Compute new parameters with L1/L2 regularization
        # This is the proximal step
        new_params = np.zeros_like(params)
        
        # Mask for non-zero updates
        mask = np.abs(z) > self.l1
        
        if np.any(mask):
            # Apply soft thresholding and L2 regularization
            sign_z = np.sign(z[mask])
            denominator = self.l2 + (self.beta + np.sqrt(n[mask])) / self.alpha
            new_params[mask] = -(z[mask] - sign_z * self.l1) / denominator
        
        return new_params
    
    def update_sparse(
        self,
        param_id: str,
        params: np.ndarray,
        grad_indices: np.ndarray,
        grad_values: np.ndarray
    ) -> np.ndarray:
        """
        Sparse gradient update for FTRL.
        
        Only updates the rows/elements specified by grad_indices.
        
        Args:
            param_id: Parameter identifier
            params: Full parameter array
            grad_indices: Indices with gradients
            grad_values: Gradient values for those indices
            
        Returns:
            Updated parameter array
        """
        # Get or create state
        z = self._get_or_create_state(param_id, "z", params.shape, params.dtype, 0.0)
        n = self._get_or_create_state(param_id, "n", params.shape, params.dtype, 0.0)
        
        # Store previous n for affected indices
        n_prev = n[grad_indices].copy()
        
        # Update n for affected indices
        n[grad_indices] += np.square(grad_values)
        
        # Compute sigma for affected indices
        sigma = (np.power(n[grad_indices], -self.learning_rate_power) - 
                 np.power(n_prev, -self.learning_rate_power)) / self.alpha
        
        # Update z for affected indices
        z[grad_indices] += grad_values - sigma * params[grad_indices]
        
        # Compute new parameters for affected indices
        z_vals = z[grad_indices]
        n_vals = n[grad_indices]
        
        mask = np.abs(z_vals) > self.l1
        new_vals = np.zeros_like(z_vals)
        
        if np.any(mask):
            sign_z = np.sign(z_vals[mask])
            denominator = self.l2 + (self.beta + np.sqrt(n_vals[mask])) / self.alpha
            new_vals[mask] = -(z_vals[mask] - sign_z * self.l1) / denominator
        
        params[grad_indices] = new_vals
        
        return params
    
    def get_state(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "type": "ftrl",
            "alpha": self.alpha,
            "beta": self.beta,
            "l1": self.l1,
            "l2": self.l2,
            "learning_rate_power": self.learning_rate_power,
            "step_count": self._step_count,
            "state": {k: v.copy() for k, v in self._state.items()},
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore optimizer state."""
        self.alpha = state.get("alpha", self.alpha)
        self.beta = state.get("beta", self.beta)
        self.l1 = state.get("l1", self.l1)
        self.l2 = state.get("l2", self.l2)
        self.learning_rate_power = state.get("learning_rate_power", self.learning_rate_power)
        self._step_count = state.get("step_count", 0)
        
        if "state" in state:
            self._state = {k: v.copy() for k, v in state["state"].items()}
    
    def get_sparsity(self, param_id: str) -> float:
        """
        Get the sparsity ratio for a parameter.
        
        Returns the fraction of zero values induced by L1 regularization.
        """
        z_key = f"{param_id}_z"
        if z_key not in self._state:
            return 0.0
        
        z = self._state[z_key]
        num_zeros = np.sum(np.abs(z) <= self.l1)
        return float(num_zeros) / z.size

