"""Tests for optimizers."""

import unittest
import numpy as np

from pyspark_ps.optimizers import (
    SGDOptimizer,
    AdamOptimizer,
    AdagradOptimizer,
    FTRLOptimizer,
    create_optimizer
)


class TestSGDOptimizer(unittest.TestCase):
    """Tests for SGD optimizer."""
    
    def test_basic_update(self):
        """Test basic SGD update without momentum."""
        opt = SGDOptimizer(learning_rate=0.1, momentum=0.0)
        
        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        new_params = opt.update("test", params, grads)
        
        expected = params - 0.1 * grads
        np.testing.assert_array_almost_equal(new_params, expected)
    
    def test_momentum(self):
        """Test SGD with momentum."""
        opt = SGDOptimizer(learning_rate=0.1, momentum=0.9)
        
        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # First update
        new_params = opt.update("test", params, grads)
        
        # Velocity should be grads (momentum * 0 + grads)
        # Update should be params - lr * velocity
        expected = params - 0.1 * grads
        np.testing.assert_array_almost_equal(new_params, expected)
        
        # Second update
        new_params = opt.update("test", new_params, grads)
        # Velocity = 0.9 * grads + grads = 1.9 * grads
        velocity = 0.9 * grads + grads
        expected = expected - 0.1 * velocity
        np.testing.assert_array_almost_equal(new_params, expected)
    
    def test_weight_decay(self):
        """Test SGD with weight decay."""
        opt = SGDOptimizer(learning_rate=0.1, weight_decay=0.01)
        
        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        new_params = opt.update("test", params, grads)
        
        # Gradients should include weight decay term
        effective_grads = grads + 0.01 * params
        expected = params - 0.1 * effective_grads
        np.testing.assert_array_almost_equal(new_params, expected)
    
    def test_state_save_restore(self):
        """Test saving and restoring optimizer state."""
        opt = SGDOptimizer(learning_rate=0.1, momentum=0.9)
        
        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Run a few updates
        for _ in range(3):
            params = opt.update("test", params, grads)
        
        # Save state
        state = opt.get_state()
        
        # Create new optimizer and restore
        opt2 = SGDOptimizer(learning_rate=0.05)  # Different LR
        opt2.set_state(state)
        
        # Should have same LR and state
        self.assertEqual(opt2.learning_rate, 0.1)
        self.assertEqual(opt2.momentum, 0.9)


class TestAdamOptimizer(unittest.TestCase):
    """Tests for Adam optimizer."""
    
    def test_basic_update(self):
        """Test basic Adam update."""
        opt = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        
        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        new_params = opt.update("test", params, grads)
        
        # Should decrease in direction of gradients
        self.assertTrue(np.all(new_params < params))
    
    def test_bias_correction(self):
        """Test that bias correction is applied."""
        opt = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        
        params = np.array([1.0], dtype=np.float32)
        grads = np.array([1.0], dtype=np.float32)
        
        # First update should have significant bias correction
        new_params = opt.update("test", params, grads)
        
        # Without bias correction, update would be very small
        # With correction, it should be closer to learning_rate
        update = params - new_params
        self.assertGreater(update[0], 0.0009)  # Close to lr
    
    def test_amsgrad(self):
        """Test AMSGrad variant."""
        opt = AdamOptimizer(learning_rate=0.001, amsgrad=True)
        
        params = np.array([1.0, 2.0], dtype=np.float32)
        
        # Updates with varying gradients
        for i in range(5):
            grads = np.array([0.1 * (i + 1), 0.1 * (5 - i)], dtype=np.float32)
            params = opt.update("test", params, grads)
        
        # Should complete without error
        self.assertTrue(np.all(np.isfinite(params)))


class TestAdagradOptimizer(unittest.TestCase):
    """Tests for Adagrad optimizer."""
    
    def test_basic_update(self):
        """Test basic Adagrad update."""
        opt = AdagradOptimizer(learning_rate=0.1, initial_accumulator=0.0)
        
        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        new_params = opt.update("test", params, grads)
        
        # Should move in direction of gradients
        self.assertTrue(np.all(new_params < params))
    
    def test_adaptive_learning_rate(self):
        """Test that learning rate adapts based on gradient history."""
        opt = AdagradOptimizer(learning_rate=0.1, initial_accumulator=0.0)
        
        params = np.array([1.0, 1.0], dtype=np.float32)
        
        # First dimension gets larger gradients
        for _ in range(10):
            grads = np.array([1.0, 0.1], dtype=np.float32)
            params = opt.update("test", params, grads)
        
        # Dimension with larger gradients should have smaller effective LR
        # So it should have moved less proportionally
        # Get accumulator and check
        acc = opt._state["test_accumulator"]
        self.assertGreater(acc[0], acc[1])
    
    def test_sparse_update(self):
        """Test sparse gradient update."""
        opt = AdagradOptimizer(learning_rate=0.1)
        
        # Large embedding table
        params = np.random.randn(1000, 64).astype(np.float32)
        
        # Sparse update (only some rows)
        indices = np.array([10, 50, 100])
        grad_values = np.random.randn(3, 64).astype(np.float32)
        
        new_params = opt.update_sparse("test", params, indices, grad_values)
        
        # Only updated rows should change
        unchanged_idx = [i for i in range(1000) if i not in indices]
        np.testing.assert_array_equal(new_params[unchanged_idx], params[unchanged_idx])


class TestFTRLOptimizer(unittest.TestCase):
    """Tests for FTRL optimizer."""
    
    def test_basic_update(self):
        """Test basic FTRL update."""
        opt = FTRLOptimizer(alpha=0.1, beta=1.0, l1=0.0, l2=0.0)
        
        params = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        new_params = opt.update("test", params, grads)
        
        # Should produce finite values
        self.assertTrue(np.all(np.isfinite(new_params)))
    
    def test_l1_sparsity(self):
        """Test that L1 regularization induces sparsity."""
        opt = FTRLOptimizer(alpha=0.1, beta=1.0, l1=1.0, l2=0.0)
        
        params = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        
        # Small gradients should be thresholded to zero with L1
        for _ in range(10):
            grads = np.array([0.001, 0.001, 0.001], dtype=np.float32)
            params = opt.update("test", params, grads)
        
        # With strong L1, params should be exactly zero
        np.testing.assert_array_equal(params, np.zeros(3))
    
    def test_sparse_update(self):
        """Test sparse FTRL update."""
        opt = FTRLOptimizer(alpha=0.1, l1=0.0, l2=0.0)
        
        params = np.random.randn(100, 10).astype(np.float32)
        
        indices = np.array([5, 25, 75])
        grad_values = np.random.randn(3, 10).astype(np.float32)
        
        new_params = opt.update_sparse("test", params, indices, grad_values)
        
        # Check that only specified rows changed
        for i in range(100):
            if i not in indices:
                np.testing.assert_array_equal(new_params[i], params[i])


class TestOptimizerFactory(unittest.TestCase):
    """Tests for optimizer factory function."""
    
    def test_create_sgd(self):
        """Test creating SGD optimizer."""
        opt = create_optimizer("sgd", learning_rate=0.01, momentum=0.9)
        self.assertIsInstance(opt, SGDOptimizer)
        self.assertEqual(opt.learning_rate, 0.01)
        self.assertEqual(opt.momentum, 0.9)
    
    def test_create_adam(self):
        """Test creating Adam optimizer."""
        opt = create_optimizer("adam", learning_rate=0.001)
        self.assertIsInstance(opt, AdamOptimizer)
    
    def test_create_adagrad(self):
        """Test creating Adagrad optimizer."""
        opt = create_optimizer("adagrad")
        self.assertIsInstance(opt, AdagradOptimizer)
    
    def test_create_ftrl(self):
        """Test creating FTRL optimizer."""
        opt = create_optimizer("ftrl", alpha=0.05, l1=0.1)
        self.assertIsInstance(opt, FTRLOptimizer)
    
    def test_invalid_optimizer(self):
        """Test that invalid optimizer name raises error."""
        with self.assertRaises(ValueError):
            create_optimizer("invalid_optimizer")


if __name__ == "__main__":
    unittest.main()

