"""Tests for TensorFlow model wrapper."""

import unittest
import numpy as np

# Skip tests if TensorFlow is not available
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

from distributed_trainer.tf_model_wrapper import TFModelWrapper


@unittest.skipIf(not HAS_TF, "TensorFlow not available")
class TestTFModelWrapper(unittest.TestCase):
    """Tests for TFModelWrapper."""
    
    def _create_simple_model(self):
        """Create a simple test model."""
        inputs = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Dense(8, activation="relu", name="dense1")(inputs)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="dense2")(x)
        return tf.keras.Model(inputs, outputs)
    
    def test_init_with_model(self):
        """Test initialization with a model."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        self.assertIsNotNone(wrapper.get_model())
    
    def test_init_with_builder(self):
        """Test initialization with a builder function."""
        wrapper = TFModelWrapper(model_builder=self._create_simple_model)
        
        model = wrapper.build_model()
        self.assertIsNotNone(model)
    
    def test_get_weights(self):
        """Test weight extraction."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        weights = wrapper.get_weights()
        
        self.assertIsInstance(weights, dict)
        self.assertGreater(len(weights), 0)
        
        for name, w in weights.items():
            self.assertIsInstance(w, np.ndarray)
    
    def test_set_weights(self):
        """Test weight loading."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        # Get original weights
        original_weights = wrapper.get_weights()
        
        # Modify weights
        modified_weights = {
            name: np.zeros_like(w)
            for name, w in original_weights.items()
        }
        
        # Set modified weights
        wrapper.set_weights(modified_weights)
        
        # Verify weights changed
        new_weights = wrapper.get_weights()
        for name in new_weights:
            np.testing.assert_array_equal(new_weights[name], np.zeros_like(new_weights[name]))
    
    def test_get_weight_shapes(self):
        """Test weight shape extraction."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        shapes = wrapper.get_weight_shapes()
        
        self.assertIsInstance(shapes, dict)
        for name, shape in shapes.items():
            self.assertIsInstance(shape, tuple)
    
    def test_compute_gradients(self):
        """Test gradient computation."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        # Create dummy data
        inputs = np.random.randn(32, 10).astype(np.float32)
        targets = np.random.randint(0, 2, (32, 1)).astype(np.float32)
        
        loss, gradients = wrapper.compute_gradients(inputs, targets)
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(gradients, dict)
        self.assertGreater(len(gradients), 0)
        
        for name, grad in gradients.items():
            self.assertIsInstance(grad, np.ndarray)
    
    def test_compute_gradients_with_weights(self):
        """Test gradient computation with sample weights."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        inputs = np.random.randn(32, 10).astype(np.float32)
        targets = np.random.randint(0, 2, (32, 1)).astype(np.float32)
        sample_weights = np.ones(32, dtype=np.float32)
        sample_weights[:16] = 2.0  # Higher weight for first half
        
        loss, gradients = wrapper.compute_gradients(
            inputs, targets, sample_weights=sample_weights
        )
        
        self.assertIsInstance(loss, float)
        self.assertGreater(len(gradients), 0)
    
    def test_predict(self):
        """Test inference."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        inputs = np.random.randn(16, 10).astype(np.float32)
        predictions = wrapper.predict(inputs)
        
        self.assertEqual(predictions.shape, (16, 1))
        # Sigmoid output should be between 0 and 1
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_clone_architecture(self):
        """Test architecture cloning."""
        model = self._create_simple_model()
        wrapper = TFModelWrapper(model=model)
        
        cloned = wrapper.clone_architecture()
        
        # Verify same architecture
        original_shapes = wrapper.get_weight_shapes()
        cloned_shapes = cloned.get_weight_shapes()
        
        self.assertEqual(set(original_shapes.keys()), set(cloned_shapes.keys()))
        for name in original_shapes:
            self.assertEqual(original_shapes[name], cloned_shapes[name])
        
        # Verify different weight values (fresh initialization)
        original_weights = wrapper.get_weights()
        cloned_weights = cloned.get_weights()
        
        # At least one weight should be different (random init)
        any_different = False
        for name in original_weights:
            if not np.allclose(original_weights[name], cloned_weights[name]):
                any_different = True
                break
        
        self.assertTrue(any_different)


if __name__ == "__main__":
    unittest.main()

