"""Tests for batch iterator."""

import unittest
import numpy as np
import pandas as pd

from distributed_trainer.config import FeatureConfig, TargetConfig, WeightConfig
from distributed_trainer.batch_iterator import Batch, BatchIterator


class TestBatch(unittest.TestCase):
    """Tests for Batch dataclass."""
    
    def test_empty_batch(self):
        """Test empty batch creation."""
        batch = Batch()
        
        self.assertEqual(batch.batch_size, 0)
        self.assertEqual(batch.sparse_features, {})
        self.assertEqual(batch.dense_features, {})
    
    def test_get_all_sparse_tokens(self):
        """Test extracting all sparse tokens."""
        batch = Batch(
            sparse_features={
                "user": np.array([1, 2, 3]),
                "item": np.array([10, 20, 30]),
            },
            batch_size=3
        )
        
        tokens = batch.get_all_sparse_tokens()
        
        self.assertEqual(set(tokens), {1, 2, 3, 10, 20, 30})
    
    def test_get_sparse_tokens_by_feature(self):
        """Test extracting tokens per feature."""
        batch = Batch(
            sparse_features={
                "user": np.array([1, 1, 2]),
                "item": np.array([10, 20, 10]),
            },
            batch_size=3
        )
        
        tokens = batch.get_sparse_tokens_by_feature()
        
        self.assertEqual(set(tokens["user"]), {1, 2})
        self.assertEqual(set(tokens["item"]), {10, 20})
    
    def test_get_dense_matrix(self):
        """Test stacking dense features."""
        batch = Batch(
            dense_features={
                "a": np.array([1.0, 2.0, 3.0]),
                "b": np.array([4.0, 5.0, 6.0]),
            },
            batch_size=3
        )
        
        matrix = batch.get_dense_matrix()
        
        self.assertEqual(matrix.shape, (3, 2))


class TestBatchIterator(unittest.TestCase):
    """Tests for BatchIterator."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "item_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "click": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "weight": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
        })
        
        self.feature_config = FeatureConfig(
            sparse_features=["user_id", "item_id"],
            dense_features=["price"],
            embedding_dims={"user_id": 8, "item_id": 8},
        )
        
        self.target_config = TargetConfig(
            target_columns=["click"],
            task_type="binary_classification"
        )
        
        self.weight_config = WeightConfig()
    
    def test_basic_iteration(self):
        """Test basic batch iteration."""
        iterator = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=self.weight_config,
            batch_size=3,
            shuffle=False,
        )
        
        batches = list(iterator)
        
        # Should have 4 batches (10 samples, batch size 3)
        self.assertEqual(len(batches), 4)
        
        # First batch should have 3 samples
        self.assertEqual(batches[0].batch_size, 3)
        
        # Last batch should have 1 sample
        self.assertEqual(batches[3].batch_size, 1)
    
    def test_sparse_features_extraction(self):
        """Test sparse feature extraction."""
        iterator = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=self.weight_config,
            batch_size=5,
            shuffle=False,
        )
        
        batch = next(iter(iterator))
        
        self.assertIn("user_id", batch.sparse_features)
        self.assertIn("item_id", batch.sparse_features)
        self.assertEqual(len(batch.sparse_features["user_id"]), 5)
    
    def test_dense_features_extraction(self):
        """Test dense feature extraction."""
        iterator = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=self.weight_config,
            batch_size=5,
            shuffle=False,
        )
        
        batch = next(iter(iterator))
        
        self.assertIn("price", batch.dense_features)
        self.assertEqual(len(batch.dense_features["price"]), 5)
    
    def test_targets_extraction(self):
        """Test target extraction."""
        iterator = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=self.weight_config,
            batch_size=5,
            shuffle=False,
        )
        
        batch = next(iter(iterator))
        
        self.assertEqual(len(batch.targets), 5)
        np.testing.assert_array_equal(batch.targets, [1, 0, 1, 0, 1])
    
    def test_sample_weights(self):
        """Test sample weight extraction."""
        weight_config = WeightConfig(
            weight_column="weight",
            normalize_weights=False
        )
        
        iterator = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=weight_config,
            batch_size=5,
            shuffle=False,
        )
        
        batch = next(iter(iterator))
        
        self.assertIsNotNone(batch.weights)
        self.assertEqual(len(batch.weights), 5)
    
    def test_weight_normalization(self):
        """Test sample weight normalization."""
        weight_config = WeightConfig(
            weight_column="weight",
            normalize_weights=True
        )
        
        iterator = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=weight_config,
            batch_size=5,
            shuffle=False,
        )
        
        batch = next(iter(iterator))
        
        # Normalized weights should sum to batch_size
        np.testing.assert_almost_equal(batch.weights.sum(), 5.0)
    
    def test_shuffling(self):
        """Test data shuffling."""
        iterator1 = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=self.weight_config,
            batch_size=10,
            shuffle=True,
            seed=42,
        )
        
        iterator2 = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=self.weight_config,
            batch_size=10,
            shuffle=True,
            seed=42,
        )
        
        batch1 = next(iter(iterator1))
        batch2 = next(iter(iterator2))
        
        # Same seed should give same order
        np.testing.assert_array_equal(
            batch1.sparse_features["user_id"],
            batch2.sparse_features["user_id"]
        )
    
    def test_num_samples_and_batches(self):
        """Test num_samples and num_batches properties."""
        iterator = BatchIterator(
            data=self.data,
            feature_config=self.feature_config,
            target_config=self.target_config,
            weight_config=self.weight_config,
            batch_size=3,
            shuffle=False,
        )
        
        self.assertEqual(iterator.num_samples, 10)
        self.assertEqual(iterator.num_batches, 4)
        self.assertEqual(len(iterator), 4)


if __name__ == "__main__":
    unittest.main()

