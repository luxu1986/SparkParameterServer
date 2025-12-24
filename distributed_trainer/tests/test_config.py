"""Tests for configuration classes."""

import unittest

from distributed_trainer.config import (
    FeatureConfig,
    TargetConfig,
    WeightConfig,
    TrainerConfig,
)
from distributed_trainer.thread_config import ThreadConfig


class TestFeatureConfig(unittest.TestCase):
    """Tests for FeatureConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = FeatureConfig()
        
        self.assertEqual(config.sparse_features, [])
        self.assertEqual(config.dense_features, [])
        self.assertEqual(config.default_embedding_dim, 64)
    
    def test_embedding_dim_lookup(self):
        """Test embedding dimension lookup."""
        config = FeatureConfig(
            sparse_features=["user_id", "item_id"],
            embedding_dims={"user_id": 128, "item_id": 64},
            default_embedding_dim=32
        )
        
        self.assertEqual(config.get_embedding_dim("user_id"), 128)
        self.assertEqual(config.get_embedding_dim("item_id"), 64)
        self.assertEqual(config.get_embedding_dim("unknown"), 32)
    
    def test_total_dimensions(self):
        """Test total dimension calculations."""
        config = FeatureConfig(
            sparse_features=["user_id", "item_id"],
            dense_features=["price", "age"],
            embedding_dims={"user_id": 64, "item_id": 32},
        )
        
        self.assertEqual(config.get_total_embedding_dim(), 96)
        self.assertEqual(config.get_total_dense_dim(), 2)
        self.assertEqual(config.get_model_input_dim(), 98)
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        config = FeatureConfig(
            sparse_features=["a", "b"],
            dense_features=["c"],
            embedding_dims={"a": 10, "b": 20},
        )
        
        d = config.to_dict()
        config2 = FeatureConfig.from_dict(d)
        
        self.assertEqual(config2.sparse_features, ["a", "b"])
        self.assertEqual(config2.embedding_dims["a"], 10)


class TestTargetConfig(unittest.TestCase):
    """Tests for TargetConfig."""
    
    def test_binary_classification(self):
        """Test binary classification setup."""
        config = TargetConfig(
            target_columns=["click"],
            task_type="binary_classification"
        )
        
        self.assertEqual(config.get_output_dim(), 1)
        self.assertEqual(config.get_activation(), "sigmoid")
        self.assertEqual(config.get_loss_name(), "binary_crossentropy")
    
    def test_multiclass(self):
        """Test multiclass classification setup."""
        config = TargetConfig(
            target_columns=["category"],
            task_type="multiclass",
            num_classes=10
        )
        
        self.assertEqual(config.get_output_dim(), 10)
        self.assertEqual(config.get_activation(), "softmax")
        self.assertEqual(config.get_loss_name(), "sparse_categorical_crossentropy")
    
    def test_regression(self):
        """Test regression setup."""
        config = TargetConfig(
            target_columns=["price"],
            task_type="regression"
        )
        
        self.assertEqual(config.get_output_dim(), 1)
        self.assertEqual(config.get_activation(), "linear")
        self.assertEqual(config.get_loss_name(), "mse")


class TestWeightConfig(unittest.TestCase):
    """Tests for WeightConfig."""
    
    def test_no_weights(self):
        """Test configuration without sample weights."""
        config = WeightConfig()
        
        self.assertFalse(config.has_weights())
        self.assertIsNone(config.weight_column)
    
    def test_with_weights(self):
        """Test configuration with sample weights."""
        config = WeightConfig(
            weight_column="sample_weight",
            normalize_weights=True,
            default_weight=0.5
        )
        
        self.assertTrue(config.has_weights())
        self.assertEqual(config.weight_column, "sample_weight")
        self.assertTrue(config.normalize_weights)


class TestTrainerConfig(unittest.TestCase):
    """Tests for TrainerConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TrainerConfig()
        
        self.assertEqual(config.batch_size, 1024)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.model_optimizer, "adam")
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = TrainerConfig(
            feature_config=FeatureConfig(sparse_features=["id"]),
            target_config=TargetConfig(target_columns=["label"]),
        )
        config.validate()  # Should not raise
        
        # Invalid batch size
        with self.assertRaises(ValueError):
            config = TrainerConfig(batch_size=0)
            config.feature_config.sparse_features = ["id"]
            config.target_config.target_columns = ["label"]
            config.validate()
        
        # No features
        with self.assertRaises(ValueError):
            config = TrainerConfig()
            config.target_config.target_columns = ["label"]
            config.validate()
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        config = TrainerConfig(
            batch_size=2048,
            num_workers=8,
            feature_config=FeatureConfig(
                sparse_features=["user", "item"],
                dense_features=["price"]
            ),
            target_config=TargetConfig(target_columns=["click"]),
        )
        
        json_str = config.to_json()
        config2 = TrainerConfig.from_json(json_str)
        
        self.assertEqual(config2.batch_size, 2048)
        self.assertEqual(config2.num_workers, 8)
        self.assertEqual(config2.feature_config.sparse_features, ["user", "item"])


class TestThreadConfig(unittest.TestCase):
    """Tests for ThreadConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ThreadConfig()
        
        self.assertEqual(config.mkl_num_threads, 4)
        self.assertEqual(config.openmp_num_threads, 4)
        self.assertEqual(config.tf_intra_op_parallelism, 4)
        self.assertEqual(config.tf_inter_op_parallelism, 2)
    
    def test_get_optimal_config(self):
        """Test optimal config calculation."""
        from distributed_trainer.thread_config import get_optimal_thread_config
        
        config = get_optimal_thread_config(num_cores=16, num_workers_per_node=4)
        
        # Should divide cores among workers
        self.assertGreater(config.mkl_num_threads, 0)
        self.assertLessEqual(config.mkl_num_threads, 16)


if __name__ == "__main__":
    unittest.main()

