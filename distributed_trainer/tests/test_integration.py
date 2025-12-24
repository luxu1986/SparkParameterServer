"""Integration tests for the complete distributed trainer."""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd

# Skip if TensorFlow not available
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

from distributed_trainer.config import (
    FeatureConfig,
    TargetConfig,
    WeightConfig,
    TrainerConfig,
)
from distributed_trainer.thread_config import ThreadConfig


@unittest.skipIf(not HAS_TF, "TensorFlow not available")
class TestLocalIntegration(unittest.TestCase):
    """Integration tests running locally (no Spark)."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary data directory
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create sample parquet files
        for i in range(2):
            data = pd.DataFrame({
                "user_id": np.random.randint(0, 100, 100),
                "item_id": np.random.randint(0, 1000, 100),
                "price": np.random.randn(100).astype(np.float32),
                "click": np.random.randint(0, 2, 100),
            })
            data.to_parquet(os.path.join(cls.temp_dir, f"part_{i}.parquet"))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def _create_model(self):
        """Create a simple test model with explicit layer names for consistency."""
        # Input shape: user_id(8) + item_id(8) + price(1) = 17
        inputs = tf.keras.Input(shape=(17,), name="input")
        x = tf.keras.layers.Dense(16, activation="relu", name="hidden")(inputs)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        return tf.keras.Model(inputs, outputs, name="test_model")
    
    def test_worker_trainer_basic(self):
        """Test WorkerTrainer with basic training."""
        from pyspark_ps import PSConfig, PSMainClient
        from distributed_trainer.worker import WorkerTrainer
        
        config = TrainerConfig(
            thread_config=ThreadConfig(mkl_num_threads=2),
            feature_config=FeatureConfig(
                sparse_features=["user_id", "item_id"],
                dense_features=["price"],
                embedding_dims={"user_id": 8, "item_id": 8},
            ),
            target_config=TargetConfig(
                target_columns=["click"],
                task_type="binary_classification"
            ),
            weight_config=WeightConfig(),
            batch_size=32,
            ps_config=PSConfig(
                num_servers=1,
                embedding_dim=8,
                server_port_base=50800,
            ),
        )
        
        # Start PS server
        ps_client = PSMainClient(None, config.ps_config)
        server_info = ps_client.start_servers()
        
        try:
            # Initialize weights using the same naming convention as TFModelWrapper
            model = self._create_model()
            weight_shapes = {}
            for layer in model.layers:
                for weight in layer.weights:
                    # Use layer.name/weight.name for unique naming
                    unique_name = f"{layer.name}/{weight.name}"
                    weight_shapes[unique_name] = tuple(weight.shape.as_list())
            
            ps_client.init_weights(weight_shapes)
            
            # Create worker
            worker = WorkerTrainer(
                worker_id=0,
                config=config,
                server_info=server_info,
                model_builder=self._create_model,
            )
            
            # Train on test data
            partition_paths = [
                os.path.join(self.temp_dir, f"part_{i}.parquet")
                for i in range(2)
            ]
            
            result = worker.train_partitions(partition_paths)
            
            self.assertGreater(result.total_samples, 0)
            self.assertGreater(result.total_batches, 0)
            self.assertIsInstance(result.avg_loss, float)
            
            worker.close()
            
        finally:
            ps_client.shutdown_servers()
    
    def test_full_trainer_local(self):
        """Test DistributedTrainer in local mode."""
        from pyspark_ps import PSConfig
        from distributed_trainer.trainer import DistributedTrainer
        
        config = TrainerConfig(
            thread_config=ThreadConfig(mkl_num_threads=2),
            feature_config=FeatureConfig(
                sparse_features=["user_id", "item_id"],
                dense_features=["price"],
                embedding_dims={"user_id": 8, "item_id": 8},
            ),
            target_config=TargetConfig(
                target_columns=["click"],
                task_type="binary_classification"
            ),
            weight_config=WeightConfig(),
            batch_size=32,
            num_workers=2,
            ps_config=PSConfig(
                num_servers=1,
                embedding_dim=8,
                server_port_base=50900,
            ),
            verbose=False,
        )
        
        trainer = DistributedTrainer(None, config)
        
        try:
            # Set model
            trainer.set_model_builder(self._create_model)
            
            # Get cluster stats
            stats = trainer.get_cluster_stats()
            self.assertIn("servers", stats)
            
        finally:
            trainer.shutdown()


class TestDataLoader(unittest.TestCase):
    """Tests for data loader functionality."""
    
    def test_partition_distribution_round_robin(self):
        """Test round-robin partition distribution."""
        from distributed_trainer.data_loader import S3ParquetDataLoader, PartitionInfo
        
        loader = S3ParquetDataLoader()
        
        partitions = [
            PartitionInfo(path=f"s3://bucket/file_{i}.parquet", size_bytes=100)
            for i in range(10)
        ]
        
        distribution = loader.distribute_partitions(
            partitions, num_workers=3, strategy="round_robin"
        )
        
        self.assertEqual(len(distribution), 3)
        self.assertEqual(len(distribution[0]), 4)  # Files 0, 3, 6, 9
        self.assertEqual(len(distribution[1]), 3)  # Files 1, 4, 7
        self.assertEqual(len(distribution[2]), 3)  # Files 2, 5, 8
    
    def test_partition_distribution_size_balanced(self):
        """Test size-balanced partition distribution."""
        from distributed_trainer.data_loader import S3ParquetDataLoader, PartitionInfo
        
        loader = S3ParquetDataLoader()
        
        # Create partitions with varying sizes
        partitions = [
            PartitionInfo(path=f"s3://bucket/file_{i}.parquet", size_bytes=(i + 1) * 100)
            for i in range(10)
        ]
        
        distribution = loader.distribute_partitions(
            partitions, num_workers=2, strategy="size_balanced"
        )
        
        self.assertEqual(len(distribution), 2)
        
        # Calculate total size per worker
        sizes = {}
        for worker_id, paths in distribution.items():
            sizes[worker_id] = sum(
                p.size_bytes for p in partitions if p.path in paths
            )
        
        # Sizes should be relatively balanced
        ratio = min(sizes.values()) / max(sizes.values())
        self.assertGreater(ratio, 0.5)  # Should be reasonably balanced
    
    def test_date_range_parsing(self):
        """Test date range parsing."""
        from distributed_trainer.data_loader import parse_date_range
        
        dates = parse_date_range("2025-01-01", "2025-01-05")
        
        self.assertEqual(len(dates), 5)
        self.assertEqual(dates[0], "2025-01-01")
        self.assertEqual(dates[-1], "2025-01-05")
    
    def test_format_date_path(self):
        """Test date path formatting."""
        from distributed_trainer.data_loader import format_date_path
        
        path = format_date_path(
            "s3://bucket/data/dt={date}/",
            "2025-01-15"
        )
        
        self.assertEqual(path, "s3://bucket/data/dt=2025-01-15/")


if __name__ == "__main__":
    unittest.main()

