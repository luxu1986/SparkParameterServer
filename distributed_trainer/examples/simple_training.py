#!/usr/bin/env python3
"""
Simple Distributed Trainer Example

This example demonstrates how to use the DistributedTrainer to train
a TensorFlow model with sparse embeddings stored in the Parameter Server.

Run with:
    python -m distributed_trainer.examples.simple_training
"""

import os
import tempfile
import numpy as np
import pandas as pd

# Configure threads before importing TensorFlow
from distributed_trainer.thread_config import configure_threads, ThreadConfig
configure_threads(ThreadConfig(mkl_num_threads=2, openmp_num_threads=2))

import tensorflow as tf
from pyspark_ps import PSConfig
from distributed_trainer import (
    DistributedTrainer,
    TrainerConfig,
    FeatureConfig,
    TargetConfig,
    WeightConfig,
)


def create_sample_data(output_dir: str, num_files: int = 3, samples_per_file: int = 1000):
    """Create sample parquet files for training."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        data = pd.DataFrame({
            # Sparse features (categorical IDs)
            "user_id": np.random.randint(0, 1000, samples_per_file),
            "item_id": np.random.randint(0, 5000, samples_per_file),
            "category_id": np.random.randint(0, 50, samples_per_file),
            
            # Dense features (numerical)
            "price": np.random.uniform(1, 100, samples_per_file).astype(np.float32),
            "user_age": np.random.randint(18, 65, samples_per_file).astype(np.float32),
            
            # Target
            "click": np.random.randint(0, 2, samples_per_file),
            
            # Sample weight (optional)
            "weight": np.random.uniform(0.5, 2.0, samples_per_file).astype(np.float32),
        })
        
        output_path = os.path.join(output_dir, f"part_{i:04d}.parquet")
        data.to_parquet(output_path, index=False)
        print(f"Created {output_path} with {samples_per_file} samples")
    
    return output_dir


def build_model():
    """
    Build a simple recommendation model.
    
    Input shape: 
        - user_id embedding (32) + item_id embedding (32) + category_id embedding (32)
        - price (1) + user_age (1)
        - Total: 32 + 32 + 32 + 1 + 1 = 98
    """
    inputs = tf.keras.Input(shape=(98,), name="input")
    
    # Hidden layers
    x = tf.keras.layers.Dense(64, activation="relu", name="hidden1")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="hidden2")(x)
    
    # Output
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    
    return tf.keras.Model(inputs, outputs, name="click_model")


def main():
    print("=" * 60)
    print("Distributed Trainer - Simple Training Example")
    print("=" * 60)
    
    # Create temporary directory for sample data
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data")
    
    print(f"\n1. Creating sample data in {data_dir}")
    create_sample_data(data_dir, num_files=3, samples_per_file=500)
    
    # Configure the trainer
    print("\n2. Configuring trainer")
    config = TrainerConfig(
        # Thread configuration
        thread_config=ThreadConfig(
            mkl_num_threads=1,
            openmp_num_threads=1,
            tf_intra_op_parallelism=1,
            tf_inter_op_parallelism=16,
        ),
        
        # Feature configuration
        # Note: All embeddings use the same dimension (from PSConfig.embedding_dim)
        feature_config=FeatureConfig(
            sparse_features=["user_id", "item_id", "category_id"],
            dense_features=["price", "user_age"],
            embedding_dims={
                "user_id": 32,
                "item_id": 32,
                "category_id": 32,  # Must match PSConfig.embedding_dim
            },
        ),
        
        # Target configuration
        target_config=TargetConfig(
            target_columns=["click"],
            task_type="binary_classification",
        ),
        
        # Sample weight configuration
        weight_config=WeightConfig(
            weight_column="weight",
            normalize_weights=False,
        ),
        
        # Training settings
        batch_size=64,
        num_workers=2,
        shuffle_data=True,
        
        # Optimizer settings
        model_optimizer="adam",
        model_learning_rate=0.001,
        embedding_optimizer="adagrad",
        embedding_learning_rate=0.01,
        
        # PS configuration
        ps_config=PSConfig(
            num_servers=2,
            embedding_dim=32,  # Default embedding dim
            server_port_base=51000,
        ),
        
        # Logging
        log_every_n_batches=10,
        verbose=True,
    )
    
    print(f"   - Sparse features: {config.feature_config.sparse_features}")
    print(f"   - Dense features: {config.feature_config.dense_features}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Model optimizer: {config.model_optimizer}")
    
    # Create trainer (local mode, no Spark)
    print("\n3. Starting trainer (local mode)")
    trainer = DistributedTrainer(spark_context=None, config=config)
    
    try:
        # Set the model
        print("\n4. Setting model")
        trainer.set_model_builder(build_model)
        
        # Get cluster stats
        stats = trainer.get_cluster_stats()
        print(f"   - PS servers: {len(stats.get('servers', []))}")
        print(f"   - Total embeddings: {stats.get('total_embeddings', 0)}")
        
        # Train on the sample data
        # In a real scenario, you would use train_date_range() with S3 paths
        print("\n5. Training on sample data")
        print("   (In production, use trainer.train_date_range() with S3 paths)")
        
        # For this example, we'll manually train using the worker
        from distributed_trainer.worker import WorkerTrainer
        
        worker = WorkerTrainer(
            worker_id=0,
            config=config,
            server_info=trainer._server_info,
            model_builder=build_model,
        )
        
        # Get all parquet files
        partition_paths = [
            os.path.join(data_dir, f"part_{i:04d}.parquet")
            for i in range(3)
        ]
        
        # Train
        result = worker.train_partitions(partition_paths)
        
        print(f"\n6. Training Results:")
        print(f"   - Total samples: {result.total_samples:,}")
        print(f"   - Total batches: {result.total_batches}")
        print(f"   - Average loss: {result.avg_loss:.6f}")
        print(f"   - Training time: {result.training_time_seconds:.2f}s")
        print(f"   - Samples/second: {result.metrics.get('samples_per_second', 0):.1f}")
        
        worker.close()
        
        # Get final stats
        final_stats = trainer.get_cluster_stats()
        print(f"\n7. Final PS Stats:")
        print(f"   - Total embeddings: {final_stats.get('total_embeddings', 0)}")
        print(f"   - Total weights: {final_stats.get('total_weights', 0)}")
        
    finally:
        # Cleanup
        print("\n8. Shutting down")
        trainer.shutdown()
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
