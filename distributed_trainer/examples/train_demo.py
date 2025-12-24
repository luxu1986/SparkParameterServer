# Databricks notebook source

"""
Train on Demo Net Revenue Data

This example trains a model to predict net_revenue using the dataset
stored on S3.

Architecture:
    - PS servers run on DRIVER node (as threads)
    - Spark executors process data partitions in parallel:
        1. Load batch from S3 parquet
        2. Pull model weights from PS (driver)
        3. Pull embeddings for batch tokens from PS (driver)
        4. Forward/backward pass locally
        5. Push gradients to PS (driver) for weight updates
    - Driver coordinates checkpointing and embedding decay

Features:
    - placement_id (sparse)
    - country (sparse)
    - prediction_floor (dense)
    - placement_type (sparse)
    - avg_device_market_price_bin (sparse)
    - count_of_device_requests_bin (sparse)
    - time_diff_hrs_bin (sparse)
    - sdk_supply_name (sparse)
    - do_not_track (sparse)

Target:
    - net_revenue (regression)

Run with:
    python -m distributed_trainer.examples.train_demo
"""

import os

# Configure threads before importing TensorFlow
from distributed_trainer.thread_config import configure_threads, ThreadConfig
configure_threads(ThreadConfig(
    mkl_num_threads=1,
    openmp_num_threads=1,
    tf_intra_op_parallelism=1,
    tf_inter_op_parallelism=16,
))

import tensorflow as tf
from pyspark_ps import PSConfig
from distributed_trainer import (
    DistributedTrainer,
    TrainerConfig,
    FeatureConfig,
    TargetConfig,
    WeightConfig,
)


# Data configuration
S3_DATA_PATH = "s3://demo/data/dt={date}/"
S3_CHECKPOINT_PATH = f"s3://demo/model/"
START_DATE = "2025-11-20"
END_DATE = "2025-11-22"

# Feature configuration
# Sparse features (categorical) - will be embedded
SPARSE_FEATURES = [
    "placement_id",
    "country",
    "placement_type",
    "avg_device_market_price_bin",
    "count_of_device_requests_bin",
    "time_diff_hrs_bin",
    "sdk_supply_name",
    "do_not_track",
]

# Dense features (numerical) - used directly
DENSE_FEATURES = [
    "prediction_floor",
]

# Target column
TARGET_COLUMN = "net_revenue"

# Embedding dimension (same for all sparse features)
EMBEDDING_DIM = 32


def build_model():
    """
    Build a regression model for net_revenue prediction.
    
    Input shape:
        - 8 sparse features × 32 embedding dim = 256
        - 1 dense feature (prediction_floor) = 1
        - Total: 257
    """
    input_dim = len(SPARSE_FEATURES) * EMBEDDING_DIM + len(DENSE_FEATURES)
    
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    
    # Hidden layers
    x = tf.keras.layers.Dense(128, activation="relu", name="hidden1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(64, activation="relu", name="hidden2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(32, activation="relu", name="hidden3")(x)
    
    # Output - regression (no activation for unbounded output)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
    
    return tf.keras.Model(inputs, outputs, name="net_revenue_model")


def main():
    print("=" * 70)
    print("Net Revenue Model Training")
    print("=" * 70)
    print(f"\nData path: {S3_DATA_PATH}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Checkpoint path: {S3_CHECKPOINT_PATH}")
    print(f"Sparse features: {SPARSE_FEATURES}")
    print(f"Dense features: {DENSE_FEATURES}")
    print(f"Target: {TARGET_COLUMN}")
    
    # Create embedding dims dict (all same dimension)
    embedding_dims = {f: EMBEDDING_DIM for f in SPARSE_FEATURES}
    
    # Configure trainer
    config = TrainerConfig(
        # Thread configuration
        thread_config=ThreadConfig(
            mkl_num_threads=1,
            openmp_num_threads=1,
            tf_intra_op_parallelism=1,
            tf_inter_op_parallelism=16,
        ),
        
        # Feature configuration
        feature_config=FeatureConfig(
            sparse_features=SPARSE_FEATURES,
            dense_features=DENSE_FEATURES,
            embedding_dims=embedding_dims,
            default_embedding_dim=EMBEDDING_DIM,
        ),
        
        # Target configuration - regression task
        target_config=TargetConfig(
            target_columns=[TARGET_COLUMN],
            task_type="regression",
        ),
        
        # No sample weights
        weight_config=WeightConfig(),
        
        # Training settings
        batch_size=256,
        num_workers=256,
        shuffle_data=True,
        
        # Optimizer settings
        model_optimizer="adam",
        model_learning_rate=0.001,
        embedding_optimizer="adagrad",
        embedding_learning_rate=0.01,
        
        # PS configuration
        ps_config=PSConfig(
            num_servers=16,
            embedding_dim=EMBEDDING_DIM,
            server_port_base=51000,
            s3_bucket="demo",
            s3_region="us-east-1",
        ),
        
        # Data paths
        s3_data_path_template=S3_DATA_PATH,
        date_format="%Y-%m-%d",
        
        # Checkpointing
        s3_checkpoint_path=S3_CHECKPOINT_PATH,
        checkpoint_after_each_day=True,
        
        # Decay settings
        decay_after_each_day=True,
        decay_factor=0.99,
        prune_threshold=5,
        
        # Logging
        log_every_n_batches=100,
        verbose=True,
    )
    
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    # Get SparkContext - works on Databricks and standalone Spark
    # On Databricks, sc is already available as a global variable
    # On standalone, you would create: sc = SparkContext("yarn", "Training")
    try:
        # Try to get SparkContext from Databricks/PySpark environment
        from pyspark import SparkContext
        sc = SparkContext.getOrCreate()
        print(f"Using SparkContext: master={sc.master}, appName={sc.appName}")
    except Exception as e:
        print(f"SparkContext not available ({e}), running in local mode")
        sc = None
    
    # Create trainer with SparkContext for distributed execution
    trainer = DistributedTrainer(spark_context=sc, config=config)
    
    try:
        # Set the model
        print("\nInitializing model...")
        trainer.set_model_builder(build_model)
        
        # Get initial stats
        stats = trainer.get_cluster_stats()
        print(f"PS servers: {len(stats.get('servers', []))}")
        
        # Train on date range
        print(f"\nTraining on dates: {START_DATE} to {END_DATE}")
        summary = trainer.train_date_range(
            start_date=START_DATE,
            end_date=END_DATE,
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)
        print(f"Total days: {summary.get('num_days', 0)}")
        print(f"Total samples: {summary.get('total_samples', 0):,}")
        print(f"Total batches: {summary.get('total_batches', 0):,}")
        print(f"Average loss (MSE): {summary.get('avg_loss', 0):.6f}")
        print(f"Total time: {summary.get('total_time_seconds', 0):.1f}s")
        
        print("\nDaily results:")
        for day_result in summary.get('daily_results', []):
            print(f"  {day_result['date']}: "
                  f"samples={day_result['samples']:,}, "
                  f"loss={day_result['avg_loss']:.6f}, "
                  f"time={day_result['time_seconds']:.1f}s")
        
        # Final stats
        final_stats = trainer.get_cluster_stats()
        print(f"\nFinal PS stats:")
        print(f"  Total embeddings: {final_stats.get('total_embeddings', 0):,}")
        print(f"  Total weights: {final_stats.get('total_weights', 0)}")
        
        # Save final checkpoint
        print(f"\nFinal checkpoint saved to: {S3_CHECKPOINT_PATH}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
        
    finally:
        # Cleanup
        print("\nShutting down...")
        trainer.shutdown()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

