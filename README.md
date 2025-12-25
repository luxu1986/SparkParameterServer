# PySpark ML Platform

A complete distributed machine learning platform for PySpark, combining a high-performance **Parameter Server** for embeddings and model weights with a **Distributed Trainer** for TensorFlow 2.0 day-by-day incremental training.

## Overview

This platform provides two integrated libraries:

| Library | Description |
|---------|-------------|
| **pyspark_ps** | Distributed Parameter Server for sparse embeddings and dense model weights |
| **distributed_trainer** | TensorFlow 2.0 training orchestration with day-by-day incremental learning |

## Features

### Parameter Server (`pyspark_ps`)
- **Distributed Architecture**: Scalable PS cluster with consistent-hash sharded embedding storage
- **Sparse Embeddings**: Efficient storage and updates for billion-scale embedding tables
- **Dense Weights**: Replicated weight storage across servers
- **Multiple Optimizers**: SGD (momentum), Adam, Adagrad, FTRL-Proximal
- **Barrier Synchronization**: Coordinate workers for synchronized training
- **Embedding Decay/Pruning**: Manage memory with decay and pruning strategies
- **S3 Checkpointing**: Save and restore models to/from S3

### Distributed Trainer (`distributed_trainer`)
- **Day-by-day Training**: Incremental training on date-partitioned data
- **TensorFlow 2.0**: Native Keras model support with numpy ↔ tensor conversion
- **Automatic Orchestration**: Data discovery, partition distribution, worker coordination
- **Thread Optimization**: MKL, OpenMP, and TensorFlow threading configuration
- **Sample Weighting**: Support for weighted training examples
- **Automatic Decay**: Decay embeddings after each training day

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DRIVER NODE                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      DistributedTrainer                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      PSMainClient                               │  │  │
│  │  │  - Server lifecycle (start/shutdown)                            │  │  │
│  │  │  - Barrier coordination                                         │  │  │
│  │  │  - Decay operations                                             │  │  │
│  │  │  - S3 checkpointing                                             │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │  - Day-by-day training orchestration                                  │  │
│  │  - Data partition distribution                                        │  │
│  │  - Broadcast TF model structure                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTOR NODES                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │  WorkerTrainer 0 │  │  WorkerTrainer 1 │  │  WorkerTrainer N │           │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │           │
│  │  │PSWorker    │  │  │  │PSWorker    │  │  │  │PSWorker    │  │           │
│  │  │Client      │  │  │  │Client      │  │  │  │Client      │  │           │
│  │  └────────────┘  │  │  └────────────┘  │  │  └────────────┘  │           │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘           │
└───────────┼─────────────────────┼─────────────────────┼─────────────────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PS Server Cluster                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ Server 0 │  │ Server 1 │  │ Server 2 │  │ Server N │                     │
│  │ Shard 0  │  │ Shard 1  │  │ Shard 2  │  │ Shard N  │                     │
│  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤                     │
│  │embedding │  │embedding │  │embedding │  │embedding │                     │
│  │  store   │  │  store   │  │  store   │  │  store   │                     │
│  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤                     │
│  │ weight   │  │ weight   │  │ weight   │  │ weight   │                     │
│  │  store   │  │  store   │  │  store   │  │  store   │                     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install both libraries
pip install -e .

# Install with optional dependencies
pip install -e ".[s3]"          # S3 filesystem support (s3fs)
pip install -e ".[compression]" # LZ4 compression
pip install -e ".[dev]"         # Development tools (pytest, black, mypy)
pip install -e ".[all]"         # All optional dependencies
```

## Quick Start

### High-Level: Using DistributedTrainer (Recommended)

For most use cases, use the `DistributedTrainer` which handles PS lifecycle automatically:

```python
from pyspark import SparkContext
from pyspark_ps import PSConfig
from distributed_trainer import (
    DistributedTrainer, TrainerConfig, FeatureConfig,
    TargetConfig, WeightConfig, ThreadConfig
)
import tensorflow as tf

# Configure training
config = TrainerConfig(
    thread_config=ThreadConfig(mkl_num_threads=4, openmp_num_threads=4),
    feature_config=FeatureConfig(
        sparse_features=["user_id", "item_id", "category"],
        dense_features=["price", "age"],
        embedding_dims={"user_id": 64, "item_id": 64, "category": 16}
    ),
    target_config=TargetConfig(
        target_columns=["click"],
        task_type="binary_classification"
    ),
    weight_config=WeightConfig(
        weight_column="sample_weight",
        normalize_weights=False
    ),
    batch_size=2048,
    num_workers=8,
    ps_config=PSConfig(num_servers=4, s3_bucket="my-bucket"),
    s3_data_path_template="s3://my-bucket/data/dt={date}/",
    s3_checkpoint_path="s3://my-bucket/checkpoints/",
    decay_after_each_day=True,
    decay_factor=0.99
)

# Create trainer
sc = SparkContext("yarn", "Training")
trainer = DistributedTrainer(sc, config)

# Define model
def build_model():
    inputs = tf.keras.Input(shape=(176,))  # embeddings + dense features
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

# Train
trainer.set_model_builder(build_model)
summary = trainer.train_date_range("2025-12-20", "2025-12-25")
print(summary["daily_results"])

# Cleanup
trainer.shutdown()
```

### Low-Level: Using Parameter Server Directly

For custom training loops or non-TensorFlow workloads:

```python
from pyspark_ps import PSMainClient, PSWorkerClient, PSConfig

# Configuration
config = PSConfig(
    num_servers=4,
    embedding_dim=128,
    embedding_optimizer="adagrad",
    weight_optimizer="adam",
)

# === DRIVER CODE ===
main_client = PSMainClient(spark_context, config)
server_info = main_client.start_servers()

# Initialize model weights
main_client.init_weights({
    "fc1/kernel": (128, 64),
    "fc1/bias": (64,),
    "fc2/kernel": (64, 1),
    "fc2/bias": (1,),
})

# Broadcast to executors
server_info_bc = sc.broadcast(server_info)
config_bc = sc.broadcast(config.to_dict())

# === EXECUTOR CODE ===
def train_partition(iterator):
    worker = PSWorkerClient(
        server_info_bc.value,
        PSConfig.from_dict(config_bc.value)
    )
    
    for batch in iterator:
        # Pull from PS
        embeddings = worker.pull_embeddings(batch.token_ids)
        weights = worker.pull_model()
        
        # Your training logic
        weight_grads, embedding_grads = compute_gradients(
            weights, embeddings, batch
        )
        
        # Push to PS
        worker.push_gradients(weight_grads, embedding_grads)
    
    worker.close()

# Train
rdd.mapPartitions(train_partition).count()

# Decay and checkpoint
main_client.decay_embeddings(method="multiply", factor=0.99)
main_client.save_to_s3("s3://bucket/checkpoints/epoch_1")
main_client.shutdown_servers()
```

## Configuration

### Parameter Server Configuration (`PSConfig`)

```python
PSConfig(
    # Server settings
    num_servers=4,                      # Number of PS servers
    server_port_base=50000,             # Base port for servers
    
    # Embedding settings
    embedding_dim=64,                   # Embedding dimension
    embedding_init="normal",            # "zeros", "random", "normal"
    embedding_init_scale=0.01,          # Initialization scale
    embedding_optimizer="adagrad",      # "sgd", "adam", "adagrad", "ftrl"
    
    # Weight settings
    weight_optimizer="adam",            # Optimizer for dense weights
    
    # Communication
    batch_size=10000,                   # Max batch size
    timeout_seconds=60.0,               # RPC timeout
    compression=True,                   # Enable compression
    
    # Decay settings
    auto_decay=False,                   # Automatic decay
    decay_interval_steps=1000,          # Steps between decays
    decay_factor=0.99,                  # Decay factor
    prune_threshold=5,                  # Min count for pruning
    
    # S3 settings
    s3_bucket="my-bucket",
    s3_region="us-east-1",
    s3_checkpoint_prefix="checkpoints/",
)
```

### Trainer Configuration (`TrainerConfig`)

```python
TrainerConfig(
    # Thread configuration (call before numpy/tensorflow import)
    thread_config=ThreadConfig(
        mkl_num_threads=4,
        openmp_num_threads=4,
        tf_intra_op_parallelism=4,
        tf_inter_op_parallelism=2
    ),
    
    # Feature configuration
    feature_config=FeatureConfig(
        sparse_features=["user_id", "item_id"],    # Categorical → embeddings
        dense_features=["price", "age"],            # Numerical features
        embedding_dims={"user_id": 64, "item_id": 32},
        default_embedding_dim=64
    ),
    
    # Target configuration
    target_config=TargetConfig(
        target_columns=["click"],
        task_type="binary_classification",  # or "multiclass", "regression", "multi_label"
        num_classes=None  # For multiclass
    ),
    
    # Sample weight configuration
    weight_config=WeightConfig(
        weight_column="sample_weight",
        normalize_weights=False,
        default_weight=1.0
    ),
    
    # Training settings
    batch_size=2048,
    num_workers=8,
    shuffle_data=True,
    
    # Optimizer settings
    model_optimizer="adam",
    model_learning_rate=0.001,
    embedding_optimizer="adagrad",
    embedding_learning_rate=0.01,
    
    # Data path template
    s3_data_path_template="s3://bucket/data/dt={date}/",
    
    # Checkpointing
    s3_checkpoint_path="s3://bucket/checkpoints/",
    checkpoint_after_each_day=True,
    
    # Decay settings
    decay_after_each_day=True,
    decay_factor=0.99,
)
```

## Optimizers

### SGD with Momentum
```python
PSConfig(
    embedding_optimizer="sgd",
    optimizer_configs={
        "sgd": {"learning_rate": 0.01, "momentum": 0.9, "nesterov": False}
    }
)
```

### Adam
```python
PSConfig(
    weight_optimizer="adam",
    optimizer_configs={
        "adam": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
    }
)
```

### Adagrad (recommended for sparse embeddings)
```python
PSConfig(
    embedding_optimizer="adagrad",
    optimizer_configs={
        "adagrad": {"learning_rate": 0.01, "epsilon": 1e-10, "initial_accumulator": 0.1}
    }
)
```

### FTRL-Proximal (optimal for sparse features with L1 regularization)
```python
PSConfig(
    embedding_optimizer="ftrl",
    optimizer_configs={
        "ftrl": {"alpha": 0.05, "beta": 1.0, "l1": 0.001, "l2": 0.0}
    }
)
```

## Training Flow

The `DistributedTrainer` orchestrates training automatically:

1. **Initialization**
   - Driver builds TF model and extracts weights
   - Driver initializes PS `weight_store` with model weights
   - Driver broadcasts `model_builder` function to workers

2. **Per-Day Training Loop**
   - Discover parquet files: `s3://bucket/data/dt=2025-12-20/*.parquet`
   - Distribute partitions to workers (size-balanced)
   - Workers train batch-by-batch:
     - Pull weights from PS `weight_store`
     - Pull embeddings from PS `embedding_store`
     - Forward/backward pass (TensorFlow GradientTape)
     - Push gradients to PS (optimizer applies updates)
   - Save checkpoint to S3 (optional)
   - Decay embeddings (optional)

3. **Cleanup**
   - Shutdown PS servers
   - Release resources

## Embedding Management

### Decay
Reduce embedding values over time to manage memory and prevent overfitting:

```python
# Multiplicative decay: embeddings *= 0.99, counts *= 0.99
main_client.decay_embeddings(method="multiply", factor=0.99)
```

### Pruning
Remove infrequently accessed embeddings:

```python
# Remove embeddings with update_count < 5
result = main_client.decay_embeddings(method="prune", min_count=5)
print(f"Pruned {result['results'][0]['pruned']} embeddings")
```

## Barrier Synchronization

Coordinate workers for synchronized operations:

```python
# Driver: Create barrier for 4 workers
main_client.create_barrier("epoch_sync", num_workers=4)

# Workers: Enter barrier (blocks until released)
worker.enter_barrier("epoch_sync")

# Driver: Wait for all workers, perform operation, release
main_client.wait_barrier("epoch_sync")
main_client.decay_embeddings(...)  # Safe: all workers paused
main_client.release_barrier("epoch_sync")
```

## S3 Checkpointing

```python
# Save checkpoint
main_client.save_to_s3(
    "s3://bucket/model/checkpoint_001",
    save_model=True,
    save_embeddings=True,
    save_optimizer_states=True
)

# List available checkpoints
checkpoints = main_client.list_checkpoints("s3://bucket/model/")
for ckpt in checkpoints:
    print(f"{ckpt.s3_path}: {ckpt.embedding_count} embeddings")

# Load checkpoint
main_client.load_from_s3("s3://bucket/model/checkpoint_001")

# Delete old checkpoint
main_client.delete_checkpoint("s3://bucket/model/checkpoint_000")
```

## Running Tests

```bash
# Run all tests
pytest pyspark_ps/tests/
pytest distributed_trainer/tests/

# Run with coverage
pytest --cov=pyspark_ps --cov=distributed_trainer
```

## Running Examples

```bash
# Parameter Server examples
python -m pyspark_ps.examples.simple_training
python -m pyspark_ps.examples.distributed_embedding

# Distributed Trainer examples
python -m distributed_trainer.examples.simple_training
python -m distributed_trainer.examples.train_demo
```

## Project Structure

```
├── pyspark_ps/                     # Parameter Server Library
│   ├── client/
│   │   ├── main_client.py          # Driver-side client (PSMainClient)
│   │   ├── worker_client.py        # Executor-side client (PSWorkerClient)
│   │   └── barrier.py              # Barrier synchronization
│   ├── server/
│   │   ├── ps_server.py            # Server implementation
│   │   ├── embedding_store.py      # Sparse embedding storage
│   │   ├── weight_store.py         # Dense weight storage
│   │   ├── shard_manager.py        # Consistent-hash sharding
│   │   └── update_counter.py       # Embedding update counting
│   ├── optimizers/
│   │   ├── base.py                 # BaseOptimizer interface
│   │   ├── sgd.py                  # SGD with momentum
│   │   ├── adam.py                 # Adam optimizer
│   │   ├── adagrad.py              # Adagrad optimizer
│   │   └── ftrl.py                 # FTRL-Proximal optimizer
│   ├── communication/
│   │   ├── protocol.py             # Message types and wire format
│   │   ├── rpc_handler.py          # RPC client/server
│   │   └── serialization.py        # Serialization with compression
│   ├── storage/
│   │   ├── s3_backend.py           # S3 storage backend
│   │   ├── checkpoint.py           # Checkpoint manager
│   │   └── serialization.py        # Storage serialization
│   ├── utils/
│   │   ├── config.py               # PSConfig
│   │   ├── sharding.py             # Consistent hash ring
│   │   └── logging.py              # Logging utilities
│   ├── examples/                   # PS examples
│   └── tests/                      # PS tests
│
├── distributed_trainer/            # Distributed Trainer Library
│   ├── trainer.py                  # DistributedTrainer (driver)
│   ├── worker.py                   # WorkerTrainer (executor)
│   ├── tf_model_wrapper.py         # TensorFlow numpy ↔ tensor conversion
│   ├── data_loader.py              # S3 parquet discovery & partitioning
│   ├── batch_iterator.py           # Batch iteration
│   ├── config.py                   # TrainerConfig, FeatureConfig, etc.
│   ├── thread_config.py            # MKL/OpenMP/TF threading
│   ├── examples/                   # Trainer examples
│   └── tests/                      # Trainer tests
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # This file
```

## Performance Tips

1. **Batch Operations**: Always use batch pull/push for better throughput
2. **Compression**: Enable compression for large payloads over network
3. **Partitioning**: Match number of workers to available parallelism
4. **Pruning**: Regularly prune low-frequency embeddings to manage memory
5. **Optimizer Choice**: Use Adagrad or FTRL for sparse embeddings, Adam for dense weights
6. **Thread Configuration**: Set thread counts before importing numpy/tensorflow

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- pandas >= 1.3.0
- PySpark >= 3.2.0
- TensorFlow >= 2.8.0 (for distributed_trainer)
- pyarrow >= 8.0.0 (for Parquet)
- msgpack >= 1.0.0 (for serialization)
- cloudpickle >= 2.0.0 (for model serialization)
- boto3 >= 1.20.0 (for S3)

## License

Apache License 2.0
