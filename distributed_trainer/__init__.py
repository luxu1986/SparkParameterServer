"""
Distributed Trainer - PySpark + TensorFlow 2.0 training with Parameter Server.

This library orchestrates day-by-day incremental training using PySpark and
TensorFlow 2.0, integrating with the pyspark_ps Parameter Server library for
both sparse embeddings and dense model weights storage.
"""

from distributed_trainer.config import (
    ThreadConfig,
    FeatureConfig,
    TargetConfig,
    WeightConfig,
    TrainerConfig,
)
from distributed_trainer.trainer import DistributedTrainer
from distributed_trainer.worker import WorkerTrainer, WorkerTrainingResult, Batch
from distributed_trainer.thread_config import configure_threads, get_optimal_thread_config

__version__ = "0.1.0"
__all__ = [
    "DistributedTrainer",
    "WorkerTrainer",
    "WorkerTrainingResult",
    "Batch",
    "ThreadConfig",
    "FeatureConfig",
    "TargetConfig",
    "WeightConfig",
    "TrainerConfig",
    "configure_threads",
    "get_optimal_thread_config",
]

