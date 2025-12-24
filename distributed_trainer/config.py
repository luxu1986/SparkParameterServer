"""Configuration dataclasses for Distributed Trainer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json

from distributed_trainer.thread_config import ThreadConfig

# Import PSConfig from pyspark_ps
try:
    from pyspark_ps import PSConfig
except ImportError:
    # Fallback for when pyspark_ps is not installed
    @dataclass
    class PSConfig:
        num_servers: int = 4
        embedding_dim: int = 64
        embedding_optimizer: str = "adagrad"
        weight_optimizer: str = "adam"
        s3_bucket: str = ""
        server_port_base: int = 50000


@dataclass
class FeatureConfig:
    """Configuration for feature columns."""
    
    # Sparse feature columns (will use embeddings from PS)
    sparse_features: List[str] = field(default_factory=list)
    
    # Dense feature columns (numerical features)
    dense_features: List[str] = field(default_factory=list)
    
    # Embedding dimensions for each sparse feature
    embedding_dims: Dict[str, int] = field(default_factory=dict)
    
    # Default embedding dimension if not specified
    default_embedding_dim: int = 64
    
    def get_embedding_dim(self, feature_name: str) -> int:
        """Get embedding dimension for a feature."""
        return self.embedding_dims.get(feature_name, self.default_embedding_dim)
    
    def get_total_embedding_dim(self) -> int:
        """Get total dimension of all sparse feature embeddings concatenated."""
        return sum(
            self.get_embedding_dim(f) for f in self.sparse_features
        )
    
    def get_total_dense_dim(self) -> int:
        """Get total dimension of dense features."""
        return len(self.dense_features)
    
    def get_model_input_dim(self) -> int:
        """Get total input dimension for the model (embeddings + dense)."""
        return self.get_total_embedding_dim() + self.get_total_dense_dim()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sparse_features": self.sparse_features,
            "dense_features": self.dense_features,
            "embedding_dims": self.embedding_dims,
            "default_embedding_dim": self.default_embedding_dim,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureConfig":
        return cls(**data)


@dataclass
class TargetConfig:
    """Configuration for target/label columns."""
    
    # Target column name(s)
    target_columns: List[str] = field(default_factory=list)
    
    # Task type: "binary_classification", "multiclass", "regression", "multi_label"
    task_type: str = "binary_classification"
    
    # Number of classes (for multiclass)
    num_classes: Optional[int] = None
    
    def get_output_dim(self) -> int:
        """Get output dimension based on task type."""
        if self.task_type == "binary_classification":
            return 1
        elif self.task_type == "multiclass":
            return self.num_classes or 2
        elif self.task_type == "multi_label":
            return len(self.target_columns)
        else:  # regression
            return len(self.target_columns)
    
    def get_activation(self) -> str:
        """Get output activation function for task type."""
        if self.task_type == "binary_classification":
            return "sigmoid"
        elif self.task_type == "multiclass":
            return "softmax"
        elif self.task_type == "multi_label":
            return "sigmoid"
        else:  # regression
            return "linear"
    
    def get_loss_name(self) -> str:
        """Get default loss function name for task type."""
        if self.task_type == "binary_classification":
            return "binary_crossentropy"
        elif self.task_type == "multiclass":
            return "sparse_categorical_crossentropy"
        elif self.task_type == "multi_label":
            return "binary_crossentropy"
        else:  # regression
            return "mse"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_columns": self.target_columns,
            "task_type": self.task_type,
            "num_classes": self.num_classes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetConfig":
        return cls(**data)


@dataclass
class WeightConfig:
    """Configuration for sample weight column (optional)."""
    
    # Sample weight column name, None means no weighting
    weight_column: Optional[str] = None
    
    # Whether to normalize weights within each batch
    normalize_weights: bool = False
    
    # Default weight value for missing weights
    default_weight: float = 1.0
    
    def has_weights(self) -> bool:
        """Check if sample weights are configured."""
        return self.weight_column is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_column": self.weight_column,
            "normalize_weights": self.normalize_weights,
            "default_weight": self.default_weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightConfig":
        return cls(**data)


@dataclass
class TrainerConfig:
    """Complete configuration for DistributedTrainer."""
    
    # Thread configuration
    thread_config: ThreadConfig = field(default_factory=ThreadConfig)
    
    # Feature, target, and weight configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    target_config: TargetConfig = field(default_factory=TargetConfig)
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    
    # Training configuration
    batch_size: int = 1024
    num_workers: int = 4
    shuffle_data: bool = True
    shuffle_seed: Optional[int] = None
    
    # Model optimizer configuration (used by PS weight_store)
    model_optimizer: str = "adam"
    model_learning_rate: float = 0.001
    model_optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding optimizer configuration (used by PS embedding_store)
    embedding_optimizer: str = "adagrad"
    embedding_learning_rate: float = 0.01
    embedding_optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # PS configuration
    ps_config: PSConfig = field(default_factory=PSConfig)
    
    # Data configuration
    s3_data_path_template: str = ""  # e.g., "s3://bucket/data/dt={date}/"
    date_format: str = "%Y-%m-%d"
    
    # Checkpointing
    s3_checkpoint_path: str = ""
    checkpoint_after_each_day: bool = True
    
    # Decay configuration
    decay_after_each_day: bool = True
    decay_method: str = "multiply"
    decay_factor: float = 0.99
    prune_threshold: int = 5
    
    # Logging
    log_every_n_batches: int = 100
    verbose: bool = True
    
    # Progress monitoring (driver prints summary after this many updates)
    progress_update_threshold: int = 100000
    
    def validate(self):
        """Validate configuration values."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if not self.feature_config.sparse_features and not self.feature_config.dense_features:
            raise ValueError("Must specify at least one sparse or dense feature")
        if not self.target_config.target_columns:
            raise ValueError("Must specify at least one target column")
        if self.decay_factor <= 0 or self.decay_factor > 1:
            raise ValueError("decay_factor must be in (0, 1]")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "thread_config": {
                "mkl_num_threads": self.thread_config.mkl_num_threads,
                "openmp_num_threads": self.thread_config.openmp_num_threads,
                "tf_intra_op_parallelism": self.thread_config.tf_intra_op_parallelism,
                "tf_inter_op_parallelism": self.thread_config.tf_inter_op_parallelism,
            },
            "feature_config": self.feature_config.to_dict(),
            "target_config": self.target_config.to_dict(),
            "weight_config": self.weight_config.to_dict(),
            "ps_config": self.ps_config.to_dict() if hasattr(self.ps_config, 'to_dict') else self.ps_config,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle_data": self.shuffle_data,
            "shuffle_seed": self.shuffle_seed,
            "model_optimizer": self.model_optimizer,
            "model_learning_rate": self.model_learning_rate,
            "model_optimizer_params": self.model_optimizer_params,
            "embedding_optimizer": self.embedding_optimizer,
            "embedding_learning_rate": self.embedding_learning_rate,
            "embedding_optimizer_params": self.embedding_optimizer_params,
            "s3_data_path_template": self.s3_data_path_template,
            "date_format": self.date_format,
            "s3_checkpoint_path": self.s3_checkpoint_path,
            "checkpoint_after_each_day": self.checkpoint_after_each_day,
            "decay_after_each_day": self.decay_after_each_day,
            "decay_method": self.decay_method,
            "decay_factor": self.decay_factor,
            "prune_threshold": self.prune_threshold,
            "log_every_n_batches": self.log_every_n_batches,
            "verbose": self.verbose,
            "progress_update_threshold": self.progress_update_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        """Create config from dictionary."""
        thread_config = ThreadConfig(**data.get("thread_config", {}))
        feature_config = FeatureConfig.from_dict(data.get("feature_config", {}))
        target_config = TargetConfig.from_dict(data.get("target_config", {}))
        weight_config = WeightConfig.from_dict(data.get("weight_config", {}))
        
        # Reconstruct PSConfig from dict
        ps_config_data = data.get("ps_config", {})
        if isinstance(ps_config_data, dict):
            from pyspark_ps import PSConfig
            ps_config = PSConfig.from_dict(ps_config_data)
        else:
            ps_config = ps_config_data
        
        return cls(
            thread_config=thread_config,
            feature_config=feature_config,
            target_config=target_config,
            weight_config=weight_config,
            ps_config=ps_config,
            batch_size=data.get("batch_size", 1024),
            num_workers=data.get("num_workers", 4),
            shuffle_data=data.get("shuffle_data", True),
            shuffle_seed=data.get("shuffle_seed"),
            model_optimizer=data.get("model_optimizer", "adam"),
            model_learning_rate=data.get("model_learning_rate", 0.001),
            model_optimizer_params=data.get("model_optimizer_params", {}),
            embedding_optimizer=data.get("embedding_optimizer", "adagrad"),
            embedding_learning_rate=data.get("embedding_learning_rate", 0.01),
            embedding_optimizer_params=data.get("embedding_optimizer_params", {}),
            s3_data_path_template=data.get("s3_data_path_template", ""),
            date_format=data.get("date_format", "%Y-%m-%d"),
            s3_checkpoint_path=data.get("s3_checkpoint_path", ""),
            checkpoint_after_each_day=data.get("checkpoint_after_each_day", True),
            decay_after_each_day=data.get("decay_after_each_day", True),
            decay_method=data.get("decay_method", "multiply"),
            decay_factor=data.get("decay_factor", 0.99),
            prune_threshold=data.get("prune_threshold", 5),
            log_every_n_batches=data.get("log_every_n_batches", 100),
            verbose=data.get("verbose", True),
            progress_update_threshold=data.get("progress_update_threshold", 100000),
        )
    
    def to_json(self) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrainerConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

