"""Configuration management for PySpark Parameter Server."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class PSConfig:
    """
    Configuration for the Parameter Server system.
    
    Attributes:
        num_servers: Number of PS server instances
        server_port_base: Base port for server communication
        server_host: Host to bind servers (default: auto-discover)
        
        embedding_dim: Dimension of embedding vectors
        embedding_init: Initialization strategy ("zeros", "random", "normal")
        embedding_init_scale: Scale for random/normal initialization
        embedding_optimizer: Optimizer for embeddings
        
        weight_optimizer: Optimizer for dense weights
        
        batch_size: Maximum batch size for operations
        timeout_seconds: Timeout for RPC calls
        compression: Enable compression for large payloads
        compression_algorithm: Compression algorithm ("lz4", "zstd", "snappy")
        
        auto_decay: Enable automatic embedding decay
        decay_interval_steps: Steps between decay operations
        decay_factor: Multiplicative decay factor
        prune_threshold: Minimum update count for pruning
        
        s3_bucket: S3 bucket for checkpoints
        s3_region: AWS region
        s3_endpoint_url: Custom S3 endpoint (for MinIO, etc.)
        s3_checkpoint_prefix: Prefix for checkpoint paths
        s3_compression: Compression for S3 files
        s3_multipart_threshold: Threshold for multipart uploads
        s3_max_concurrency: Max concurrent S3 operations
        checkpoint_on_shutdown: Auto-save on graceful shutdown
        
        optimizer_configs: Per-optimizer configuration overrides
    """
    
    # Server settings
    num_servers: int = 4
    server_port_base: int = 50000
    server_host: Optional[str] = None
    
    # Embedding settings
    embedding_dim: int = 64
    embedding_init: str = "normal"
    embedding_init_scale: float = 0.01
    embedding_optimizer: str = "adagrad"
    
    # Weight settings
    weight_optimizer: str = "adam"
    
    # Communication settings
    batch_size: int = 10000
    timeout_seconds: float = 60.0
    compression: bool = True
    compression_algorithm: str = "lz4"
    max_message_size: int = 100 * 1024 * 1024  # 100MB
    
    # Decay settings
    auto_decay: bool = False
    decay_interval_steps: int = 1000
    decay_factor: float = 0.99
    prune_threshold: int = 5
    
    # S3 Storage settings
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_endpoint_url: Optional[str] = None
    s3_checkpoint_prefix: str = "checkpoints/"
    s3_compression: str = "zstd"
    s3_multipart_threshold: int = 8 * 1024 * 1024
    s3_max_concurrency: int = 10
    checkpoint_on_shutdown: bool = True
    
    # Optimizer configurations
    optimizer_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "sgd": {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "nesterov": False
        },
        "adam": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.0
        },
        "adagrad": {
            "learning_rate": 0.01,
            "epsilon": 1e-10,
            "initial_accumulator": 0.1
        },
        "ftrl": {
            "alpha": 0.05,
            "beta": 1.0,
            "l1": 0.0,
            "l2": 0.0
        }
    })
    
    # Sharding settings
    virtual_nodes_per_server: int = 150
    
    # Memory settings
    use_mmap: bool = False
    max_embeddings_per_shard: int = 10_000_000
    
    # Threading settings
    num_worker_threads: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "num_servers": self.num_servers,
            "server_port_base": self.server_port_base,
            "server_host": self.server_host,
            "embedding_dim": self.embedding_dim,
            "embedding_init": self.embedding_init,
            "embedding_init_scale": self.embedding_init_scale,
            "embedding_optimizer": self.embedding_optimizer,
            "weight_optimizer": self.weight_optimizer,
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "compression": self.compression,
            "compression_algorithm": self.compression_algorithm,
            "max_message_size": self.max_message_size,
            "auto_decay": self.auto_decay,
            "decay_interval_steps": self.decay_interval_steps,
            "decay_factor": self.decay_factor,
            "prune_threshold": self.prune_threshold,
            "s3_bucket": self.s3_bucket,
            "s3_region": self.s3_region,
            "s3_endpoint_url": self.s3_endpoint_url,
            "s3_checkpoint_prefix": self.s3_checkpoint_prefix,
            "s3_compression": self.s3_compression,
            "s3_multipart_threshold": self.s3_multipart_threshold,
            "s3_max_concurrency": self.s3_max_concurrency,
            "checkpoint_on_shutdown": self.checkpoint_on_shutdown,
            "optimizer_configs": self.optimizer_configs,
            "virtual_nodes_per_server": self.virtual_nodes_per_server,
            "use_mmap": self.use_mmap,
            "max_embeddings_per_shard": self.max_embeddings_per_shard,
            "num_worker_threads": self.num_worker_threads,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PSConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})
    
    def to_json(self) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PSConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_optimizer_config(self, optimizer_name: str) -> Dict[str, Any]:
        """Get configuration for a specific optimizer."""
        return self.optimizer_configs.get(optimizer_name, {})
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.num_servers < 1:
            raise ValueError("num_servers must be at least 1")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if self.embedding_init not in ("zeros", "random", "normal"):
            raise ValueError(f"Invalid embedding_init: {self.embedding_init}")
        if self.embedding_optimizer not in ("sgd", "adam", "adagrad", "ftrl"):
            raise ValueError(f"Invalid embedding_optimizer: {self.embedding_optimizer}")
        if self.weight_optimizer not in ("sgd", "adam", "adagrad", "ftrl"):
            raise ValueError(f"Invalid weight_optimizer: {self.weight_optimizer}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.decay_factor <= 0 or self.decay_factor > 1:
            raise ValueError("decay_factor must be in (0, 1]")

