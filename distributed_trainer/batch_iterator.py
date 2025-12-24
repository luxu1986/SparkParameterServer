"""Batch iteration over DataFrames for training."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np

from distributed_trainer.config import FeatureConfig, TargetConfig, WeightConfig


@dataclass
class Batch:
    """A batch of training data."""
    
    # Sparse feature token IDs: feature_name -> array of token IDs per sample
    sparse_features: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Dense feature values: feature_name -> array of values
    dense_features: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Target values
    targets: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Optional sample weights
    weights: Optional[np.ndarray] = None
    
    # Batch size
    batch_size: int = 0
    
    def get_all_sparse_tokens(self) -> List[int]:
        """Get all unique sparse token IDs across all features."""
        all_tokens = set()
        for feature_tokens in self.sparse_features.values():
            all_tokens.update(feature_tokens.flatten().tolist())
        return list(all_tokens)
    
    def get_sparse_tokens_by_feature(self) -> Dict[str, List[int]]:
        """Get unique sparse token IDs per feature."""
        return {
            name: list(set(tokens.flatten().tolist()))
            for name, tokens in self.sparse_features.items()
        }
    
    def get_dense_matrix(self) -> np.ndarray:
        """Stack all dense features into a matrix."""
        if not self.dense_features:
            return np.zeros((self.batch_size, 0), dtype=np.float32)
        
        arrays = [
            self.dense_features[name].reshape(-1, 1) if len(self.dense_features[name].shape) == 1
            else self.dense_features[name]
            for name in sorted(self.dense_features.keys())
        ]
        
        return np.hstack(arrays).astype(np.float32)


class BatchIterator:
    """
    Memory-efficient batch iterator for training data.
    
    Features:
    - Configurable batch size
    - Optional shuffling with seed
    - Sample weights support
    - Optional weight normalization within batch
    """
    
    def __init__(
        self,
        data: Any,  # pandas DataFrame or similar
        feature_config: FeatureConfig,
        target_config: TargetConfig,
        weight_config: WeightConfig,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize batch iterator.
        
        Args:
            data: DataFrame with training data
            feature_config: Feature column configuration
            target_config: Target column configuration
            weight_config: Sample weight configuration
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.data = data
        self.feature_config = feature_config
        self.target_config = target_config
        self.weight_config = weight_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        self._num_samples = len(data)
        self._indices: Optional[np.ndarray] = None
    
    @staticmethod
    def _hash_token(value) -> int:
        """
        Convert a token value to an integer hash.
        
        Handles:
        - Integers: returned as-is
        - Strings: hashed to positive int64
        - None/NaN: returns 0
        """
        if value is None:
            return 0
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float):
            if np.isnan(value):
                return 0
            return int(value)
        if isinstance(value, str):
            # Use hash and ensure positive
            h = hash(value)
            # Convert to positive int64 range
            return abs(h) % (2**62)
        # Fallback: try to hash
        try:
            return abs(hash(value)) % (2**62)
        except Exception:
            return 0
    
    def _get_indices(self) -> np.ndarray:
        """Get sample indices, optionally shuffled."""
        if self._indices is not None and not self.shuffle:
            return self._indices
        
        indices = np.arange(self._num_samples)
        
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        
        self._indices = indices
        return indices
    
    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        indices = self._get_indices()
        
        for start_idx in range(0, self._num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self._num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield self._create_batch(batch_indices)
    
    def _create_batch(self, indices: np.ndarray) -> Batch:
        """Create a Batch from the given indices."""
        batch_data = self.data.iloc[indices]
        batch_size = len(indices)
        
        # Extract sparse features
        sparse_features = {}
        for feature in self.feature_config.sparse_features:
            if feature in batch_data.columns:
                values = batch_data[feature].values
                # Handle different data types
                if hasattr(values[0], '__iter__') and not isinstance(values[0], str):
                    # Feature is a list/array of token IDs
                    sparse_features[feature] = np.array([
                        np.array([self._hash_token(t) for t in v], dtype=np.int64) 
                        for v in values
                    ], dtype=object)
                else:
                    # Feature is a single token ID - hash if string
                    hashed_values = np.array([
                        self._hash_token(v) for v in values
                    ], dtype=np.int64)
                    sparse_features[feature] = hashed_values
        
        # Extract dense features
        dense_features = {}
        for feature in self.feature_config.dense_features:
            if feature in batch_data.columns:
                values = batch_data[feature].values
                dense_features[feature] = np.array(values, dtype=np.float32)
        
        # Extract targets
        if len(self.target_config.target_columns) == 1:
            targets = batch_data[self.target_config.target_columns[0]].values
        else:
            targets = batch_data[self.target_config.target_columns].values
        targets = np.array(targets, dtype=np.float32)
        
        # Extract sample weights
        weights = None
        if self.weight_config.has_weights():
            if self.weight_config.weight_column in batch_data.columns:
                weights = batch_data[self.weight_config.weight_column].values
                weights = np.array(weights, dtype=np.float32)
                
                # Fill missing weights with default
                weights = np.nan_to_num(
                    weights,
                    nan=self.weight_config.default_weight
                )
                
                # Normalize weights within batch if configured
                if self.weight_config.normalize_weights and weights.sum() > 0:
                    weights = weights / weights.sum() * len(weights)
            else:
                weights = np.full(
                    batch_size,
                    self.weight_config.default_weight,
                    dtype=np.float32
                )
        
        return Batch(
            sparse_features=sparse_features,
            dense_features=dense_features,
            targets=targets,
            weights=weights,
            batch_size=batch_size,
        )
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (self._num_samples + self.batch_size - 1) // self.batch_size
    
    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return self._num_samples
    
    @property
    def num_batches(self) -> int:
        """Total number of batches."""
        return len(self)


class MultiFileIterator:
    """
    Iterator over multiple data files.
    
    Loads files one at a time to manage memory.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        feature_config: FeatureConfig,
        target_config: TargetConfig,
        weight_config: WeightConfig,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize multi-file iterator.
        
        Args:
            file_paths: List of file paths to iterate
            feature_config: Feature column configuration
            target_config: Target column configuration
            weight_config: Sample weight configuration
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data within each file
            seed: Random seed
        """
        self.file_paths = file_paths
        self.feature_config = feature_config
        self.target_config = target_config
        self.weight_config = weight_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
    
    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches from all files."""
        for file_path in self.file_paths:
            data = self._load_file(file_path)
            
            iterator = BatchIterator(
                data=data,
                feature_config=self.feature_config,
                target_config=self.target_config,
                weight_config=self.weight_config,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                seed=self.seed,
            )
            
            for batch in iterator:
                yield batch
    
    def _load_file(self, path: str) -> Any:
        """
        Load a data file into a DataFrame.
        
        For S3 paths, uses pyarrow with fsspec/s3fs.
        On Databricks, credentials should be configured via instance profile
        or cluster configuration.
        """
        import pandas as pd
        
        # Handle S3 paths
        if path.startswith("s3://") or path.startswith("s3a://"):
            return self._load_s3_file(path)
        
        # Local files
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".json"):
            return pd.read_json(path, lines=True)
        else:
            return pd.read_parquet(path)
    
    def _load_s3_file(self, path: str) -> Any:
        """Load a file from S3."""
        import pandas as pd
        
        try:
            # Try using pandas with pyarrow/s3fs (preferred)
            return pd.read_parquet(path)
        except Exception as e:
            # Fallback: try using boto3 directly
            try:
                import boto3
                import io
                import pyarrow.parquet as pq
                
                # Parse S3 path
                if path.startswith("s3://"):
                    path = path[5:]
                elif path.startswith("s3a://"):
                    path = path[6:]
                
                bucket, key = path.split("/", 1)
                
                s3 = boto3.client("s3")
                response = s3.get_object(Bucket=bucket, Key=key)
                data = response["Body"].read()
                
                table = pq.read_table(io.BytesIO(data))
                return table.to_pandas()
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load S3 file {path}. "
                    f"First error: {e}, Second error: {e2}. "
                    f"Ensure AWS credentials are configured on executors "
                    f"(e.g., via instance profile on Databricks)."
                )

