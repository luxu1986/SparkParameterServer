"""Worker Trainer for executor nodes."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from distributed_trainer.config import TrainerConfig
from distributed_trainer.thread_config import configure_threads, configure_tensorflow_threads
from distributed_trainer.batch_iterator import Batch, MultiFileIterator
from distributed_trainer.tf_model_wrapper import TFModelWrapper


@dataclass
class WorkerTrainingResult:
    """Result from a worker's complete training run (metrics only)."""
    
    worker_id: int
    total_samples: int = 0
    total_batches: int = 0
    total_weighted_samples: float = 0.0  # sum of sample weights
    avg_loss: float = 0.0
    total_loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "total_weighted_samples": self.total_weighted_samples,
            "avg_loss": self.avg_loss,
            "total_loss": self.total_loss,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
        }


class WorkerTrainer:
    """
    Worker Trainer running on Spark EXECUTOR nodes.
    
    Responsibilities:
    - Configure threads (MKL, OpenMP, TF)
    - Create PS worker client
    - Build TF model from broadcast model_builder (structure only)
    - Load assigned data partitions
    - Train batch-by-batch:
      - Pull model weights from PS weight_store
      - Pull embeddings from PS embedding_store
      - Forward/backward pass
      - Push model gradients to PS weight_store
      - Push embedding gradients to PS embedding_store
    - Return metrics only
    """
    
    def __init__(
        self,
        worker_id: int,
        config: TrainerConfig,
        server_info: List[Any],  # List[ServerInfo]
        model_builder: Callable[[], Any]  # Callable[[], tf.keras.Model]
    ):
        """
        Initialize WorkerTrainer on executor.
        
        Steps:
        1. Configure threads
        2. Create PSWorkerClient
        3. Build TF model structure using model_builder (no weights yet)
        4. Create TFModelWrapper for numpy <-> tensor conversion
        
        Args:
            worker_id: Unique identifier for this worker
            config: Training configuration
            server_info: List of PS server info (broadcast from driver)
            model_builder: Function that builds the TF model
        """
        self.worker_id = worker_id
        self.config = config
        self.server_info = server_info
        self.model_builder = model_builder
        
        # Configure threading (before TF import)
        configure_threads(config.thread_config)
        
        # Import and configure TensorFlow
        configure_tensorflow_threads(config.thread_config)
        
        # Create PS worker client
        from pyspark_ps import PSWorkerClient, PSConfig
        
        # Reconstruct PSConfig if needed
        ps_config = config.ps_config
        if isinstance(ps_config, dict):
            ps_config = PSConfig(**ps_config)
        
        self._ps_client = PSWorkerClient(
            server_info=server_info,
            config=ps_config,
            client_id=f"worker_{worker_id}"
        )
        
        # Build TF model structure
        self._model_wrapper = TFModelWrapper(model_builder=model_builder)
        self._model_wrapper.build_model()
        
        # Get loss function
        self._loss_fn = self._create_loss_function()
        
        # Statistics
        self._batches_processed = 0
        self._samples_processed = 0
    
    def _create_loss_function(self) -> Any:
        """Create the appropriate loss function based on config."""
        import tensorflow as tf
        
        loss_name = self.config.target_config.get_loss_name()
        
        loss_map = {
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE  # For sample weighting
            ),
            "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE
            ),
            "mse": tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            "mae": tf.keras.losses.MeanAbsoluteError(
                reduction=tf.keras.losses.Reduction.NONE
            ),
        }
        
        return loss_map.get(loss_name, loss_map["binary_crossentropy"])
    
    def train_partitions(
        self,
        partition_paths: List[str]
    ) -> WorkerTrainingResult:
        """
        Train on assigned partitions.
        
        Process:
        1. Load parquet files into DataFrame
        2. Create batch iterator
        3. For each batch:
           a. Pull model weights from PS (via ps_client.pull_model)
           b. Load weights into local TF model (numpy -> tensor)
           c. Extract sparse feature tokens
           d. Pull embeddings from PS (via ps_client.pull_embeddings)
           e. Forward pass with GradientTape
           f. Compute weighted loss
           g. Compute gradients (tensor -> numpy)
           h. Push model gradients to PS (via ps_client.push_gradients)
           i. Push embedding gradients to PS (via ps_client.push_gradients)
        4. Enter barrier synchronization
        5. Return WorkerTrainingResult (metrics only)
        
        Args:
            partition_paths: List of S3/local paths to training data
            
        Returns:
            WorkerTrainingResult with training metrics
        """
        import tensorflow as tf
        
        start_time = time.time()
        
        # Create iterator over all partitions
        iterator = MultiFileIterator(
            file_paths=partition_paths,
            feature_config=self.config.feature_config,
            target_config=self.config.target_config,
            weight_config=self.config.weight_config,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_data,
            seed=self.config.shuffle_seed,
        )
        
        total_loss = 0.0
        total_samples = 0
        total_weighted_samples = 0.0
        total_batches = 0
        
        for batch in iterator:
            batch_result = self._train_batch(batch)
            
            total_loss += batch_result["loss"] * batch_result["batch_size"]
            total_samples += batch_result["batch_size"]
            total_weighted_samples += batch_result["weighted_samples"]
            total_batches += 1
            
            # Log progress
            if self.config.verbose and total_batches % self.config.log_every_n_batches == 0:
                avg_loss = total_loss / max(total_samples, 1)
                print(
                    f"Worker {self.worker_id}: batch {total_batches}, "
                    f"samples {total_samples}, avg_loss {avg_loss:.6f}"
                )
        
        training_time = time.time() - start_time
        
        return WorkerTrainingResult(
            worker_id=self.worker_id,
            total_samples=total_samples,
            total_batches=total_batches,
            total_weighted_samples=total_weighted_samples,
            avg_loss=total_loss / max(total_samples, 1),
            total_loss=total_loss,
            training_time_seconds=training_time,
            metrics={
                "samples_per_second": total_samples / max(training_time, 0.001),
                "batches_per_second": total_batches / max(training_time, 0.001),
            }
        )
    
    def _train_batch(self, batch: Batch) -> Dict[str, Any]:
        """
        Train on a single batch.
        
        Args:
            batch: Training batch
            
        Returns:
            Dict with batch training results
        """
        import tensorflow as tf
        
        # 1. Pull model weights from PS
        model_weights = self._ps_client.pull_model()
        
        # 2. Load weights into local TF model
        self._model_wrapper.set_weights(model_weights)
        
        # 3. Extract sparse feature tokens and pull embeddings
        embeddings_per_feature = {}
        token_to_idx_per_feature = {}
        
        for feature_name, token_ids in batch.sparse_features.items():
            # Get unique tokens for this feature
            if isinstance(token_ids[0], np.ndarray):
                # Variable length sequences
                unique_tokens = list(set(
                    t for seq in token_ids for t in seq
                ))
            else:
                # Single token per sample
                unique_tokens = list(set(token_ids.tolist()))
            
            # Pull embeddings from PS
            embeddings = self._ps_client.pull_embeddings(unique_tokens)
            
            # Create mapping from token to embedding index
            token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
            
            embeddings_per_feature[feature_name] = embeddings
            token_to_idx_per_feature[feature_name] = token_to_idx
        
        # 4. Build batch input by looking up embeddings
        batch_embeddings = self._lookup_embeddings(
            batch,
            embeddings_per_feature,
            token_to_idx_per_feature
        )
        
        # 5. Get dense features
        dense_matrix = batch.get_dense_matrix()
        
        # 6. Concatenate embeddings and dense features
        if batch_embeddings.size > 0 and dense_matrix.size > 0:
            model_input = np.concatenate([batch_embeddings, dense_matrix], axis=1)
        elif batch_embeddings.size > 0:
            model_input = batch_embeddings
        else:
            model_input = dense_matrix
        
        # 7. Forward/backward pass and compute gradients
        loss, model_grads = self._model_wrapper.compute_gradients(
            inputs=model_input,
            targets=batch.targets,
            loss_fn=self._loss_fn,
            sample_weights=batch.weights,
        )
        
        # 8. Compute embedding gradients (backprop through embedding lookup)
        embedding_grads = self._compute_embedding_gradients(
            batch,
            embeddings_per_feature,
            token_to_idx_per_feature,
            model_input,
            batch.targets,
            batch.weights,
        )
        
        # 9. Push gradients to PS (include loss for real-time tracking)
        self._ps_client.push_gradients(
            weight_gradients=model_grads,
            embedding_gradients=embedding_grads,
            batch_loss=loss,
            batch_size=batch.batch_size,
        )
        
        # Calculate weighted samples
        if batch.weights is not None:
            weighted_samples = float(batch.weights.sum())
        else:
            weighted_samples = float(batch.batch_size)
        
        return {
            "loss": loss,
            "batch_size": batch.batch_size,
            "weighted_samples": weighted_samples,
        }
    
    def _lookup_embeddings(
        self,
        batch: Batch,
        embeddings_per_feature: Dict[str, np.ndarray],
        token_to_idx_per_feature: Dict[str, Dict[int, int]]
    ) -> np.ndarray:
        """
        Look up embeddings for each sample in the batch.
        
        For each sparse feature, looks up the embedding and returns
        the concatenated embeddings for each sample.
        """
        if not embeddings_per_feature:
            return np.zeros((batch.batch_size, 0), dtype=np.float32)
        
        feature_embeddings = []
        
        for feature_name in sorted(batch.sparse_features.keys()):
            token_ids = batch.sparse_features[feature_name]
            embeddings = embeddings_per_feature[feature_name]
            token_to_idx = token_to_idx_per_feature[feature_name]
            
            emb_dim = self.config.feature_config.get_embedding_dim(feature_name)
            batch_emb = np.zeros((batch.batch_size, emb_dim), dtype=np.float32)
            
            for i in range(batch.batch_size):
                sample_tokens = token_ids[i]
                
                if isinstance(sample_tokens, np.ndarray):
                    # Variable length sequence - average embeddings
                    if len(sample_tokens) > 0:
                        indices = [token_to_idx.get(int(t), 0) for t in sample_tokens]
                        batch_emb[i] = embeddings[indices].mean(axis=0)
                else:
                    # Single token
                    idx = token_to_idx.get(int(sample_tokens), 0)
                    batch_emb[i] = embeddings[idx]
            
            feature_embeddings.append(batch_emb)
        
        return np.concatenate(feature_embeddings, axis=1)
    
    def _compute_embedding_gradients(
        self,
        batch: Batch,
        embeddings_per_feature: Dict[str, np.ndarray],
        token_to_idx_per_feature: Dict[str, Dict[int, int]],
        model_input: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Compute gradients for embeddings.
        
        Uses automatic differentiation through the embedding lookup.
        """
        import tensorflow as tf
        
        # For simplicity, we compute gradients by treating embeddings
        # as part of the input and using TF gradient tape
        
        # Create embedding variables
        all_token_grads = {}
        
        # Build input with embedding as tf.Variable
        embedding_vars = {}
        emb_start_idx = 0
        
        for feature_name in sorted(batch.sparse_features.keys()):
            emb_dim = self.config.feature_config.get_embedding_dim(feature_name)
            embeddings = embeddings_per_feature[feature_name]
            
            embedding_vars[feature_name] = tf.Variable(
                embeddings, trainable=True, dtype=tf.float32
            )
            emb_start_idx += emb_dim
        
        # Forward pass with tape watching embeddings
        with tf.GradientTape() as tape:
            # Reconstruct input with embeddings
            feature_embs = []
            for feature_name in sorted(batch.sparse_features.keys()):
                token_ids = batch.sparse_features[feature_name]
                emb_var = embedding_vars[feature_name]
                token_to_idx = token_to_idx_per_feature[feature_name]
                
                emb_dim = self.config.feature_config.get_embedding_dim(feature_name)
                batch_size = batch.batch_size
                
                # Build indices for gather
                indices = []
                for i in range(batch_size):
                    sample_tokens = token_ids[i]
                    if isinstance(sample_tokens, np.ndarray):
                        if len(sample_tokens) > 0:
                            # Take first token for simplicity
                            # TODO: Handle variable length properly
                            idx = token_to_idx.get(int(sample_tokens[0]), 0)
                        else:
                            idx = 0
                    else:
                        idx = token_to_idx.get(int(sample_tokens), 0)
                    indices.append(idx)
                
                gathered = tf.gather(emb_var, indices)
                feature_embs.append(gathered)
            
            # Concatenate all feature embeddings
            if feature_embs:
                emb_concat = tf.concat(feature_embs, axis=1)
            else:
                emb_concat = tf.zeros((batch.batch_size, 0), dtype=tf.float32)
            
            # Add dense features
            dense_tensor = tf.convert_to_tensor(
                batch.get_dense_matrix(), dtype=tf.float32
            )
            
            if emb_concat.shape[1] > 0 and dense_tensor.shape[1] > 0:
                input_tensor = tf.concat([emb_concat, dense_tensor], axis=1)
            elif emb_concat.shape[1] > 0:
                input_tensor = emb_concat
            else:
                input_tensor = dense_tensor
            
            # Forward pass
            model = self._model_wrapper.get_model()
            predictions = model(input_tensor, training=True)
            
            # Compute loss
            targets_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
            per_sample_loss = self._loss_fn(targets_tensor, predictions)
            
            if sample_weights is not None:
                weights_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
                loss = tf.reduce_mean(per_sample_loss * weights_tensor)
            else:
                loss = tf.reduce_mean(per_sample_loss)
        
        # Compute gradients for embeddings
        emb_grads = tape.gradient(loss, list(embedding_vars.values()))
        
        # Map gradients back to token IDs
        for (feature_name, emb_var), grad in zip(
            sorted(embedding_vars.items()),
            emb_grads
        ):
            if grad is None:
                continue
            
            # Handle IndexedSlices (sparse gradients) from tf.gather
            if isinstance(grad, tf.IndexedSlices):
                # Convert to dense gradient
                grad_np = tf.convert_to_tensor(grad).numpy()
            else:
                grad_np = grad.numpy()
            
            token_to_idx = token_to_idx_per_feature[feature_name]
            idx_to_token = {v: k for k, v in token_to_idx.items()}
            
            for idx, token_id in idx_to_token.items():
                if idx < len(grad_np):
                    all_token_grads[token_id] = grad_np[idx]
        
        return all_token_grads
    
    def close(self):
        """Release resources."""
        if hasattr(self, '_ps_client') and self._ps_client is not None:
            self._ps_client.close()


def train_partition_function(
    partition_id: int,
    partition_paths: List[str],
    config_dict: Dict[str, Any],
    server_info_list: List[Dict[str, Any]],
    model_builder_serialized: bytes
) -> WorkerTrainingResult:
    """
    Training function to be executed on each Spark partition.
    
    This function is designed to be used with Spark's mapPartitions.
    
    Args:
        partition_id: Unique partition/worker ID
        partition_paths: List of data file paths for this partition
        config_dict: TrainerConfig as dictionary
        server_info_list: List of ServerInfo as dictionaries
        model_builder_serialized: Serialized model builder function
        
    Returns:
        WorkerTrainingResult
    """
    import cloudpickle
    
    from distributed_trainer.config import TrainerConfig
    from pyspark_ps.communication.protocol import ServerInfo
    
    # Deserialize config
    config = TrainerConfig.from_dict(config_dict)
    
    # Deserialize server info
    server_info = [ServerInfo.from_dict(s) for s in server_info_list]
    
    # Deserialize model builder
    model_builder = cloudpickle.loads(model_builder_serialized)
    
    # Create worker and train
    worker = WorkerTrainer(
        worker_id=partition_id,
        config=config,
        server_info=server_info,
        model_builder=model_builder,
    )
    
    try:
        result = worker.train_partitions(partition_paths)
    finally:
        worker.close()
    
    return result

