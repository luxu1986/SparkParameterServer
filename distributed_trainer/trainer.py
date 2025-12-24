"""Distributed Trainer orchestrator for driver node."""

import json
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from distributed_trainer.config import TrainerConfig
from distributed_trainer.thread_config import configure_threads, configure_tensorflow_threads
from distributed_trainer.tf_model_wrapper import TFModelWrapper
from distributed_trainer.data_loader import (
    S3ParquetDataLoader,
    S3Config,
    format_date_path,
    parse_date_range,
)
from distributed_trainer.worker import WorkerTrainer, WorkerTrainingResult


class LossTracker:
    """
    Thread-safe loss tracker for moving average calculation.
    
    Workers report batch losses, and driver queries the moving average.
    Uses a simple exponential moving average (EMA).
    """
    
    def __init__(self, ema_alpha: float = 0.01):
        """
        Initialize loss tracker.
        
        Args:
            ema_alpha: Smoothing factor for EMA (smaller = smoother)
        """
        self._lock = threading.Lock()
        self._ema_alpha = ema_alpha
        
        self._ema_loss: Optional[float] = None
        self._total_samples = 0
        self._total_loss = 0.0
        self._batch_count = 0
        self._last_losses: List[float] = []  # Recent losses for display
        self._max_recent = 100
    
    def report_loss(self, loss: float, batch_size: int = 1):
        """Report a batch loss from a worker."""
        with self._lock:
            # Update EMA
            if self._ema_loss is None:
                self._ema_loss = loss
            else:
                self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss
            
            # Update totals
            self._total_samples += batch_size
            self._total_loss += loss * batch_size
            self._batch_count += 1
            
            # Track recent losses
            self._last_losses.append(loss)
            if len(self._last_losses) > self._max_recent:
                self._last_losses.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current loss statistics."""
        with self._lock:
            avg_loss = self._total_loss / max(self._total_samples, 1)
            recent_avg = sum(self._last_losses) / max(len(self._last_losses), 1) if self._last_losses else 0.0
            
            return {
                "ema_loss": self._ema_loss,
                "avg_loss": avg_loss,
                "recent_avg_loss": recent_avg,
                "total_samples": self._total_samples,
                "batch_count": self._batch_count,
            }
    
    def reset(self):
        """Reset all stats."""
        with self._lock:
            self._ema_loss = None
            self._total_samples = 0
            self._total_loss = 0.0
            self._batch_count = 0
            self._last_losses.clear()


# Global loss tracker instance (accessible from workers via broadcast)
_global_loss_tracker: Optional[LossTracker] = None


def get_loss_tracker() -> Optional[LossTracker]:
    """Get the global loss tracker."""
    return _global_loss_tracker


def set_loss_tracker(tracker: LossTracker):
    """Set the global loss tracker."""
    global _global_loss_tracker
    _global_loss_tracker = tracker


class ProgressMonitor:
    """
    Background thread that monitors PS server stats and prints progress.
    
    Runs on the driver node to provide visibility into training progress
    while workers are running on executors.
    """
    
    def __init__(
        self,
        ps_client: Any,
        interval_seconds: float = 30.0,
        update_threshold: int = 100000,
        loss_tracker: Optional[LossTracker] = None,
    ):
        """
        Initialize progress monitor.
        
        Args:
            ps_client: PSMainClient to query stats from
            interval_seconds: How often to check stats
            update_threshold: Print summary after this many updates
            loss_tracker: Optional loss tracker for moving average
        """
        self._ps_client = ps_client
        self._interval = interval_seconds
        self._update_threshold = update_threshold
        self._loss_tracker = loss_tracker
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Stats tracking
        self._start_time = time.time()
        self._last_print_time = time.time()
        self._last_update_count = 0
        self._last_embedding_count = 0
        self._total_updates_printed = 0
    
    def start(self):
        """Start the background monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self._last_print_time = time.time()
        self._last_update_count = 0
        self._last_embedding_count = 0
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_and_print_stats()
            except Exception as e:
                # Don't crash on stats errors
                pass
            
            time.sleep(self._interval)
    
    def _check_and_print_stats(self):
        """Query PS stats and print progress if threshold reached."""
        try:
            stats = self._ps_client.get_cluster_stats()
        except Exception:
            return
        
        current_time = time.time()
        elapsed = current_time - self._start_time
        interval_elapsed = current_time - self._last_print_time
        
        # Aggregate stats from all servers
        total_embeddings = stats.get("total_embeddings", 0)
        total_memory = stats.get("total_memory_bytes", 0)
        
        # Count total updates from server stats
        total_updates = 0
        for server_stats in stats.get("servers", []):
            emb_stats = server_stats.get("embedding_stats", {})
            total_updates += emb_stats.get("total_updates", 0)
        
        # Check if we crossed an update threshold
        updates_since_last = total_updates - self._last_update_count
        
        if updates_since_last >= self._update_threshold or interval_elapsed >= 60:
            # Calculate rates
            updates_per_sec = updates_since_last / max(interval_elapsed, 0.001)
            new_embeddings = total_embeddings - self._last_embedding_count
            
            # Format memory
            if total_memory > 1024 * 1024 * 1024:
                memory_str = f"{total_memory / (1024**3):.2f} GB"
            elif total_memory > 1024 * 1024:
                memory_str = f"{total_memory / (1024**2):.1f} MB"
            else:
                memory_str = f"{total_memory / 1024:.0f} KB"
            
            # Get loss info from PS servers (real-time from gradient pushes)
            loss_str = ""
            total_loss_samples = 0
            weighted_loss_sum = 0.0
            ema_losses = []
            
            for server_stats in stats.get("servers", []):
                loss_stats = server_stats.get("loss_stats", {})
                ema = loss_stats.get("ema_loss")
                samples = loss_stats.get("total_samples", 0)
                if ema is not None and samples > 0:
                    ema_losses.append(ema)
                    total_loss_samples += samples
                    weighted_loss_sum += loss_stats.get("avg_loss", 0) * samples
            
            if ema_losses:
                # Average EMA across servers
                avg_ema = sum(ema_losses) / len(ema_losses)
                loss_str = f" | loss={avg_ema:.6f}"
            
            # Print progress
            print(
                f"[Progress] elapsed={elapsed:.0f}s | "
                f"updates={total_updates:,} (+{updates_since_last:,}) | "
                f"rate={updates_per_sec:,.0f}/s | "
                f"embeddings={total_embeddings:,} (+{new_embeddings:,}) | "
                f"memory={memory_str}{loss_str}"
            )
            
            # Update tracking
            self._last_print_time = current_time
            self._last_update_count = total_updates
            self._last_embedding_count = total_embeddings
            self._total_updates_printed = total_updates
    
    def print_final_summary(self):
        """Print final summary when training completes."""
        try:
            stats = self._ps_client.get_cluster_stats()
            elapsed = time.time() - self._start_time
            
            total_embeddings = stats.get("total_embeddings", 0)
            total_memory = stats.get("total_memory_bytes", 0)
            
            total_updates = 0
            for server_stats in stats.get("servers", []):
                emb_stats = server_stats.get("embedding_stats", {})
                total_updates += emb_stats.get("total_updates", 0)
            
            # Format memory
            if total_memory > 1024 * 1024 * 1024:
                memory_str = f"{total_memory / (1024**3):.2f} GB"
            elif total_memory > 1024 * 1024:
                memory_str = f"{total_memory / (1024**2):.1f} MB"
            else:
                memory_str = f"{total_memory / 1024:.0f} KB"
            
            # Get loss stats from PS servers
            loss_str = "N/A"
            ema_losses = []
            total_batches = 0
            weighted_loss_sum = 0.0
            total_loss_samples = 0
            
            for server_stats in stats.get("servers", []):
                loss_stats = server_stats.get("loss_stats", {})
                ema = loss_stats.get("ema_loss")
                samples = loss_stats.get("total_samples", 0)
                batches = loss_stats.get("batch_count", 0)
                if ema is not None and samples > 0:
                    ema_losses.append(ema)
                    total_loss_samples += samples
                    total_batches += batches
                    weighted_loss_sum += loss_stats.get("avg_loss", 0) * samples
            
            if ema_losses:
                avg_ema = sum(ema_losses) / len(ema_losses)
                avg_loss = weighted_loss_sum / max(total_loss_samples, 1)
                loss_str = f"{avg_ema:.6f} (EMA), {avg_loss:.6f} (avg over {total_batches:,} batches)"
            
            print("\n" + "=" * 70)
            print("Training Progress Summary")
            print("=" * 70)
            print(f"  Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"  Total embedding updates: {total_updates:,}")
            print(f"  Unique embeddings: {total_embeddings:,}")
            print(f"  Memory usage: {memory_str}")
            print(f"  Average update rate: {total_updates / max(elapsed, 1):,.0f}/s")
            print(f"  Loss: {loss_str}")
            print("=" * 70)
            
        except Exception as e:
            print(f"[Warning] Could not get final stats: {e}")


@dataclass
class DayTrainingResult:
    """Result from training on a single day's data."""
    
    date: str
    total_samples: int
    total_batches: int
    avg_loss: float
    worker_results: List[Dict[str, Any]]
    training_time_seconds: float
    checkpoint_path: Optional[str] = None


class DistributedTrainer:
    """
    Distributed Trainer orchestrator running on Spark DRIVER node.
    
    Responsibilities:
    - Configure MKL/OpenMP threading
    - Manage PS client lifecycle (creates PSMainClient internally)
    - Initialize PS weight_store with model weights
    - Broadcast TF model structure (architecture only, not weights)
    - Orchestrate day-by-day training
    - Distribute data partitions to workers
    - Save checkpoints to S3
    - Trigger embedding decay
    """
    
    def __init__(self, spark_context: Any, config: TrainerConfig):
        """
        Initialize DistributedTrainer on driver node.
        
        Steps:
        1. Configure threads (MKL, OpenMP, TF)
        2. Create PSMainClient (internal, not exposed)
        3. Start PS servers
        4. Initialize data loader
        
        Args:
            spark_context: Spark context (can be None for local testing)
            config: Training configuration
        """
        # Validate config
        config.validate()
        
        self.spark_context = spark_context
        self.config = config
        
        # Configure threading (before TF import)
        configure_threads(config.thread_config)
        
        # Import and configure TensorFlow
        configure_tensorflow_threads(config.thread_config)
        
        # Create PS main client
        from pyspark_ps import PSMainClient, PSConfig
        
        ps_config = config.ps_config
        if isinstance(ps_config, dict):
            ps_config = PSConfig(**ps_config)
        
        # Update PS config with trainer settings
        ps_config.embedding_optimizer = config.embedding_optimizer
        ps_config.weight_optimizer = config.model_optimizer
        
        self._ps_client = PSMainClient(spark_context, ps_config)
        self._server_info = self._ps_client.start_servers()
        
        # Data loader
        self._data_loader = S3ParquetDataLoader(S3Config())
        
        # Model management
        self._model_wrapper: Optional[TFModelWrapper] = None
        self._model_builder: Optional[Callable[[], Any]] = None
        self._model_initialized = False
        
        # Training state
        self._total_steps = 0
        self._training_history: List[DayTrainingResult] = []
        
        # Progress monitoring
        self._progress_monitor: Optional['ProgressMonitor'] = None
        self._last_stats_snapshot: Dict[str, Any] = {}
        
        if config.verbose:
            print(f"DistributedTrainer initialized with {len(self._server_info)} PS servers")
    
    # Model Management
    
    def set_model(self, model: Any):
        """
        Set the TensorFlow 2.0 model to train.
        
        Steps:
        1. Store model structure via TFModelWrapper
        2. Extract initial weights as Dict[str, np.ndarray]
        3. Initialize PS weight_store with these weights
        
        Args:
            model: tf.keras.Model instance
        """
        self._model_wrapper = TFModelWrapper(model=model)
        
        # Create model builder from model (for serialization)
        model_json = model.to_json()
        
        def model_builder():
            import tensorflow as tf
            return tf.keras.models.model_from_json(model_json)
        
        self._model_builder = model_builder
        
        # Initialize PS weight_store with model weights
        self._initialize_weights()
    
    def set_model_builder(self, builder_fn: Callable[[], Any]):
        """
        Set a function that builds the model.
        Preferred for distributed execution (avoids serialization issues).
        
        Steps:
        1. Build model using builder_fn
        2. Store builder_fn for broadcasting to workers
        3. Extract initial weights and init PS weight_store
        
        Args:
            builder_fn: Function that returns a tf.keras.Model
        """
        self._model_builder = builder_fn
        self._model_wrapper = TFModelWrapper(model_builder=builder_fn)
        self._model_wrapper.build_model()
        
        # Initialize PS weight_store with model weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize PS weight_store with current model weights."""
        if self._model_wrapper is None:
            raise ValueError("No model set. Call set_model() or set_model_builder() first.")
        
        # Get weight shapes for initialization
        weight_shapes = self._model_wrapper.get_weight_shapes()
        
        # Initialize weights in PS
        self._ps_client.init_weights(
            weight_shapes=weight_shapes,
            init_strategy="normal",
            init_scale=0.01,
        )
        
        # Get initial weights and push to PS (to use model's initialization)
        initial_weights = self._model_wrapper.get_weights()
        
        # Push initial weights as "gradients" with learning rate = -1
        # This is a workaround - ideally PS would have a set_weights method
        # For now, we rely on the PS initialization
        
        self._model_initialized = True
        
        if self.config.verbose:
            total_params = sum(
                int(shape[0] * shape[1]) if len(shape) == 2 else int(shape[0])
                for shape in weight_shapes.values()
            )
            print(f"Initialized {len(weight_shapes)} weight tensors ({total_params:,} parameters)")
    
    def get_model(self) -> Any:
        """
        Get current model with latest weights from PS.
        
        Steps:
        1. Create a worker client temporarily
        2. Pull latest weights from PS weight_store
        3. Load weights into local model
        4. Return model
        
        Returns:
            tf.keras.Model with current weights
        """
        if self._model_wrapper is None:
            raise ValueError("No model set")
        
        from pyspark_ps import PSWorkerClient
        
        # Create temporary worker client to pull weights
        temp_client = PSWorkerClient(
            server_info=self._server_info,
            config=self.config.ps_config,
            client_id="driver_temp"
        )
        
        try:
            weights = temp_client.pull_model()
            self._model_wrapper.set_weights(weights)
        finally:
            temp_client.close()
        
        return self._model_wrapper.get_model()
    
    # Training
    
    def train_date_range(
        self,
        start_date: Union[str, date],
        end_date: Union[str, date],
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train on data from start_date to end_date (inclusive).
        
        For each date:
        1. Construct S3 path from template
        2. Discover parquet partitions
        3. Distribute partitions to workers (size-balanced)
        4. Broadcast model_builder to workers (structure only)
        5. Execute distributed training (workers pull weights from PS)
        6. Save checkpoint to S3 (if configured)
        7. Call decay_embeddings via PSMainClient (if configured)
        
        Args:
            start_date: Start date (string or date object)
            end_date: End date (inclusive)
            resume_from_checkpoint: Optional S3 path to resume from
            
        Returns:
            Training summary with per-day metrics
        """
        if self._model_builder is None:
            raise ValueError("No model set. Call set_model() or set_model_builder() first.")
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Parse dates
        if isinstance(start_date, date):
            start_date = start_date.strftime(self.config.date_format)
        if isinstance(end_date, date):
            end_date = end_date.strftime(self.config.date_format)
        
        dates = parse_date_range(start_date, end_date, self.config.date_format)
        
        # Start progress monitor with loss tracking
        self._loss_tracker = LossTracker(ema_alpha=0.01)
        set_loss_tracker(self._loss_tracker)
        
        if self.config.verbose:
            self._progress_monitor = ProgressMonitor(
                ps_client=self._ps_client,
                interval_seconds=30.0,
                update_threshold=self.config.progress_update_threshold,
                loss_tracker=self._loss_tracker,
            )
            self._progress_monitor.start()
            print(f"[Progress Monitor] Started - will report every 100k updates or 60s")
        
        if self.config.verbose:
            print(f"\nTraining on {len(dates)} days: {start_date} to {end_date}")
        
        start_time = time.time()
        daily_results = []
        
        for training_date in dates:
            day_result = self.train_single_day(training_date)
            daily_results.append(day_result)
            
            # Checkpoint after each day
            if self.config.checkpoint_after_each_day and self.config.s3_checkpoint_path:
                checkpoint_path = f"{self.config.s3_checkpoint_path}/dt={training_date}"
                self.save_checkpoint(checkpoint_path)
                day_result.checkpoint_path = checkpoint_path
            
            # Decay embeddings after each day
            if self.config.decay_after_each_day:
                self._ps_client.decay_embeddings(
                    method=self.config.decay_method,
                    factor=self.config.decay_factor,
                    min_count=self.config.prune_threshold,
                )
                
                if self.config.verbose:
                    print(f"  Applied decay (factor={self.config.decay_factor})")
            
            self._training_history.append(day_result)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        total_samples = sum(r.total_samples for r in daily_results)
        total_batches = sum(r.total_batches for r in daily_results)
        avg_loss = sum(r.avg_loss * r.total_samples for r in daily_results) / max(total_samples, 1)
        
        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "num_days": len(dates),
            "total_samples": total_samples,
            "total_batches": total_batches,
            "avg_loss": avg_loss,
            "total_time_seconds": total_time,
            "daily_results": [
                {
                    "date": r.date,
                    "samples": r.total_samples,
                    "batches": r.total_batches,
                    "avg_loss": r.avg_loss,
                    "time_seconds": r.training_time_seconds,
                }
                for r in daily_results
            ]
        }
        
        # Stop progress monitor and print final summary
        if self._progress_monitor:
            self._progress_monitor.stop()
            self._progress_monitor.print_final_summary()
            self._progress_monitor = None
        
        if self.config.verbose:
            print(f"\nTraining complete:")
            print(f"  Total samples: {total_samples:,}")
            print(f"  Average loss: {avg_loss:.6f}")
            print(f"  Total time: {total_time:.1f}s")
        
        return summary
    
    def train_single_day(self, training_date: Union[str, date]) -> DayTrainingResult:
        """
        Train on a single day's data.
        
        Args:
            training_date: Date to train on
            
        Returns:
            DayTrainingResult with training metrics
        """
        if isinstance(training_date, date):
            training_date = training_date.strftime(self.config.date_format)
        
        if self.config.verbose:
            print(f"\nTraining day: {training_date}")
        
        start_time = time.time()
        
        # Construct S3 path
        s3_path = format_date_path(
            self.config.s3_data_path_template,
            training_date,
            self.config.date_format
        )
        
        # Discover partitions
        partitions = self._data_loader.discover_partitions(s3_path)
        
        if not partitions:
            if self.config.verbose:
                print(f"  No data found at {s3_path}")
            return DayTrainingResult(
                date=training_date,
                total_samples=0,
                total_batches=0,
                avg_loss=0.0,
                worker_results=[],
                training_time_seconds=0.0,
            )
        
        if self.config.verbose:
            print(f"  Found {len(partitions)} partitions")
        
        # Distribute partitions to workers
        worker_partitions = self._data_loader.distribute_partitions(
            partitions,
            self.config.num_workers,
            strategy="size_balanced"
        )
        
        # Execute distributed training
        worker_results = self._execute_distributed_training(worker_partitions)
        
        # Aggregate results
        total_samples = sum(r.total_samples for r in worker_results)
        total_batches = sum(r.total_batches for r in worker_results)
        
        if total_samples > 0:
            avg_loss = sum(r.total_loss for r in worker_results) / total_samples
        else:
            avg_loss = 0.0
        
        training_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"  Samples: {total_samples:,}, Loss: {avg_loss:.6f}, Time: {training_time:.1f}s")
        
        self._total_steps += total_batches
        
        return DayTrainingResult(
            date=training_date,
            total_samples=total_samples,
            total_batches=total_batches,
            avg_loss=avg_loss,
            worker_results=[r.to_dict() for r in worker_results],
            training_time_seconds=training_time,
        )
    
    def _execute_distributed_training(
        self,
        worker_partitions: Dict[int, List[str]]
    ) -> List[WorkerTrainingResult]:
        """
        Execute distributed training across workers.
        
        Uses Spark if available, otherwise runs locally with threads.
        
        Args:
            worker_partitions: Dict mapping worker_id -> list of partition paths
            
        Returns:
            List of WorkerTrainingResult from all workers
        """
        is_local = self._is_local_mode()
        
        if self.config.verbose:
            if self.spark_context is None:
                print(f"  [Execution] No SparkContext - using local threads")
            else:
                master = getattr(self.spark_context, 'master', 'unknown')
                print(f"  [Execution] SparkContext master: {master}")
                print(f"  [Execution] Mode: {'LOCAL' if is_local else 'DISTRIBUTED'}")
        
        if self.spark_context is not None and not is_local:
            return self._execute_spark_training(worker_partitions)
        else:
            return self._execute_local_training(worker_partitions)
    
    def _is_local_mode(self) -> bool:
        """Check if Spark is in local mode."""
        if self.spark_context is None:
            return True
        
        try:
            master = self.spark_context.master
            return master.startswith("local")
        except Exception:
            return True
    
    def _execute_spark_training(
        self,
        worker_partitions: Dict[int, List[str]]
    ) -> List[WorkerTrainingResult]:
        """Execute training using Spark."""
        import cloudpickle
        
        if self.config.verbose:
            print(f"  [Spark] Preparing {len(worker_partitions)} worker tasks...")
            total_partitions = sum(len(p) for p in worker_partitions.values())
            print(f"  [Spark] Total data partitions: {total_partitions}")
            print(f"  [Spark] PS servers: {[(s.host, s.port) for s in self._server_info]}")
        
        # Serialize model builder
        model_builder_serialized = cloudpickle.dumps(self._model_builder)
        
        # Prepare broadcast variables
        config_dict = self.config.to_dict()
        server_info_list = [s.to_dict() for s in self._server_info]
        
        if self.config.verbose:
            print(f"  [Spark] Broadcasting config and model...")
        
        config_bc = self.spark_context.broadcast(config_dict)
        server_info_bc = self.spark_context.broadcast(server_info_list)
        model_builder_bc = self.spark_context.broadcast(model_builder_serialized)
        
        # Create RDD with worker assignments
        worker_data = list(worker_partitions.items())
        rdd = self.spark_context.parallelize(worker_data, len(worker_data))
        
        if self.config.verbose:
            print(f"  [Spark] Submitting {len(worker_data)} tasks to executors...")
            print(f"  [Spark] Waiting for tasks to complete (this may take a while)...")
        
        # Train on each partition
        def train_worker(item):
            worker_id, partition_paths = item
            
            # Log on executor (will appear in Spark executor logs)
            import sys
            print(f"[Worker {worker_id}] Starting with {len(partition_paths)} partitions", file=sys.stderr)
            
            from distributed_trainer.worker import train_partition_function
            
            try:
                result = train_partition_function(
                    partition_id=worker_id,
                    partition_paths=partition_paths,
                    config_dict=config_bc.value,
                    server_info_list=server_info_bc.value,
                    model_builder_serialized=model_builder_bc.value,
                )
                print(f"[Worker {worker_id}] Completed: {result.total_samples} samples, loss={result.avg_loss:.6f}", file=sys.stderr)
                return result
            except Exception as e:
                print(f"[Worker {worker_id}] ERROR: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise
        
        start_time = time.time()
        results = rdd.map(train_worker).collect()
        elapsed = time.time() - start_time
        
        # Update loss tracker with worker results
        if hasattr(self, '_loss_tracker') and self._loss_tracker:
            for result in results:
                if result.total_samples > 0:
                    self._loss_tracker.report_loss(result.avg_loss, result.total_samples)
        
        if self.config.verbose:
            print(f"  [Spark] All {len(results)} tasks completed in {elapsed:.1f}s")
        
        return results
    
    def _execute_local_training(
        self,
        worker_partitions: Dict[int, List[str]]
    ) -> List[WorkerTrainingResult]:
        """Execute training locally with threads."""
        import threading
        
        results = []
        results_lock = threading.Lock()
        
        def train_worker(worker_id: int, partition_paths: List[str]):
            worker = WorkerTrainer(
                worker_id=worker_id,
                config=self.config,
                server_info=self._server_info,
                model_builder=self._model_builder,
            )
            
            try:
                result = worker.train_partitions(partition_paths)
                with results_lock:
                    results.append(result)
            finally:
                worker.close()
        
        threads = []
        for worker_id, partition_paths in worker_partitions.items():
            if partition_paths:  # Skip workers with no data
                t = threading.Thread(
                    target=train_worker,
                    args=(worker_id, partition_paths)
                )
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join()
        
        # Update loss tracker with worker results
        if hasattr(self, '_loss_tracker') and self._loss_tracker:
            for result in results:
                if result.total_samples > 0:
                    self._loss_tracker.report_loss(result.avg_loss, result.total_samples)
        
        return results
    
    # Checkpointing
    
    def save_checkpoint(self, s3_path: str):
        """
        Save checkpoint to S3.
        
        Saves:
        - PS embeddings (via PSMainClient.save_to_s3)
        - PS model weights (via PSMainClient.save_to_s3)
        - Model structure/architecture
        - Training config
        
        Args:
            s3_path: S3 path for checkpoint
        """
        if self.config.verbose:
            print(f"  Saving checkpoint to {s3_path}")
        
        # Save PS state (embeddings + weights)
        self._ps_client.save_to_s3(
            s3_path=s3_path,
            save_model=True,
            save_embeddings=True,
            save_optimizer_states=True,
        )
        
        # Save model architecture
        if self._model_wrapper is not None:
            model = self._model_wrapper.get_model()
            model_json = model.to_json()
            
            # Save via PS S3 backend
            from pyspark_ps.storage.s3_backend import S3Backend
            
            s3_backend = S3Backend(self.config.ps_config)
            s3_backend.upload(
                f"{s3_path}/model_architecture.json",
                model_json.encode(),
                content_type="application/json"
            )
            
            # Save training config
            config_json = self.config.to_json()
            s3_backend.upload(
                f"{s3_path}/trainer_config.json",
                config_json.encode(),
                content_type="application/json"
            )
    
    def load_checkpoint(self, s3_path: str):
        """
        Load and restore all state from S3.
        
        Args:
            s3_path: S3 path to checkpoint
        """
        if self.config.verbose:
            print(f"Loading checkpoint from {s3_path}")
        
        # Load PS state
        self._ps_client.load_from_s3(
            s3_path=s3_path,
            load_model=True,
            load_embeddings=True,
            load_optimizer_states=True,
        )
        
        # Load model architecture if not set
        if self._model_wrapper is None:
            from pyspark_ps.storage.s3_backend import S3Backend
            
            s3_backend = S3Backend(self.config.ps_config)
            
            try:
                model_json = s3_backend.download(
                    f"{s3_path}/model_architecture.json"
                ).decode()
                
                import tensorflow as tf
                model = tf.keras.models.model_from_json(model_json)
                self._model_wrapper = TFModelWrapper(model=model)
                
                # Create model builder
                def model_builder():
                    import tensorflow as tf
                    return tf.keras.models.model_from_json(model_json)
                
                self._model_builder = model_builder
                self._model_initialized = True
                
            except Exception as e:
                print(f"Warning: Could not load model architecture: {e}")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the PS cluster.
        
        Returns:
            Dict with cluster statistics
        """
        return self._ps_client.get_cluster_stats()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history.
        
        Returns:
            List of training results per day
        """
        return [
            {
                "date": r.date,
                "total_samples": r.total_samples,
                "total_batches": r.total_batches,
                "avg_loss": r.avg_loss,
                "training_time_seconds": r.training_time_seconds,
            }
            for r in self._training_history
        ]
    
    # Lifecycle
    
    def shutdown(self):
        """Shutdown PS servers and release resources."""
        if self.config.verbose:
            print("Shutting down DistributedTrainer...")
        
        self._ps_client.shutdown_servers()
        
        if self.config.verbose:
            print("Shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()

