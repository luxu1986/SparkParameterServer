"""
Distributed Trainer for PyTorch on Spark.
Encapsulates the training logic executed by Spark workers.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Iterator, Tuple
from pyspark.sql import Row
from pyspark import TaskContext

from core.ps_client import ParameterServerClient
from models.base_model import MainModel, CombinedModel
from utils.spark_utils import get_worker_hosts


logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Distributed trainer that coordinates training across Spark workers."""
    
    def __init__(self, config: Dict[str, Any], num_workers: int, 
                 driver_ps_host: str, driver_ps_port: int):
        """
        Initialize distributed trainer.
        
        Args:
            config: Training configuration
            num_workers: Number of worker nodes
            driver_ps_host: Driver parameter server hostname
            driver_ps_port: Driver parameter server port
        """
        self.config = config
        self.num_workers = num_workers
        self.driver_ps_host = driver_ps_host
        self.driver_ps_port = driver_ps_port
        
        # Training parameters
        self.batch_size = config["training"]["batch_size"]
        self.loss_fn_name = config["training"]["loss_function"]
        
        logger.info(f"Distributed trainer initialized for {num_workers} workers")
    
    def train_epoch(self, training_rdd, epoch: int):
        """
        Train for one epoch on the given RDD.
        
        Args:
            training_rdd: RDD containing training data
            epoch: Current epoch number
        """
        logger.info(f"Starting distributed training for epoch {epoch}")
        
        # Map training function over RDD partitions
        def train_partition(partition_iter: Iterator[Row]) -> Iterator[Dict[str, Any]]:
            """Train on a single partition."""
            return self._train_partition_with_ps(partition_iter, epoch)
        
        # Execute training on all partitions
        results = training_rdd.mapPartitions(train_partition).collect()
        
        # Aggregate and log results
        self._log_epoch_results(results, epoch)
    
    def _train_partition_with_ps(self, partition_iter: Iterator[Row], epoch: int) -> Iterator[Dict[str, Any]]:
        """
        Train on a single partition with parameter server communication.
        
        Args:
            partition_iter: Iterator over rows in this partition
            epoch: Current epoch number
            
        Yields:
            Training statistics for this partition
        """
        # Get task context for worker identification
        task_context = TaskContext.get()
        partition_id = task_context.partitionId() if task_context else 0
        
        logger.info(f"Worker {partition_id} starting training on partition for epoch {epoch}")
        
        # Get worker hosts for parameter server client
        worker_hosts = get_worker_hosts()
        
        # Initialize parameter server client
        ps_client = ParameterServerClient(
            driver_host=self.driver_ps_host,
            driver_port=self.driver_ps_port,
            worker_hosts=worker_hosts,
            worker_port_base=self.config["parameter_server"]["worker_port_base"],
            timeout=self.config["parameter_server"].get("timeout", 30.0)
        )
        
        try:
            # Initialize local model
            local_model = CombinedModel(
                main_model_config=self.config["model"],
                embedding_config=self.config["embedding"]
            )
            
            # Initialize loss function
            loss_fn = self._get_loss_function()
            
            # Training statistics
            total_loss = 0.0
            num_batches = 0
            num_samples = 0
            
            # Process data in batches
            batch_data = []
            for row in partition_iter:
                batch_data.append(row)
                
                if len(batch_data) >= self.batch_size:
                    # Process batch
                    batch_loss, batch_samples = self._process_batch(
                        batch_data, local_model, loss_fn, ps_client
                    )
                    
                    total_loss += batch_loss
                    num_samples += batch_samples
                    num_batches += 1
                    
                    # Clear batch
                    batch_data = []
            
            # Process remaining data
            if batch_data:
                batch_loss, batch_samples = self._process_batch(
                    batch_data, local_model, loss_fn, ps_client
                )
                total_loss += batch_loss
                num_samples += batch_samples
                num_batches += 1
            
            # Return training statistics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            yield {
                "partition_id": partition_id,
                "epoch": epoch,
                "num_batches": num_batches,
                "num_samples": num_samples,
                "avg_loss": avg_loss,
                "total_loss": total_loss
            }
            
        except Exception as e:
            logger.error(f"Error in partition {partition_id}: {str(e)}")
            raise
        finally:
            ps_client.close()
    
    def _process_batch(self, batch_data: List[Row], model: CombinedModel, 
                      loss_fn: nn.Module, ps_client: ParameterServerClient) -> Tuple[float, int]:
        """
        Process a single batch of data.
        
        Args:
            batch_data: List of data rows
            model: Local model instance
            loss_fn: Loss function
            ps_client: Parameter server client
            
        Returns:
            Tuple of (batch_loss, num_samples)
        """
        try:
            # Convert batch data to tensors
            features, targets, feature_ids = self._prepare_batch_tensors(batch_data)
            
            # Fetch current model parameters from driver
            model_params = ps_client.fetch_model_from_driver()
            model.load_main_model_state(model_params)
            
            # Fetch embeddings from workers
            embeddings = ps_client.fetch_embeddings_from_workers(feature_ids)
            
            # Forward pass
            model.eval()  # Set to eval mode for inference
            predictions = model.forward_with_embeddings(features, embeddings)
            
            # Compute loss
            loss = loss_fn(predictions, targets)
            
            # Backward pass
            model.train()  # Set to train mode for gradient computation
            loss.backward()
            
            # Extract gradients
            main_model_gradients = model.get_main_model_gradients()
            embedding_gradients = model.get_embedding_gradients(feature_ids)
            
            # Push gradients to parameter servers
            ps_client.push_gradients_to_driver(main_model_gradients)
            ps_client.push_embedding_gradients_to_workers(embedding_gradients)
            
            # Clear gradients
            model.zero_grad()
            
            return loss.item(), len(batch_data)
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def _prepare_batch_tensors(self, batch_data: List[Row]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Convert batch data to PyTorch tensors.
        
        Args:
            batch_data: List of Spark Row objects
            
        Returns:
            Tuple of (features, targets, feature_ids)
        """
        # Extract features, targets, and categorical feature IDs
        features_list = []
        targets_list = []
        feature_ids = set()
        
        for row in batch_data:
            # Assuming row has 'features', 'target', and 'categorical_features' columns
            row_dict = row.asDict()
            
            # Numerical features
            features = torch.tensor(row_dict.get('features', []), dtype=torch.float32)
            features_list.append(features)
            
            # Target
            target = torch.tensor(row_dict.get('target', 0.0), dtype=torch.float32)
            targets_list.append(target)
            
            # Categorical feature IDs
            cat_features = row_dict.get('categorical_features', [])
            feature_ids.update(cat_features)
        
        # Stack into batch tensors
        batch_features = torch.stack(features_list)
        batch_targets = torch.stack(targets_list)
        
        return batch_features, batch_targets, list(feature_ids)
    
    def _get_loss_function(self) -> nn.Module:
        """Get the loss function based on configuration."""
        loss_name = self.loss_fn_name.lower()
        
        if loss_name == "mse" or loss_name == "mean_squared_error":
            return nn.MSELoss()
        elif loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "binary_cross_entropy":
            return nn.BCEWithLogitsLoss()
        elif loss_name == "l1" or loss_name == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def _log_epoch_results(self, results: List[Dict[str, Any]], epoch: int):
        """
        Log aggregated results from all partitions.
        
        Args:
            results: List of results from all partitions
            epoch: Current epoch number
        """
        if not results:
            logger.warning(f"No results received for epoch {epoch}")
            return
        
        # Aggregate statistics
        total_batches = sum(r["num_batches"] for r in results)
        total_samples = sum(r["num_samples"] for r in results)
        total_loss = sum(r["total_loss"] for r in results)
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  Total partitions: {len(results)}")
        logger.info(f"  Total batches: {total_batches}")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Average loss: {avg_loss:.6f}")
        
        # Log per-partition statistics (debug level)
        for result in results:
            logger.debug(f"  Partition {result['partition_id']}: "
                        f"{result['num_batches']} batches, "
                        f"{result['num_samples']} samples, "
                        f"avg_loss={result['avg_loss']:.6f}")


class WorkerTrainer:
    """
    Simplified trainer for individual worker processes.
    This can be used for more fine-grained control over worker training.
    """
    
    def __init__(self, config: Dict[str, Any], worker_id: int, ps_client: ParameterServerClient):
        """
        Initialize worker trainer.
        
        Args:
            config: Training configuration
            worker_id: Worker identifier
            ps_client: Parameter server client
        """
        self.config = config
        self.worker_id = worker_id
        self.ps_client = ps_client
        
        # Initialize local model
        self.model = CombinedModel(
            main_model_config=config["model"],
            embedding_config=config["embedding"]
        )
        
        # Initialize loss function
        self.loss_fn = self._get_loss_function()
        
        logger.info(f"Worker trainer {worker_id} initialized")
    
    def train_on_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train on a single batch of data.
        
        Args:
            batch_data: List of data samples
            
        Returns:
            Training metrics for this batch
        """
        try:
            # Prepare batch
            features, targets, feature_ids = self._prepare_batch(batch_data)
            
            # Fetch parameters
            model_params = self.ps_client.fetch_model_from_driver()
            self.model.load_main_model_state(model_params)
            
            embeddings = self.ps_client.fetch_embeddings_from_workers(feature_ids)
            
            # Forward pass
            predictions = self.model.forward_with_embeddings(features, embeddings)
            loss = self.loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Push gradients
            main_gradients = self.model.get_main_model_gradients()
            embedding_gradients = self.model.get_embedding_gradients(feature_ids)
            
            self.ps_client.push_gradients_to_driver(main_gradients)
            self.ps_client.push_embedding_gradients_to_workers(embedding_gradients)
            
            # Clear gradients
            self.model.zero_grad()
            
            return {
                "loss": loss.item(),
                "num_samples": len(batch_data)
            }
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} batch training failed: {str(e)}")
            raise
    
    def _prepare_batch(self, batch_data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Prepare batch data for training."""
        features_list = []
        targets_list = []
        feature_ids = set()
        
        for sample in batch_data:
            features = torch.tensor(sample.get('features', []), dtype=torch.float32)
            target = torch.tensor(sample.get('target', 0.0), dtype=torch.float32)
            cat_features = sample.get('categorical_features', [])
            
            features_list.append(features)
            targets_list.append(target)
            feature_ids.update(cat_features)
        
        batch_features = torch.stack(features_list)
        batch_targets = torch.stack(targets_list)
        
        return batch_features, batch_targets, list(feature_ids)
    
    def _get_loss_function(self) -> nn.Module:
        """Get the loss function based on configuration."""
        loss_name = self.config["training"]["loss_function"].lower()
        
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "binary_cross_entropy":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}") 