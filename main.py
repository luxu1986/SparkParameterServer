#!/usr/bin/env python3
"""
Main entry point for distributed PyTorch training on Spark.
Orchestrates the training process with parameter servers.
"""

import argparse
import logging
import yaml
from typing import Dict, Any
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
import torch.multiprocessing as mp

from core.parameter_server import DriverParameterServer, WorkerParameterServer
from core.trainer import DistributedTrainer
from utils.spark_utils import get_worker_info, setup_logging
from models.base_model import MainModel, EmbeddingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed PyTorch Training on Spark")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training configuration file")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_spark_session(config: Dict[str, Any]) -> SparkSession:
    """Create and configure Spark session."""
    spark = SparkSession.builder \
        .appName("DistributedPyTorchTraining") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel(config.get("spark_log_level", "WARN"))
    return spark


def setup_parameter_servers(spark: SparkSession, config: Dict[str, Any]):
    """Initialize parameter servers on driver and workers."""
    # Start driver parameter server
    driver_ps = DriverParameterServer(
        model_config=config["model"],
        learning_rate=config["training"]["learning_rate"],
        port=config["parameter_server"]["driver_port"]
    )
    driver_ps.start()
    
    # Get worker information
    worker_info = get_worker_info(spark)
    num_workers = len(worker_info)
    
    # Broadcast worker setup function
    def setup_worker_ps(partition_id):
        """Setup worker parameter server on each worker node."""
        worker_ps = WorkerParameterServer(
            worker_id=partition_id,
            num_workers=num_workers,
            embedding_config=EmbeddingConfig(**config["embedding"]),
            learning_rate=config["training"]["learning_rate"],
            port=config["parameter_server"]["worker_port_base"] + partition_id
        )
        worker_ps.start()
        return [f"Worker PS {partition_id} started"]
    
    # Initialize worker parameter servers
    worker_rdd = spark.sparkContext.parallelize(range(num_workers), num_workers)
    worker_rdd.mapPartitionsWithIndex(lambda idx, _: setup_worker_ps(idx)).collect()
    
    return driver_ps, num_workers


def main():
    """Main training function."""
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting distributed PyTorch training on Spark")
    logger.info(f"Configuration: {config}")
    
    # Create Spark session
    spark = create_spark_session(config)
    
    try:
        # Setup parameter servers
        logger.info("Setting up parameter servers...")
        driver_ps, num_workers = setup_parameter_servers(spark, config)
        
        # Load training data
        logger.info(f"Loading training data from {args.data_path}")
        df = spark.read.parquet(args.data_path)
        
        # Initialize distributed trainer
        trainer = DistributedTrainer(
            config=config,
            num_workers=num_workers,
            driver_ps_host="localhost",
            driver_ps_port=config["parameter_server"]["driver_port"]
        )
        
        # Training loop
        num_epochs = config["training"]["num_epochs"]
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Shuffle data for each epoch
            shuffled_df = df.orderBy(rand())
            training_rdd = shuffled_df.rdd
            
            # Repartition if needed
            if config["training"].get("num_partitions"):
                training_rdd = training_rdd.repartition(config["training"]["num_partitions"])
            
            # Train on this epoch's data
            trainer.train_epoch(training_rdd, epoch)
            
            # Synchronization barrier
            spark.sparkContext.setJobGroup(f"epoch_{epoch}", f"Training epoch {epoch}")
            
            # Optional: Save checkpoint
            if (epoch + 1) % config["training"].get("checkpoint_interval", 10) == 0:
                logger.info(f"Saving checkpoint at epoch {epoch + 1}")
                driver_ps.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
            
            logger.info(f"Completed epoch {epoch + 1}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        spark.stop()


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    main() 