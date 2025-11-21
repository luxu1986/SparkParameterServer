#!/usr/bin/env python3
"""
Example script demonstrating distributed PyTorch training on Spark.
This script generates synthetic data and runs a complete training example.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType, IntegerType

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import main as train_main
from utils.spark_utils import setup_logging


def generate_synthetic_data(num_samples: int = 10000, num_dense_features: int = 50, 
                          num_categorical_features: int = 5, max_category_id: int = 1000,
                          output_path: str = None) -> str:
    """
    Generate synthetic training data for demonstration.
    
    Args:
        num_samples: Number of training samples
        num_dense_features: Number of dense numerical features
        num_categorical_features: Number of categorical features per sample
        max_category_id: Maximum categorical feature ID
        output_path: Path to save the data (if None, uses temp directory)
        
    Returns:
        Path to the generated data file
    """
    print(f"Generating {num_samples} synthetic training samples...")
    
    # Generate dense features (normally distributed)
    dense_features = np.random.randn(num_samples, num_dense_features)
    
    # Generate categorical features (random integers)
    categorical_features = []
    for _ in range(num_samples):
        cat_features = np.random.randint(0, max_category_id, num_categorical_features)
        categorical_features.append(cat_features.tolist())
    
    # Generate synthetic targets (regression task)
    # Target is a linear combination of dense features plus some noise
    weights = np.random.randn(num_dense_features) * 0.1
    targets = np.dot(dense_features, weights) + np.random.randn(num_samples) * 0.1
    
    # Create DataFrame
    data = {
        'features': [row.tolist() for row in dense_features],
        'target': targets.tolist(),
        'categorical_features': categorical_features
    }
    
    df = pd.DataFrame(data)
    
    # Save to parquet
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), 'synthetic_training_data.parquet')
    
    df.to_parquet(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")
    
    return output_path


def create_example_config(num_dense_features: int = 50) -> dict:
    """
    Create an example configuration for training.
    
    Args:
        num_dense_features: Number of dense features in the data
        
    Returns:
        Configuration dictionary
    """
    return {
        "model": {
            "input_dim": num_dense_features + 64,  # dense features + embedding dim
            "hidden_dims": [256, 128, 64],
            "output_dim": 1,
            "dropout": 0.1,
            "activation": "relu"
        },
        "embedding": {
            "embedding_dim": 64,
            "max_features": 1000,
            "init_std": 0.1
        },
        "training": {
            "num_epochs": 5,
            "batch_size": 512,
            "learning_rate": 0.01,
            "loss_function": "mse",
            "checkpoint_interval": 2
        },
        "parameter_server": {
            "driver_port": 5000,
            "worker_port_base": 5001,
            "timeout": 30.0,
            "max_retries": 3
        },
        "spark": {
            "app_name": "DistributedPyTorchExample",
            "log_level": "WARN"
        },
        "logging": {
            "level": "INFO"
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "./example_checkpoints"
        }
    }


def run_example():
    """Run the complete example."""
    print("=" * 60)
    print("Distributed PyTorch Training on Spark - Example")
    print("=" * 60)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Generate synthetic data
        data_path = generate_synthetic_data(
            num_samples=5000,  # Smaller dataset for example
            num_dense_features=50,
            num_categorical_features=3,
            max_category_id=500
        )
        
        # Create configuration
        config = create_example_config(num_dense_features=50)
        
        # Save configuration to temporary file
        import yaml
        config_path = os.path.join(tempfile.gettempdir(), 'example_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Configuration saved to: {config_path}")
        
        # Create Spark session for validation
        print("\nCreating Spark session...")
        spark = SparkSession.builder \
            .appName("DistributedPyTorchExample") \
            .master("local[2]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        # Validate data
        print("Validating generated data...")
        df = spark.read.parquet(data_path)
        print(f"Data schema:")
        df.printSchema()
        print(f"Number of rows: {df.count()}")
        print(f"Sample data:")
        df.show(5, truncate=False)
        
        spark.stop()
        
        # Run training (simulate command line arguments)
        print("\n" + "=" * 60)
        print("Starting distributed training...")
        print("=" * 60)
        
        # Simulate sys.argv for main function
        original_argv = sys.argv
        sys.argv = [
            'main.py',
            '--config', config_path,
            '--data-path', data_path,
            '--log-level', 'INFO'
        ]
        
        try:
            # Run the main training function
            train_main()
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            print(f"\nTraining failed with error: {str(e)}")
            print("This is expected in a demo environment without proper cluster setup.")
            
        finally:
            sys.argv = original_argv
        
        # Cleanup
        print(f"\nCleaning up temporary files...")
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists(config_path):
            os.remove(config_path)
        
        print("Example completed!")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        print(f"Example failed with error: {str(e)}")
        raise


def validate_environment():
    """Validate that the environment is properly set up."""
    print("Validating environment...")
    
    # Check Python packages
    required_packages = ['torch', 'pyspark', 'numpy', 'pandas', 'yaml', 'requests', 'flask']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    # Check Spark
    try:
        spark = SparkSession.builder \
            .appName("ValidationTest") \
            .master("local[1]") \
            .getOrCreate()
        print("✓ Spark is working")
        spark.stop()
    except Exception as e:
        print(f"✗ Spark validation failed: {str(e)}")
        return False
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} is available")
        if torch.cuda.is_available():
            print(f"✓ CUDA is available with {torch.cuda.device_count()} devices")
        else:
            print("ℹ CUDA is not available (CPU-only mode)")
    except Exception as e:
        print(f"✗ PyTorch validation failed: {str(e)}")
        return False
    
    print("Environment validation completed!")
    return True


if __name__ == "__main__":
    print("Distributed PyTorch Training on Spark - Example Script")
    print()
    
    # Validate environment first
    if not validate_environment():
        print("Environment validation failed. Please fix the issues and try again.")
        sys.exit(1)
    
    print()
    
    # Run the example
    try:
        run_example()
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed: {str(e)}")
        sys.exit(1) 