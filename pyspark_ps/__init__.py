"""
PySpark Parameter Server - A distributed parameter server for large-scale machine learning.

This package provides:
- PSMainClient: Main client running on Spark driver for coordination
- PSWorkerClient: Worker client running on executors for training
- PSConfig: Configuration for the parameter server system
"""

from pyspark_ps.utils.config import PSConfig
from pyspark_ps.client.main_client import PSMainClient
from pyspark_ps.client.worker_client import PSWorkerClient

__version__ = "0.1.0"
__all__ = ["PSMainClient", "PSWorkerClient", "PSConfig"]

