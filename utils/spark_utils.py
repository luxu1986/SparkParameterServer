"""
Spark utility functions for distributed PyTorch training.
Helper functions for Spark context management and worker information.
"""

import logging
import socket
from typing import List, Dict, Any, Optional
from pyspark.sql import SparkSession
from pyspark import SparkContext, TaskContext
import os


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration for distributed training.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Set up basic logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce verbosity of some third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('py4j').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")


def get_worker_info(spark: SparkSession) -> List[Dict[str, Any]]:
    """
    Get information about Spark worker nodes.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        List of dictionaries containing worker information
    """
    sc = spark.sparkContext
    
    # Get executor information
    status = sc.statusTracker()
    executor_infos = status.getExecutorInfos()
    
    workers = []
    for executor in executor_infos:
        if executor.executorId != "driver":  # Exclude driver
            worker_info = {
                'executor_id': executor.executorId,
                'host': executor.host,
                'port': executor.port if hasattr(executor, 'port') else None,
                'total_cores': executor.totalCores,
                'max_memory': executor.maxMemory,
                'is_active': executor.isActive
            }
            workers.append(worker_info)
    
    return workers


def get_worker_hosts(spark: Optional[SparkSession] = None) -> List[str]:
    """
    Get list of worker hostnames.
    
    Args:
        spark: SparkSession instance (optional, will get active session if None)
        
    Returns:
        List of worker hostnames
    """
    if spark is None:
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found")
    
    worker_info = get_worker_info(spark)
    return [worker['host'] for worker in worker_info]


def get_current_worker_id() -> Optional[int]:
    """
    Get the current worker/executor ID from task context.
    
    Returns:
        Worker ID if running in a Spark task, None otherwise
    """
    task_context = TaskContext.get()
    if task_context:
        return task_context.partitionId()
    return None


def get_current_hostname() -> str:
    """
    Get the hostname of the current machine.
    
    Returns:
        Hostname string
    """
    return socket.gethostname()


def get_current_ip() -> str:
    """
    Get the IP address of the current machine.
    
    Returns:
        IP address string
    """
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip
    except Exception:
        return "127.0.0.1"


def broadcast_parameter_servers(spark: SparkSession, ps_info: Dict[str, Any]) -> None:
    """
    Broadcast parameter server information to all workers.
    
    Args:
        spark: SparkSession instance
        ps_info: Dictionary containing parameter server information
    """
    sc = spark.sparkContext
    broadcast_ps = sc.broadcast(ps_info)
    
    # Store in SparkContext for later retrieval
    sc.setLocalProperty("ps_info", broadcast_ps)


def get_broadcasted_ps_info(spark: SparkSession) -> Optional[Dict[str, Any]]:
    """
    Get broadcasted parameter server information.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        Parameter server information dictionary or None
    """
    sc = spark.sparkContext
    broadcast_ps = sc.getLocalProperty("ps_info")
    
    if broadcast_ps:
        return broadcast_ps.value
    return None


def configure_spark_for_pytorch(spark_builder) -> SparkSession:
    """
    Configure Spark session with optimal settings for PyTorch training.
    
    Args:
        spark_builder: SparkSession.Builder instance
        
    Returns:
        Configured SparkSession
    """
    return spark_builder \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
        .config("spark.task.maxFailures", "3") \
        .config("spark.stage.maxConsecutiveAttempts", "8") \
        .config("spark.kubernetes.executor.deleteOnTermination", "true") \
        .getOrCreate()


def create_worker_rdd(spark: SparkSession, num_workers: int):
    """
    Create an RDD with one partition per worker for initialization.
    
    Args:
        spark: SparkSession instance
        num_workers: Number of workers
        
    Returns:
        RDD with worker IDs
    """
    sc = spark.sparkContext
    return sc.parallelize(range(num_workers), num_workers)


def barrier_sync(spark: SparkSession, message: str = "sync") -> None:
    """
    Synchronize all workers using Spark's barrier execution mode.
    
    Args:
        spark: SparkSession instance
        message: Sync message for logging
    """
    sc = spark.sparkContext
    
    def sync_function(iterator):
        # This function will be executed on all workers
        yield f"Worker synced: {message}"
    
    # Create a simple RDD and use barrier to sync
    rdd = sc.parallelize([1], 1)
    rdd.barrier().mapPartitions(sync_function).collect()


def get_spark_config_summary(spark: SparkSession) -> Dict[str, str]:
    """
    Get a summary of important Spark configuration settings.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        Dictionary of configuration settings
    """
    conf = spark.sparkContext.getConf()
    
    important_configs = [
        "spark.app.name",
        "spark.master",
        "spark.executor.instances",
        "spark.executor.cores",
        "spark.executor.memory",
        "spark.driver.memory",
        "spark.sql.adaptive.enabled",
        "spark.serializer",
        "spark.sql.execution.arrow.pyspark.enabled"
    ]
    
    config_summary = {}
    for config_key in important_configs:
        config_summary[config_key] = conf.get(config_key, "Not set")
    
    return config_summary


def check_pytorch_availability() -> Dict[str, Any]:
    """
    Check PyTorch availability and configuration on workers.
    
    Returns:
        Dictionary with PyTorch information
    """
    try:
        import torch
        
        info = {
            "pytorch_available": True,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_cached"] = torch.cuda.memory_reserved(0)
        
        return info
        
    except ImportError:
        return {
            "pytorch_available": False,
            "error": "PyTorch not available"
        }


def distributed_pytorch_check(spark: SparkSession) -> List[Dict[str, Any]]:
    """
    Check PyTorch availability across all workers.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        List of PyTorch information from each worker
    """
    def check_worker_pytorch(partition_id):
        """Check PyTorch on a single worker."""
        info = check_pytorch_availability()
        info["worker_id"] = partition_id
        info["hostname"] = get_current_hostname()
        info["ip_address"] = get_current_ip()
        return [info]
    
    # Get number of workers
    num_workers = len(get_worker_info(spark))
    if num_workers == 0:
        num_workers = 2  # Default fallback
    
    # Create RDD and check PyTorch on each worker
    worker_rdd = create_worker_rdd(spark, num_workers)
    results = worker_rdd.mapPartitionsWithIndex(
        lambda idx, _: check_worker_pytorch(idx)
    ).collect()
    
    return results


def setup_worker_environment(python_path: Optional[str] = None, 
                           additional_env: Optional[Dict[str, str]] = None) -> None:
    """
    Setup environment variables for workers.
    
    Args:
        python_path: Additional Python path entries
        additional_env: Additional environment variables
    """
    if python_path:
        current_path = os.environ.get('PYTHONPATH', '')
        if current_path:
            os.environ['PYTHONPATH'] = f"{python_path}:{current_path}"
        else:
            os.environ['PYTHONPATH'] = python_path
    
    if additional_env:
        for key, value in additional_env.items():
            os.environ[key] = value
    
    # Set optimal threading for PyTorch
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def log_cluster_info(spark: SparkSession) -> None:
    """
    Log comprehensive cluster information.
    
    Args:
        spark: SparkSession instance
    """
    logger = logging.getLogger(__name__)
    
    # Spark configuration
    config_summary = get_spark_config_summary(spark)
    logger.info("Spark Configuration:")
    for key, value in config_summary.items():
        logger.info(f"  {key}: {value}")
    
    # Worker information
    workers = get_worker_info(spark)
    logger.info(f"Cluster has {len(workers)} workers:")
    for worker in workers:
        logger.info(f"  Worker {worker['executor_id']}: {worker['host']} "
                   f"({worker['total_cores']} cores, {worker['max_memory']} memory)")
    
    # PyTorch availability
    pytorch_info = distributed_pytorch_check(spark)
    logger.info("PyTorch availability across workers:")
    for info in pytorch_info:
        status = "Available" if info["pytorch_available"] else "Not Available"
        logger.info(f"  Worker {info['worker_id']} ({info['hostname']}): {status}")
        if info["pytorch_available"]:
            logger.info(f"    PyTorch version: {info['pytorch_version']}")
            logger.info(f"    CUDA available: {info['cuda_available']}")


# Utility class for managing Spark resources
class SparkResourceManager:
    """Utility class for managing Spark resources and configuration."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.sc = spark.sparkContext
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_optimal_partitions(self, data_size_mb: float) -> int:
        """
        Calculate optimal number of partitions based on data size.
        
        Args:
            data_size_mb: Data size in megabytes
            
        Returns:
            Recommended number of partitions
        """
        # Rule of thumb: 128MB per partition
        target_partition_size_mb = 128
        num_partitions = max(1, int(data_size_mb / target_partition_size_mb))
        
        # Don't exceed number of available cores
        total_cores = self.get_total_cores()
        num_partitions = min(num_partitions, total_cores * 2)
        
        self.logger.info(f"Recommended partitions for {data_size_mb}MB: {num_partitions}")
        return num_partitions
    
    def get_total_cores(self) -> int:
        """Get total number of cores across all workers."""
        workers = get_worker_info(self.spark)
        return sum(worker['total_cores'] for worker in workers)
    
    def get_total_memory(self) -> int:
        """Get total memory across all workers."""
        workers = get_worker_info(self.spark)
        return sum(worker['max_memory'] for worker in workers)
    
    def optimize_for_training(self) -> None:
        """Apply optimizations for training workloads."""
        # Set appropriate checkpoint directory
        self.sc.setCheckpointDir("/tmp/spark-checkpoints")
        
        # Configure for iterative algorithms
        self.sc.setLocalProperty("spark.sql.adaptive.enabled", "true")
        self.sc.setLocalProperty("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        self.logger.info("Applied training optimizations") 