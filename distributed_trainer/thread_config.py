"""Thread configuration for MKL, OpenMP, and TensorFlow.

IMPORTANT: configure_threads() MUST be called before importing numpy or tensorflow
to ensure environment variables are set before library initialization.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThreadConfig:
    """Configuration for MKL and OpenMP threading."""
    
    mkl_num_threads: int = 4
    openmp_num_threads: int = 4
    tf_intra_op_parallelism: int = 4
    tf_inter_op_parallelism: int = 2


def configure_threads(config: ThreadConfig):
    """
    Configure MKL, OpenMP, and TensorFlow threading.
    
    MUST be called before importing numpy or tensorflow to ensure
    environment variables are properly set before library initialization.
    
    Sets environment variables:
    - MKL_NUM_THREADS: Intel MKL threading
    - OMP_NUM_THREADS: OpenMP threading
    - OPENBLAS_NUM_THREADS: OpenBLAS threading
    - NUMEXPR_NUM_THREADS: NumExpr threading
    - TF_NUM_INTEROP_THREADS: TensorFlow inter-op parallelism
    - TF_NUM_INTRAOP_THREADS: TensorFlow intra-op parallelism
    
    Args:
        config: ThreadConfig with thread counts
    """
    # Set environment variables for BLAS/LAPACK libraries
    os.environ["MKL_NUM_THREADS"] = str(config.mkl_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(config.openmp_num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(config.openmp_num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(config.openmp_num_threads)
    
    # Set TensorFlow threading via environment variables
    # (works before TF is imported)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(config.tf_inter_op_parallelism)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(config.tf_intra_op_parallelism)
    
    # Disable TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def configure_tensorflow_threads(config: ThreadConfig):
    """
    Configure TensorFlow threading after TF is imported.
    
    Call this after importing tensorflow if you couldn't call
    configure_threads() before the import.
    
    Args:
        config: ThreadConfig with thread counts
    """
    try:
        import tensorflow as tf
        
        tf.config.threading.set_inter_op_parallelism_threads(
            config.tf_inter_op_parallelism
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            config.tf_intra_op_parallelism
        )
    except Exception:
        # TensorFlow not available or already configured
        pass


def get_optimal_thread_config(
    num_cores: int,
    num_workers_per_node: int
) -> ThreadConfig:
    """
    Calculate optimal thread configuration based on available cores.
    
    Divides cores among workers with some overhead for system processes.
    
    Args:
        num_cores: Total number of CPU cores available
        num_workers_per_node: Number of worker processes per node
        
    Returns:
        Optimal ThreadConfig for the given resources
    """
    # Reserve ~10% of cores for system overhead
    usable_cores = max(1, int(num_cores * 0.9))
    
    # Divide cores among workers
    cores_per_worker = max(1, usable_cores // num_workers_per_node)
    
    # For TensorFlow, use most cores for intra-op (within single op)
    # and fewer for inter-op (between ops)
    tf_intra = max(1, int(cores_per_worker * 0.75))
    tf_inter = max(1, cores_per_worker - tf_intra)
    
    return ThreadConfig(
        mkl_num_threads=cores_per_worker,
        openmp_num_threads=cores_per_worker,
        tf_intra_op_parallelism=tf_intra,
        tf_inter_op_parallelism=tf_inter,
    )


def get_current_thread_config() -> dict:
    """
    Get current thread configuration from environment.
    
    Returns:
        Dict with current thread settings
    """
    return {
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "not set"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "not set"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "not set"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "not set"),
        "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS", "not set"),
        "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS", "not set"),
    }

