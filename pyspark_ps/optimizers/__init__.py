"""Optimizers for PySpark Parameter Server."""

from pyspark_ps.optimizers.base import BaseOptimizer
from pyspark_ps.optimizers.sgd import SGDOptimizer
from pyspark_ps.optimizers.adam import AdamOptimizer
from pyspark_ps.optimizers.adagrad import AdagradOptimizer
from pyspark_ps.optimizers.ftrl import FTRLOptimizer

__all__ = [
    "BaseOptimizer",
    "SGDOptimizer",
    "AdamOptimizer",
    "AdagradOptimizer",
    "FTRLOptimizer",
]


def create_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """
    Factory function to create optimizer by name.
    
    Args:
        name: Optimizer name ("sgd", "adam", "adagrad", "ftrl")
        **kwargs: Optimizer-specific parameters
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        "sgd": SGDOptimizer,
        "adam": AdamOptimizer,
        "adagrad": AdagradOptimizer,
        "ftrl": FTRLOptimizer,
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name.lower()](**kwargs)

