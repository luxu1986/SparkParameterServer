"""Distributed logging utilities for PySpark Parameter Server."""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
import threading


class PSLogger:
    """
    Thread-safe logger for distributed parameter server components.
    
    Provides consistent logging format across driver and executors
    with optional file output and log level configuration.
    """
    
    _instances: dict = {}
    _lock = threading.Lock()
    
    def __new__(cls, name: str, *args, **kwargs):
        """Singleton pattern per logger name."""
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        include_hostname: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name (typically component name)
            level: Logging level
            log_file: Optional file path for log output
            include_hostname: Include hostname in log messages
        """
        if hasattr(self, "_initialized"):
            return
        
        self._initialized = True
        self.name = name
        self.include_hostname = include_hostname
        
        # Get hostname for distributed logging
        try:
            import socket
            self.hostname = socket.gethostname()
        except Exception:
            self.hostname = "unknown"
        
        # Get process/thread info
        self.pid = os.getpid()
        
        # Create logger
        self.logger = logging.getLogger(f"pyspark_ps.{name}")
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatter
        if include_hostname:
            fmt = f"%(asctime)s | {self.hostname}:{self.pid} | %(name)s | %(levelname)s | %(message)s"
        else:
            fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def set_level(self, level: int):
        """Set logging level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


def get_logger(name: str, level: int = logging.INFO) -> PSLogger:
    """
    Get a logger instance for the given component name.
    
    Args:
        name: Component name (e.g., "server", "worker_client", "main_client")
        level: Logging level
        
    Returns:
        PSLogger instance
    """
    return PSLogger(name, level=level)


class MetricsLogger:
    """
    Logger for performance metrics and statistics.
    
    Collects and aggregates metrics across the distributed system.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"metrics.{name}")
        self._metrics: dict = {}
        self._lock = threading.Lock()
    
    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                }
            
            m = self._metrics[metric_name]
            m["count"] += 1
            m["sum"] += value
            m["min"] = min(m["min"], value)
            m["max"] = max(m["max"], value)
    
    def get_stats(self, metric_name: str) -> dict:
        """Get statistics for a metric."""
        with self._lock:
            if metric_name not in self._metrics:
                return {}
            
            m = self._metrics[metric_name]
            count = m["count"]
            return {
                "count": count,
                "sum": m["sum"],
                "mean": m["sum"] / count if count > 0 else 0,
                "min": m["min"] if count > 0 else 0,
                "max": m["max"] if count > 0 else 0,
            }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all metrics."""
        with self._lock:
            return {name: self.get_stats(name) for name in self._metrics}
    
    def log_stats(self):
        """Log all metric statistics."""
        stats = self.get_all_stats()
        for name, s in stats.items():
            self.logger.info(
                f"{name}: count={s['count']}, mean={s['mean']:.4f}, "
                f"min={s['min']:.4f}, max={s['max']:.4f}"
            )
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()

