"""S3 Parquet data loading and partitioning."""

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class PartitionInfo:
    """Information about a data partition."""
    
    path: str
    size_bytes: int
    num_rows: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class S3Config:
    """S3 configuration for data loading."""
    
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None


class S3ParquetDataLoader:
    """
    S3 Parquet data loading and partitioning.
    
    Features:
    - Discover parquet files under S3 prefix
    - Estimate partition sizes for balanced distribution
    - Multiple distribution strategies
    """
    
    def __init__(self, s3_config: Optional[S3Config] = None):
        """
        Initialize the data loader.
        
        Args:
            s3_config: Optional S3 configuration
        """
        self.s3_config = s3_config or S3Config()
        self._s3_client = None
    
    def _get_s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            try:
                import boto3
                
                client_kwargs = {
                    "region_name": self.s3_config.region,
                }
                
                if self.s3_config.endpoint_url:
                    client_kwargs["endpoint_url"] = self.s3_config.endpoint_url
                
                if self.s3_config.access_key and self.s3_config.secret_key:
                    client_kwargs["aws_access_key_id"] = self.s3_config.access_key
                    client_kwargs["aws_secret_access_key"] = self.s3_config.secret_key
                
                self._s3_client = boto3.client("s3", **client_kwargs)
            except ImportError:
                raise ImportError("boto3 is required for S3 operations")
        
        return self._s3_client
    
    def _parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and prefix."""
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        elif s3_path.startswith("s3a://"):
            s3_path = s3_path[6:]
        
        parts = s3_path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        return bucket, prefix
    
    def discover_partitions(
        self,
        s3_path: str,
        file_pattern: str = "*.parquet"
    ) -> List[PartitionInfo]:
        """
        Discover all parquet files under S3 path.
        
        Args:
            s3_path: S3 path to search (e.g., "s3://bucket/path/")
            file_pattern: Glob pattern for file matching
            
        Returns:
            List of PartitionInfo for discovered files
        """
        s3 = self._get_s3_client()
        bucket, prefix = self._parse_s3_path(s3_path)
        
        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        partitions = []
        paginator = s3.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                
                # Match against pattern
                if fnmatch.fnmatch(filename, file_pattern):
                    partitions.append(PartitionInfo(
                        path=f"s3://{bucket}/{key}",
                        size_bytes=obj["Size"],
                        metadata={
                            "last_modified": obj["LastModified"].isoformat(),
                            "etag": obj["ETag"],
                        }
                    ))
        
        return partitions
    
    def distribute_partitions(
        self,
        partitions: List[PartitionInfo],
        num_workers: int,
        strategy: str = "size_balanced"
    ) -> Dict[int, List[str]]:
        """
        Distribute partitions across workers.
        
        Strategies:
        - "size_balanced": Balance by total bytes per worker
        - "count_balanced": Equal number of files per worker
        - "round_robin": Simple round-robin assignment
        
        Args:
            partitions: List of PartitionInfo to distribute
            num_workers: Number of workers
            strategy: Distribution strategy
            
        Returns:
            Dict mapping worker_id -> list of partition paths
        """
        if not partitions:
            return {i: [] for i in range(num_workers)}
        
        if strategy == "round_robin":
            return self._distribute_round_robin(partitions, num_workers)
        elif strategy == "count_balanced":
            return self._distribute_count_balanced(partitions, num_workers)
        elif strategy == "size_balanced":
            return self._distribute_size_balanced(partitions, num_workers)
        else:
            raise ValueError(f"Unknown distribution strategy: {strategy}")
    
    def _distribute_round_robin(
        self,
        partitions: List[PartitionInfo],
        num_workers: int
    ) -> Dict[int, List[str]]:
        """Simple round-robin distribution."""
        result = {i: [] for i in range(num_workers)}
        
        for i, partition in enumerate(partitions):
            worker_id = i % num_workers
            result[worker_id].append(partition.path)
        
        return result
    
    def _distribute_count_balanced(
        self,
        partitions: List[PartitionInfo],
        num_workers: int
    ) -> Dict[int, List[str]]:
        """Distribute equal number of partitions to each worker."""
        result = {i: [] for i in range(num_workers)}
        
        # Sort by size descending for better balance
        sorted_partitions = sorted(
            partitions,
            key=lambda p: p.size_bytes,
            reverse=True
        )
        
        for i, partition in enumerate(sorted_partitions):
            worker_id = i % num_workers
            result[worker_id].append(partition.path)
        
        return result
    
    def _distribute_size_balanced(
        self,
        partitions: List[PartitionInfo],
        num_workers: int
    ) -> Dict[int, List[str]]:
        """
        Distribute partitions to balance total size per worker.
        
        Uses greedy algorithm: assign next largest to least loaded worker.
        """
        result = {i: [] for i in range(num_workers)}
        worker_sizes = {i: 0 for i in range(num_workers)}
        
        # Sort by size descending
        sorted_partitions = sorted(
            partitions,
            key=lambda p: p.size_bytes,
            reverse=True
        )
        
        for partition in sorted_partitions:
            # Find worker with minimum total size
            min_worker = min(worker_sizes, key=worker_sizes.get)
            result[min_worker].append(partition.path)
            worker_sizes[min_worker] += partition.size_bytes
        
        return result
    
    def get_partition_schema(self, s3_path: str) -> Dict[str, str]:
        """
        Read schema from a sample parquet file.
        
        Args:
            s3_path: Path to a parquet file
            
        Returns:
            Dict mapping column name -> data type string
        """
        try:
            import pyarrow.parquet as pq
            import s3fs
            
            fs = s3fs.S3FileSystem(
                anon=False,
                client_kwargs={
                    "region_name": self.s3_config.region,
                    "endpoint_url": self.s3_config.endpoint_url,
                }
            )
            
            with fs.open(s3_path, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                schema = parquet_file.schema_arrow
            
            return {
                field.name: str(field.type)
                for field in schema
            }
            
        except ImportError:
            raise ImportError("pyarrow and s3fs are required for schema reading")
    
    def estimate_total_rows(self, partitions: List[PartitionInfo]) -> int:
        """
        Estimate total rows across all partitions.
        
        Uses file size heuristics if row counts not available.
        
        Args:
            partitions: List of partitions
            
        Returns:
            Estimated total row count
        """
        total_rows = 0
        total_size_without_rows = 0
        rows_per_byte = None
        
        for p in partitions:
            if p.num_rows is not None:
                total_rows += p.num_rows
                if rows_per_byte is None and p.size_bytes > 0:
                    rows_per_byte = p.num_rows / p.size_bytes
            else:
                total_size_without_rows += p.size_bytes
        
        # Estimate remaining rows using average rows per byte
        if total_size_without_rows > 0 and rows_per_byte is not None:
            total_rows += int(total_size_without_rows * rows_per_byte)
        elif total_size_without_rows > 0:
            # Default estimate: ~100 bytes per row for compressed parquet
            total_rows += total_size_without_rows // 100
        
        return total_rows


def format_date_path(template: str, date_str: str, date_format: str = "%Y-%m-%d") -> str:
    """
    Format a path template with a date.
    
    Supports {date} placeholder in template.
    
    Args:
        template: Path template (e.g., "s3://bucket/data/dt={date}/")
        date_str: Date string to substitute
        date_format: Expected date format
        
    Returns:
        Formatted path
    """
    return template.format(date=date_str)


def parse_date_range(
    start_date: str,
    end_date: str,
    date_format: str = "%Y-%m-%d"
) -> List[str]:
    """
    Generate list of dates in range.
    
    Args:
        start_date: Start date string
        end_date: End date string (inclusive)
        date_format: Date format string
        
    Returns:
        List of date strings
    """
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime(date_format))
        current += timedelta(days=1)
    
    return dates

