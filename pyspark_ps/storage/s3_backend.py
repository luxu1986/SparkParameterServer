"""S3 backend for checkpoint storage."""

import io
import json
import time
from typing import Any, Dict, List, Optional, BinaryIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyspark_ps.utils.config import PSConfig
from pyspark_ps.utils.logging import get_logger

try:
    import boto3
    from botocore.config import Config as BotoConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class S3Backend:
    """
    S3 storage backend for checkpoints and model artifacts.
    
    Features:
    - Multipart uploads for large files
    - Parallel uploads/downloads
    - Compression support
    - S3-compatible storage (MinIO, etc.)
    """
    
    def __init__(self, config: PSConfig):
        """
        Initialize S3 backend.
        
        Args:
            config: PS configuration with S3 settings
        """
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for S3 backend. Install with: pip install boto3")
        
        self.config = config
        self.logger = get_logger("s3_backend")
        
        # S3 client configuration
        boto_config = BotoConfig(
            max_pool_connections=config.s3_max_concurrency,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )
        
        client_kwargs = {
            "config": boto_config,
            "region_name": config.s3_region,
        }
        
        if config.s3_endpoint_url:
            client_kwargs["endpoint_url"] = config.s3_endpoint_url
        
        self._s3 = boto3.client("s3", **client_kwargs)
        self._executor = ThreadPoolExecutor(max_workers=config.s3_max_concurrency)
    
    def _parse_s3_path(self, s3_path: str) -> tuple:
        """Parse S3 URI into bucket and key."""
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        
        parts = s3_path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        return bucket, key
    
    def upload(
        self,
        s3_path: str,
        data: bytes,
        content_type: str = "application/octet-stream"
    ):
        """
        Upload data to S3.
        
        Args:
            s3_path: Full S3 path (s3://bucket/key)
            data: Bytes to upload
            content_type: MIME type
        """
        bucket, key = self._parse_s3_path(s3_path)
        
        if len(data) > self.config.s3_multipart_threshold:
            self._multipart_upload(bucket, key, data, content_type)
        else:
            self._s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type
            )
        
        self.logger.debug(f"Uploaded {len(data)} bytes to {s3_path}")
    
    def _multipart_upload(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str
    ):
        """Perform multipart upload for large files."""
        # Create multipart upload
        mpu = self._s3.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            ContentType=content_type
        )
        upload_id = mpu["UploadId"]
        
        try:
            parts = []
            part_size = self.config.s3_multipart_threshold
            num_parts = (len(data) + part_size - 1) // part_size
            
            futures = []
            for i in range(num_parts):
                start = i * part_size
                end = min(start + part_size, len(data))
                part_data = data[start:end]
                
                future = self._executor.submit(
                    self._upload_part,
                    bucket, key, upload_id, i + 1, part_data
                )
                futures.append((i + 1, future))
            
            for part_num, future in futures:
                etag = future.result()
                parts.append({"PartNumber": part_num, "ETag": etag})
            
            # Complete multipart upload
            parts.sort(key=lambda x: x["PartNumber"])
            self._s3.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts}
            )
            
        except Exception as e:
            # Abort on failure
            self._s3.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id
            )
            raise
    
    def _upload_part(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes
    ) -> str:
        """Upload a single part of multipart upload."""
        response = self._s3.upload_part(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=data
        )
        return response["ETag"]
    
    def download(self, s3_path: str) -> bytes:
        """
        Download data from S3.
        
        Args:
            s3_path: Full S3 path
            
        Returns:
            Downloaded bytes
        """
        bucket, key = self._parse_s3_path(s3_path)
        
        response = self._s3.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        
        self.logger.debug(f"Downloaded {len(data)} bytes from {s3_path}")
        return data
    
    def exists(self, s3_path: str) -> bool:
        """Check if an S3 object exists."""
        bucket, key = self._parse_s3_path(s3_path)
        
        try:
            self._s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    
    def list_objects(
        self,
        s3_prefix: str,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects under an S3 prefix.
        
        Args:
            s3_prefix: S3 path prefix
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object metadata dicts
        """
        bucket, prefix = self._parse_s3_path(s3_prefix)
        
        objects = []
        paginator = self._s3.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys
        ):
            for obj in page.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].timestamp(),
                    "etag": obj["ETag"],
                })
        
        return objects
    
    def delete(self, s3_path: str):
        """Delete an S3 object."""
        bucket, key = self._parse_s3_path(s3_path)
        self._s3.delete_object(Bucket=bucket, Key=key)
        self.logger.debug(f"Deleted {s3_path}")
    
    def delete_prefix(self, s3_prefix: str):
        """Delete all objects under a prefix."""
        bucket, prefix = self._parse_s3_path(s3_prefix)
        
        objects = self.list_objects(s3_prefix)
        
        if not objects:
            return
        
        # Delete in batches of 1000 (S3 limit)
        for i in range(0, len(objects), 1000):
            batch = objects[i:i + 1000]
            self._s3.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [{"Key": obj["key"]} for obj in batch]
                }
            )
        
        self.logger.debug(f"Deleted {len(objects)} objects under {s3_prefix}")
    
    def upload_stream(
        self,
        s3_path: str,
        stream: BinaryIO,
        content_type: str = "application/octet-stream"
    ):
        """Upload from a file-like stream."""
        bucket, key = self._parse_s3_path(s3_path)
        self._s3.upload_fileobj(stream, bucket, key)
    
    def download_stream(self, s3_path: str, stream: BinaryIO):
        """Download to a file-like stream."""
        bucket, key = self._parse_s3_path(s3_path)
        self._s3.download_fileobj(bucket, key, stream)
    
    def close(self):
        """Close backend resources."""
        self._executor.shutdown(wait=True)


class LocalStorageBackend:
    """
    Local filesystem backend for testing.
    
    Mimics S3Backend interface for local file operations.
    """
    
    def __init__(self, base_path: str = "/tmp/pyspark_ps"):
        """
        Initialize local backend.
        
        Args:
            base_path: Base directory for storage
        """
        import os
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.logger = get_logger("local_backend")
    
    def _get_path(self, path: str) -> str:
        """Convert path to local filesystem path."""
        import os
        
        # Strip s3:// prefix if present
        if path.startswith("s3://"):
            path = path[5:]
        
        # Remove leading slashes
        path = path.lstrip("/")
        
        full_path = os.path.join(self.base_path, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path
    
    def upload(
        self,
        path: str,
        data: bytes,
        content_type: str = "application/octet-stream"
    ):
        """Write data to local file."""
        local_path = self._get_path(path)
        with open(local_path, "wb") as f:
            f.write(data)
        self.logger.debug(f"Wrote {len(data)} bytes to {local_path}")
    
    def download(self, path: str) -> bytes:
        """Read data from local file."""
        local_path = self._get_path(path)
        with open(local_path, "rb") as f:
            data = f.read()
        self.logger.debug(f"Read {len(data)} bytes from {local_path}")
        return data
    
    def exists(self, path: str) -> bool:
        """Check if file exists."""
        import os
        return os.path.exists(self._get_path(path))
    
    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """List files under prefix."""
        import os
        
        local_prefix = self._get_path(prefix)
        objects = []
        
        if os.path.isdir(local_prefix):
            for root, dirs, files in os.walk(local_prefix):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    stat = os.stat(full_path)
                    objects.append({
                        "key": os.path.relpath(full_path, self.base_path),
                        "size": stat.st_size,
                        "last_modified": stat.st_mtime,
                    })
                    
                    if len(objects) >= max_keys:
                        return objects
        
        return objects
    
    def delete(self, path: str):
        """Delete a file."""
        import os
        local_path = self._get_path(path)
        if os.path.exists(local_path):
            os.remove(local_path)
    
    def delete_prefix(self, prefix: str):
        """Delete all files under prefix."""
        import os
        import shutil
        
        local_prefix = self._get_path(prefix)
        if os.path.isdir(local_prefix):
            shutil.rmtree(local_prefix)
    
    def close(self):
        """No-op for local backend."""
        pass

