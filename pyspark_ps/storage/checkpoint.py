"""Checkpoint management for Parameter Server."""

import json
import time
import os
from typing import Any, Dict, List, Optional

from pyspark_ps.communication.protocol import CheckpointInfo
from pyspark_ps.storage.serialization import ModelSerializer, CheckpointSerializer
from pyspark_ps.utils.logging import get_logger


class CheckpointManager:
    """
    Manages checkpoint save/load operations to S3.
    
    Checkpoint structure:
        s3_path/
        ├── metadata.json          # Config, shard info, timestamp
        ├── model/
        │   ├── weights.npz        # Dense weights
        │   └── optimizer.pkl      # Weight optimizer states
        ├── embeddings/
        │   ├── shard_0.parquet    # Embeddings by server shard
        │   ├── shard_1.parquet
        │   └── ...
        └── embedding_optimizer/
            ├── shard_0.pkl        # Embedding optimizer states
            └── ...
    """
    
    def __init__(self, backend):
        """
        Initialize checkpoint manager.
        
        Args:
            backend: Storage backend (S3Backend or LocalStorageBackend)
        """
        self.backend = backend
        self.logger = get_logger("checkpoint_manager")
        self.serializer = CheckpointSerializer()
    
    def save(
        self,
        s3_path: str,
        server_states: List[Dict[str, Any]],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save checkpoint to S3.
        
        Args:
            s3_path: S3 path for checkpoint
            server_states: List of state dicts from each server
            config: Configuration dictionary
            metadata: Additional metadata
        """
        self.logger.info(f"Saving checkpoint to {s3_path}")
        
        start_time = time.time()
        
        # Calculate statistics
        total_embeddings = 0
        total_weights = 0
        
        for ss in server_states:
            state = ss.get("state", {})
            emb_state = state.get("embedding_state", {})
            weight_state = state.get("weight_state", {})
            
            total_embeddings += len(emb_state.get("embeddings", {}))
            total_weights += len(weight_state.get("weights", {}))
        
        # Save metadata
        meta = {
            "config": config,
            "timestamp": time.time(),
            "num_servers": len(server_states),
            "total_embeddings": total_embeddings,
            "total_weights": total_weights,
            "version": "1.0",
            **(metadata or {})
        }
        
        self.backend.upload(
            f"{s3_path}/metadata.json",
            json.dumps(meta, indent=2).encode(),
            content_type="application/json"
        )
        
        # Save server states
        model_serializer = ModelSerializer()
        
        for ss in server_states:
            server_id = ss["server_id"]
            state = ss.get("state", {})
            
            # Save embeddings
            emb_state = state.get("embedding_state", {})
            if emb_state.get("embeddings"):
                emb_data = model_serializer.serialize_embeddings(
                    emb_state["embeddings"],
                    emb_state.get("update_counts", {})
                )
                self.backend.upload(
                    f"{s3_path}/embeddings/shard_{server_id}.parquet",
                    emb_data,
                    content_type="application/octet-stream"
                )
            
            # Save embedding optimizer state
            if emb_state.get("optimizer_state"):
                opt_data = model_serializer.serialize_optimizer_state(
                    emb_state["optimizer_state"]
                )
                self.backend.upload(
                    f"{s3_path}/embedding_optimizer/shard_{server_id}.pkl",
                    opt_data
                )
            
            # Save weights (from first server only - they're replicated)
            if server_id == 0:
                weight_state = state.get("weight_state", {})
                if weight_state.get("weights"):
                    weight_data = model_serializer.serialize_weights(
                        weight_state["weights"],
                        format="npz"
                    )
                    self.backend.upload(
                        f"{s3_path}/model/weights.npz",
                        weight_data
                    )
                
                if weight_state.get("optimizer_state"):
                    opt_data = model_serializer.serialize_optimizer_state(
                        weight_state["optimizer_state"]
                    )
                    self.backend.upload(
                        f"{s3_path}/model/optimizer.pkl",
                        opt_data
                    )
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"Checkpoint saved: {total_embeddings} embeddings, "
            f"{total_weights} weight tensors in {elapsed:.2f}s"
        )
    
    def load(self, s3_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from S3.
        
        Args:
            s3_path: S3 path to checkpoint
            
        Returns:
            Checkpoint data including server states
        """
        self.logger.info(f"Loading checkpoint from {s3_path}")
        
        start_time = time.time()
        
        # Load metadata
        meta_data = self.backend.download(f"{s3_path}/metadata.json")
        metadata = json.loads(meta_data.decode())
        
        num_servers = metadata.get("num_servers", 1)
        model_serializer = ModelSerializer()
        
        # Load weights (shared across all servers)
        weights = {}
        weight_optimizer_state = {}
        
        if self.backend.exists(f"{s3_path}/model/weights.npz"):
            weight_data = self.backend.download(f"{s3_path}/model/weights.npz")
            weights = model_serializer.deserialize_weights(weight_data, format="npz")
        
        if self.backend.exists(f"{s3_path}/model/optimizer.pkl"):
            opt_data = self.backend.download(f"{s3_path}/model/optimizer.pkl")
            weight_optimizer_state = model_serializer.deserialize_optimizer_state(opt_data)
        
        # Load per-server states
        server_states = []
        
        for server_id in range(num_servers):
            state = {
                "embedding_state": {
                    "embeddings": {},
                    "update_counts": {},
                    "optimizer_state": {},
                },
                "weight_state": {
                    "weights": weights.copy(),
                    "optimizer_state": weight_optimizer_state.copy() if weight_optimizer_state else {},
                }
            }
            
            # Load embeddings for this shard
            emb_path = f"{s3_path}/embeddings/shard_{server_id}.parquet"
            if self.backend.exists(emb_path):
                emb_data = self.backend.download(emb_path)
                embeddings, counts = model_serializer.deserialize_embeddings(emb_data)
                state["embedding_state"]["embeddings"] = embeddings
                state["embedding_state"]["update_counts"] = counts
            
            # Load embedding optimizer state
            opt_path = f"{s3_path}/embedding_optimizer/shard_{server_id}.pkl"
            if self.backend.exists(opt_path):
                opt_data = self.backend.download(opt_path)
                state["embedding_state"]["optimizer_state"] = (
                    model_serializer.deserialize_optimizer_state(opt_data)
                )
            
            server_states.append({
                "server_id": server_id,
                "state": state
            })
        
        elapsed = time.time() - start_time
        self.logger.info(f"Checkpoint loaded in {elapsed:.2f}s")
        
        return {
            "metadata": metadata,
            "config": metadata.get("config", {}),
            "server_states": server_states,
        }
    
    def list_checkpoints(self, s3_prefix: str) -> List[CheckpointInfo]:
        """
        List available checkpoints under a prefix.
        
        Args:
            s3_prefix: S3 prefix to search
            
        Returns:
            List of CheckpointInfo sorted by timestamp (newest first)
        """
        objects = self.backend.list_objects(s3_prefix)
        
        # Find checkpoint directories by looking for metadata.json
        checkpoints = []
        seen_paths = set()
        
        for obj in objects:
            key = obj["key"]
            if key.endswith("/metadata.json"):
                ckpt_path = key.rsplit("/metadata.json", 1)[0]
                if ckpt_path not in seen_paths:
                    seen_paths.add(ckpt_path)
                    
                    # Load metadata
                    try:
                        full_path = f"s3://{ckpt_path}" if not ckpt_path.startswith("s3://") else ckpt_path
                        if s3_prefix.startswith("s3://"):
                            bucket = s3_prefix.split("/")[2]
                            full_path = f"s3://{bucket}/{key.rsplit('/metadata.json', 1)[0]}"
                        
                        meta_data = self.backend.download(f"{full_path}/metadata.json")
                        meta = json.loads(meta_data.decode())
                        
                        checkpoints.append(CheckpointInfo(
                            s3_path=full_path,
                            timestamp=meta.get("timestamp", obj["last_modified"]),
                            embedding_count=meta.get("total_embeddings", 0),
                            model_size_bytes=meta.get("model_size_bytes", 0),
                            config=meta.get("config", {}),
                            metadata=meta
                        ))
                    except Exception as e:
                        self.logger.warning(f"Error loading checkpoint metadata: {e}")
        
        # Sort by timestamp, newest first
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        return checkpoints
    
    def delete(self, s3_path: str):
        """Delete a checkpoint."""
        self.logger.info(f"Deleting checkpoint {s3_path}")
        self.backend.delete_prefix(s3_path)


class LocalCheckpointManager:
    """
    Simple local filesystem checkpoint manager for testing.
    """
    
    def __init__(self, base_path: str = "/tmp/pyspark_ps_checkpoints"):
        """
        Initialize local checkpoint manager.
        
        Args:
            base_path: Base directory for checkpoints
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.logger = get_logger("local_checkpoint")
    
    def _get_path(self, path: str) -> str:
        """Get full local path."""
        if path.startswith("/"):
            return path
        return os.path.join(self.base_path, path)
    
    def save(
        self,
        path: str,
        server_states: List[Dict[str, Any]],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save checkpoint to local filesystem."""
        import pickle
        
        full_path = self._get_path(path)
        os.makedirs(full_path, exist_ok=True)
        
        # Save metadata
        meta = {
            "config": config,
            "timestamp": time.time(),
            "num_servers": len(server_states),
            **(metadata or {})
        }
        
        with open(os.path.join(full_path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save server states
        with open(os.path.join(full_path, "states.pkl"), "wb") as f:
            pickle.dump(server_states, f)
        
        self.logger.info(f"Saved checkpoint to {full_path}")
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load checkpoint from local filesystem."""
        import pickle
        
        full_path = self._get_path(path)
        
        with open(os.path.join(full_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        with open(os.path.join(full_path, "states.pkl"), "rb") as f:
            server_states = pickle.load(f)
        
        return {
            "metadata": metadata,
            "config": metadata.get("config", {}),
            "server_states": server_states,
        }
    
    def list_checkpoints(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List available checkpoints."""
        checkpoints = []
        
        search_path = self._get_path(prefix) if prefix else self.base_path
        
        if os.path.isdir(search_path):
            for name in os.listdir(search_path):
                ckpt_path = os.path.join(search_path, name)
                meta_path = os.path.join(ckpt_path, "metadata.json")
                
                if os.path.isfile(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    
                    checkpoints.append({
                        "path": ckpt_path,
                        "timestamp": meta.get("timestamp", 0),
                        "metadata": meta
                    })
        
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def delete(self, path: str):
        """Delete a checkpoint."""
        import shutil
        
        full_path = self._get_path(path)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            self.logger.info(f"Deleted checkpoint {full_path}")

