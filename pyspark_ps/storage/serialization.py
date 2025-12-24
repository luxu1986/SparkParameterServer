"""Model and embedding serialization for storage."""

import io
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


class ModelSerializer:
    """
    Serializer for model weights and embeddings.
    
    Supports multiple formats:
    - NumPy (.npy, .npz)
    - Parquet (for embeddings with metadata)
    - Arrow (for efficient columnar storage)
    """
    
    def __init__(self, compression: str = "zstd"):
        """
        Initialize serializer.
        
        Args:
            compression: Compression algorithm for storage
        """
        self.compression = compression
    
    def serialize_weights(
        self,
        weights: Dict[str, np.ndarray],
        format: str = "npz"
    ) -> bytes:
        """
        Serialize weight tensors.
        
        Args:
            weights: Dict mapping layer name to weight array
            format: Output format ("npz", "npy", "arrow")
            
        Returns:
            Serialized bytes
        """
        if format == "npz":
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **weights)
            return buffer.getvalue()
        
        elif format == "npy":
            # For single tensor
            if len(weights) != 1:
                raise ValueError("npy format only supports single tensor")
            
            buffer = io.BytesIO()
            np.save(buffer, list(weights.values())[0])
            return buffer.getvalue()
        
        elif format == "arrow" and HAS_ARROW:
            # Use Arrow for columnar storage
            arrays = {}
            for name, arr in weights.items():
                # Flatten for storage
                arrays[name] = arr.flatten()
                arrays[f"{name}_shape"] = np.array(arr.shape, dtype=np.int64)
            
            table = pa.table(arrays)
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression=self.compression)
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def deserialize_weights(
        self,
        data: bytes,
        format: str = "npz"
    ) -> Dict[str, np.ndarray]:
        """
        Deserialize weight tensors.
        
        Args:
            data: Serialized bytes
            format: Input format
            
        Returns:
            Dict mapping layer name to weight array
        """
        if format == "npz":
            buffer = io.BytesIO(data)
            with np.load(buffer) as npz:
                return {name: npz[name] for name in npz.files}
        
        elif format == "npy":
            buffer = io.BytesIO(data)
            arr = np.load(buffer)
            return {"weights": arr}
        
        elif format == "arrow" and HAS_ARROW:
            buffer = io.BytesIO(data)
            table = pq.read_table(buffer)
            
            weights = {}
            for name in table.column_names:
                if name.endswith("_shape"):
                    continue
                
                shape_name = f"{name}_shape"
                if shape_name in table.column_names:
                    arr = table.column(name).to_numpy()
                    shape = tuple(table.column(shape_name).to_numpy())
                    weights[name] = arr.reshape(shape)
                else:
                    weights[name] = table.column(name).to_numpy()
            
            return weights
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def serialize_embeddings(
        self,
        embeddings: Dict[int, np.ndarray],
        update_counts: Optional[Dict[int, float]] = None
    ) -> bytes:
        """
        Serialize embeddings to Parquet format.
        
        Args:
            embeddings: Dict mapping token_id to embedding
            update_counts: Optional dict of update counts
            
        Returns:
            Serialized Parquet bytes
        """
        if not embeddings:
            # Return empty Parquet
            buffer = io.BytesIO()
            if HAS_ARROW:
                table = pa.table({
                    "token_id": pa.array([], type=pa.int64()),
                    "embedding": pa.array([], type=pa.list_(pa.float32())),
                    "update_count": pa.array([], type=pa.float64()),
                })
                pq.write_table(table, buffer, compression=self.compression)
            return buffer.getvalue()
        
        token_ids = list(embeddings.keys())
        emb_list = [embeddings[tid].tolist() for tid in token_ids]
        counts = [
            update_counts.get(tid, 1.0) if update_counts else 1.0
            for tid in token_ids
        ]
        
        if HAS_ARROW:
            table = pa.table({
                "token_id": pa.array(token_ids, type=pa.int64()),
                "embedding": pa.array(emb_list, type=pa.list_(pa.float32())),
                "update_count": pa.array(counts, type=pa.float64()),
            })
            
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression=self.compression)
            return buffer.getvalue()
        
        else:
            # Fallback to numpy
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                token_ids=np.array(token_ids, dtype=np.int64),
                embeddings=np.stack([embeddings[tid] for tid in token_ids]),
                update_counts=np.array(counts, dtype=np.float64)
            )
            return buffer.getvalue()
    
    def deserialize_embeddings(
        self,
        data: bytes
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Deserialize embeddings from Parquet format.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Tuple of (embeddings dict, update_counts dict)
        """
        if not data:
            return {}, {}
        
        buffer = io.BytesIO(data)
        
        if HAS_ARROW:
            table = pq.read_table(buffer)
            
            token_ids = table.column("token_id").to_pylist()
            emb_arrays = table.column("embedding").to_pylist()
            counts = table.column("update_count").to_pylist()
            
            embeddings = {
                tid: np.array(emb, dtype=np.float32)
                for tid, emb in zip(token_ids, emb_arrays)
            }
            update_counts = dict(zip(token_ids, counts))
            
            return embeddings, update_counts
        
        else:
            # Fallback to numpy
            with np.load(buffer) as npz:
                token_ids = npz["token_ids"]
                emb_array = npz["embeddings"]
                counts = npz["update_counts"]
                
                embeddings = {
                    int(tid): emb_array[i]
                    for i, tid in enumerate(token_ids)
                }
                update_counts = {
                    int(tid): float(counts[i])
                    for i, tid in enumerate(token_ids)
                }
                
                return embeddings, update_counts
    
    def serialize_optimizer_state(
        self,
        state: Dict[str, Any]
    ) -> bytes:
        """
        Serialize optimizer state.
        
        Args:
            state: Optimizer state dictionary
            
        Returns:
            Serialized bytes
        """
        import pickle
        return pickle.dumps(state)
    
    def deserialize_optimizer_state(
        self,
        data: bytes
    ) -> Dict[str, Any]:
        """
        Deserialize optimizer state.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Optimizer state dictionary
        """
        import pickle
        return pickle.loads(data)


class CheckpointSerializer:
    """
    Serializer for complete checkpoint data.
    
    Handles metadata, model weights, embeddings, and optimizer states.
    """
    
    def __init__(self, compression: str = "zstd"):
        self.model_serializer = ModelSerializer(compression)
    
    def serialize_metadata(
        self,
        config: Dict[str, Any],
        stats: Dict[str, Any],
        additional: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Serialize checkpoint metadata to JSON."""
        metadata = {
            "config": config,
            "stats": stats,
            **(additional or {})
        }
        return json.dumps(metadata, indent=2).encode()
    
    def deserialize_metadata(self, data: bytes) -> Dict[str, Any]:
        """Deserialize checkpoint metadata from JSON."""
        return json.loads(data.decode())

