"""Efficient serialization for Parameter Server communication."""

import io
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


class Serializer:
    """
    High-performance serializer for PS data types.
    
    Supports:
    - NumPy arrays (zero-copy when possible)
    - Python primitives via msgpack/pickle
    - Optional compression (lz4, zstd)
    """
    
    # Type markers for deserialization
    TYPE_NUMPY = 0x01
    TYPE_DICT = 0x02
    TYPE_LIST = 0x03
    TYPE_PRIMITIVE = 0x04
    TYPE_EMBEDDING_BATCH = 0x05
    TYPE_GRADIENT_BATCH = 0x06
    
    def __init__(
        self,
        compression: bool = True,
        compression_algorithm: str = "lz4",
        compression_level: int = 1
    ):
        """
        Initialize serializer.
        
        Args:
            compression: Enable compression
            compression_algorithm: Algorithm ("lz4", "zstd", "none")
            compression_level: Compression level (higher = more compression)
        """
        self.compression = compression
        self.compression_algorithm = compression_algorithm
        self.compression_level = compression_level
        
        # Validate compression availability
        if compression and compression_algorithm == "lz4" and not HAS_LZ4:
            self.compression = False
        if compression and compression_algorithm == "zstd" and not HAS_ZSTD:
            self.compression = False
    
    def serialize(self, data: Any) -> bytes:
        """
        Serialize data to bytes.
        
        Args:
            data: Data to serialize (numpy array, dict, list, or primitive)
            
        Returns:
            Serialized bytes
        """
        raw_bytes = self._serialize_internal(data)
        
        if self.compression and len(raw_bytes) > 1024:  # Only compress if > 1KB
            return self._compress(raw_bytes)
        
        # Prepend compression flag (0 = uncompressed)
        return b'\x00' + raw_bytes
    
    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to data.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized data
        """
        if not data:
            return None
        
        # Check compression flag
        compression_flag = data[0]
        raw_bytes = data[1:]
        
        if compression_flag != 0:
            raw_bytes = self._decompress(raw_bytes, compression_flag)
        
        return self._deserialize_internal(raw_bytes)
    
    def _serialize_internal(self, data: Any) -> bytes:
        """Internal serialization without compression."""
        if isinstance(data, np.ndarray):
            return self._serialize_numpy(data)
        elif isinstance(data, dict):
            return self._serialize_dict(data)
        elif isinstance(data, (list, tuple)):
            return self._serialize_list(data)
        else:
            return self._serialize_primitive(data)
    
    def _deserialize_internal(self, data: bytes) -> Any:
        """Internal deserialization."""
        if not data:
            return None
        
        type_marker = data[0]
        payload = data[1:]
        
        if type_marker == self.TYPE_NUMPY:
            return self._deserialize_numpy(payload)
        elif type_marker == self.TYPE_DICT:
            return self._deserialize_dict(payload)
        elif type_marker == self.TYPE_LIST:
            return self._deserialize_list(payload)
        elif type_marker == self.TYPE_PRIMITIVE:
            return self._deserialize_primitive(payload)
        elif type_marker == self.TYPE_EMBEDDING_BATCH:
            return self._deserialize_embedding_batch(payload)
        elif type_marker == self.TYPE_GRADIENT_BATCH:
            return self._deserialize_gradient_batch(payload)
        else:
            raise ValueError(f"Unknown type marker: {type_marker}")
    
    def _serialize_numpy(self, arr: np.ndarray) -> bytes:
        """Serialize numpy array."""
        buffer = io.BytesIO()
        buffer.write(bytes([self.TYPE_NUMPY]))
        
        # Write dtype string
        dtype_str = str(arr.dtype).encode()
        buffer.write(struct.pack("<I", len(dtype_str)))
        buffer.write(dtype_str)
        
        # Write shape
        buffer.write(struct.pack("<I", len(arr.shape)))
        for dim in arr.shape:
            buffer.write(struct.pack("<Q", dim))
        
        # Write data
        arr_bytes = arr.tobytes()
        buffer.write(struct.pack("<Q", len(arr_bytes)))
        buffer.write(arr_bytes)
        
        return buffer.getvalue()
    
    def _deserialize_numpy(self, data: bytes) -> np.ndarray:
        """Deserialize numpy array."""
        buffer = io.BytesIO(data)
        
        # Read dtype
        dtype_len = struct.unpack("<I", buffer.read(4))[0]
        dtype_str = buffer.read(dtype_len).decode()
        dtype = np.dtype(dtype_str)
        
        # Read shape
        ndim = struct.unpack("<I", buffer.read(4))[0]
        shape = tuple(struct.unpack("<Q", buffer.read(8))[0] for _ in range(ndim))
        
        # Read data
        data_len = struct.unpack("<Q", buffer.read(8))[0]
        arr_bytes = buffer.read(data_len)
        
        return np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
    
    def _serialize_dict(self, d: Dict) -> bytes:
        """Serialize dictionary."""
        buffer = io.BytesIO()
        buffer.write(bytes([self.TYPE_DICT]))
        
        # Write number of items
        buffer.write(struct.pack("<I", len(d)))
        
        for key, value in d.items():
            # Serialize key
            key_bytes = self._serialize_key(key)
            buffer.write(struct.pack("<I", len(key_bytes)))
            buffer.write(key_bytes)
            
            # Serialize value
            value_bytes = self._serialize_internal(value)
            buffer.write(struct.pack("<I", len(value_bytes)))
            buffer.write(value_bytes)
        
        return buffer.getvalue()
    
    def _deserialize_dict(self, data: bytes) -> Dict:
        """Deserialize dictionary."""
        buffer = io.BytesIO(data)
        
        num_items = struct.unpack("<I", buffer.read(4))[0]
        result = {}
        
        for _ in range(num_items):
            # Read key
            key_len = struct.unpack("<I", buffer.read(4))[0]
            key_bytes = buffer.read(key_len)
            key = self._deserialize_key(key_bytes)
            
            # Read value
            value_len = struct.unpack("<I", buffer.read(4))[0]
            value_bytes = buffer.read(value_len)
            value = self._deserialize_internal(value_bytes)
            
            result[key] = value
        
        return result
    
    def _serialize_list(self, lst: Union[List, Tuple]) -> bytes:
        """Serialize list or tuple."""
        buffer = io.BytesIO()
        buffer.write(bytes([self.TYPE_LIST]))
        
        # Write number of items
        buffer.write(struct.pack("<I", len(lst)))
        
        for item in lst:
            item_bytes = self._serialize_internal(item)
            buffer.write(struct.pack("<I", len(item_bytes)))
            buffer.write(item_bytes)
        
        return buffer.getvalue()
    
    def _deserialize_list(self, data: bytes) -> List:
        """Deserialize list."""
        buffer = io.BytesIO(data)
        
        num_items = struct.unpack("<I", buffer.read(4))[0]
        result = []
        
        for _ in range(num_items):
            item_len = struct.unpack("<I", buffer.read(4))[0]
            item_bytes = buffer.read(item_len)
            result.append(self._deserialize_internal(item_bytes))
        
        return result
    
    def _serialize_primitive(self, value: Any) -> bytes:
        """Serialize primitive value using msgpack or pickle."""
        buffer = io.BytesIO()
        buffer.write(bytes([self.TYPE_PRIMITIVE]))
        
        if HAS_MSGPACK:
            packed = msgpack.packb(value, use_bin_type=True)
        else:
            import pickle
            packed = pickle.dumps(value)
        
        buffer.write(packed)
        return buffer.getvalue()
    
    def _deserialize_primitive(self, data: bytes) -> Any:
        """Deserialize primitive value."""
        if HAS_MSGPACK:
            return msgpack.unpackb(data, raw=False)
        else:
            import pickle
            return pickle.loads(data)
    
    def _serialize_key(self, key: Any) -> bytes:
        """Serialize dictionary key."""
        if isinstance(key, str):
            return b's' + key.encode()
        elif isinstance(key, int):
            return b'i' + struct.pack("<q", key)
        else:
            return b'p' + self._serialize_primitive(key)[1:]
    
    def _deserialize_key(self, data: bytes) -> Any:
        """Deserialize dictionary key."""
        type_flag = chr(data[0])
        payload = data[1:]
        
        if type_flag == 's':
            return payload.decode()
        elif type_flag == 'i':
            return struct.unpack("<q", payload)[0]
        else:
            return self._deserialize_primitive(payload)
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.compression_algorithm == "lz4" and HAS_LZ4:
            compressed = lz4.compress(data, compression_level=self.compression_level)
            return b'\x01' + compressed  # 1 = lz4
        elif self.compression_algorithm == "zstd" and HAS_ZSTD:
            compressed = zstd.compress(data, self.compression_level)
            return b'\x02' + compressed  # 2 = zstd
        else:
            return b'\x00' + data  # 0 = uncompressed
    
    def _decompress(self, data: bytes, algorithm: int) -> bytes:
        """Decompress data."""
        if algorithm == 1 and HAS_LZ4:
            return lz4.decompress(data)
        elif algorithm == 2 and HAS_ZSTD:
            return zstd.decompress(data)
        else:
            return data
    
    def serialize_embedding_batch(
        self,
        token_ids: List[int],
        embeddings: np.ndarray
    ) -> bytes:
        """
        Serialize a batch of embeddings efficiently.
        
        Args:
            token_ids: List of token IDs
            embeddings: 2D array of shape (len(token_ids), embedding_dim)
            
        Returns:
            Serialized bytes
        """
        buffer = io.BytesIO()
        buffer.write(bytes([self.TYPE_EMBEDDING_BATCH]))
        
        # Write token IDs as array
        token_arr = np.array(token_ids, dtype=np.int64)
        token_bytes = token_arr.tobytes()
        buffer.write(struct.pack("<I", len(token_ids)))
        buffer.write(token_bytes)
        
        # Write embeddings
        emb_bytes = self._serialize_numpy(embeddings)[1:]  # Skip type marker
        buffer.write(emb_bytes)
        
        raw_bytes = buffer.getvalue()
        
        if self.compression and len(raw_bytes) > 1024:
            return self._compress(raw_bytes)
        return b'\x00' + raw_bytes
    
    def _deserialize_embedding_batch(self, data: bytes) -> Tuple[List[int], np.ndarray]:
        """Deserialize embedding batch."""
        buffer = io.BytesIO(data)
        
        # Read token IDs
        num_tokens = struct.unpack("<I", buffer.read(4))[0]
        token_bytes = buffer.read(num_tokens * 8)
        token_ids = np.frombuffer(token_bytes, dtype=np.int64).tolist()
        
        # Read embeddings
        embeddings = self._deserialize_numpy(buffer.read())
        
        return token_ids, embeddings
    
    def serialize_gradient_batch(
        self,
        embedding_grads: Dict[int, np.ndarray]
    ) -> bytes:
        """
        Serialize a batch of embedding gradients efficiently.
        
        Uses sparse representation - only non-zero gradients.
        """
        buffer = io.BytesIO()
        buffer.write(bytes([self.TYPE_GRADIENT_BATCH]))
        
        # Number of gradients
        buffer.write(struct.pack("<I", len(embedding_grads)))
        
        if embedding_grads:
            # Get embedding dimension from first gradient
            first_grad = next(iter(embedding_grads.values()))
            emb_dim = first_grad.shape[0]
            dtype = first_grad.dtype
            
            buffer.write(struct.pack("<I", emb_dim))
            dtype_str = str(dtype).encode()
            buffer.write(struct.pack("<I", len(dtype_str)))
            buffer.write(dtype_str)
            
            # Write token IDs
            token_ids = np.array(list(embedding_grads.keys()), dtype=np.int64)
            buffer.write(token_ids.tobytes())
            
            # Stack and write gradients
            grads = np.stack([embedding_grads[tid] for tid in token_ids])
            buffer.write(grads.tobytes())
        
        raw_bytes = buffer.getvalue()
        
        if self.compression and len(raw_bytes) > 1024:
            return self._compress(raw_bytes)
        return b'\x00' + raw_bytes
    
    def _deserialize_gradient_batch(self, data: bytes) -> Dict[int, np.ndarray]:
        """Deserialize gradient batch."""
        buffer = io.BytesIO(data)
        
        num_grads = struct.unpack("<I", buffer.read(4))[0]
        
        if num_grads == 0:
            return {}
        
        emb_dim = struct.unpack("<I", buffer.read(4))[0]
        dtype_len = struct.unpack("<I", buffer.read(4))[0]
        dtype = np.dtype(buffer.read(dtype_len).decode())
        
        # Read token IDs
        token_bytes = buffer.read(num_grads * 8)
        token_ids = np.frombuffer(token_bytes, dtype=np.int64)
        
        # Read gradients
        grad_bytes = buffer.read()
        grads = np.frombuffer(grad_bytes, dtype=dtype).reshape(num_grads, emb_dim)
        
        return {int(tid): grads[i] for i, tid in enumerate(token_ids)}

