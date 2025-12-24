"""Message protocol definitions for Parameter Server communication."""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


class MessageType(IntEnum):
    """Message types for PS protocol."""
    
    # Pull operations
    PULL_MODEL = 1
    PULL_EMBEDDINGS = 2
    PULL_MODEL_PARTIAL = 3
    
    # Push operations
    PUSH_WEIGHT_GRADS = 10
    PUSH_EMBEDDING_GRADS = 11
    PUSH_GRADIENTS = 12
    
    # Control operations
    BARRIER_ENTER = 20
    BARRIER_RELEASE = 21
    BARRIER_CREATE = 22
    BARRIER_STATUS = 23
    
    # Decay operations
    DECAY_EMBEDDINGS = 30
    DECAY_MULTIPLY = 31
    DECAY_PRUNE = 32
    
    # Lifecycle operations
    SHUTDOWN = 40
    PING = 41
    HEARTBEAT = 42
    
    # Initialization
    INIT_EMBEDDINGS = 50
    INIT_WEIGHTS = 51
    REGISTER_CLIENT = 52
    
    # Checkpointing
    SAVE_CHECKPOINT = 60
    LOAD_CHECKPOINT = 61
    GET_STATS = 62
    
    # Responses
    RESPONSE_OK = 100
    RESPONSE_ERROR = 101
    RESPONSE_DATA = 102
    RESPONSE_ACK = 103


@dataclass
class PSMessage:
    """
    Wire protocol message format.
    
    All communication between clients and servers uses this format.
    """
    
    msg_type: MessageType
    client_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: bytes = b""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "msg_type": int(self.msg_type),
            "client_id": self.client_id,
            "request_id": self.request_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PSMessage":
        """Create message from dictionary."""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            client_id=data["client_id"],
            request_id=data["request_id"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )
    
    def create_response(
        self,
        msg_type: MessageType = MessageType.RESPONSE_OK,
        payload: bytes = b"",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "PSMessage":
        """Create a response message for this request."""
        return PSMessage(
            msg_type=msg_type,
            client_id="server",
            request_id=self.request_id,
            payload=payload,
            metadata=metadata or {},
        )


@dataclass
class ServerInfo:
    """Information about a PS server node."""
    
    server_id: int
    host: str
    port: int
    status: str = "unknown"
    shard_range: tuple = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        """Get server address string."""
        return f"{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_id": self.server_id,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "shard_range": self.shard_range,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerInfo":
        """Create from dictionary."""
        return cls(
            server_id=data["server_id"],
            host=data["host"],
            port=data["port"],
            status=data.get("status", "unknown"),
            shard_range=tuple(data.get("shard_range", (0, 0))),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    
    s3_path: str
    timestamp: float
    embedding_count: int
    model_size_bytes: int
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "s3_path": self.s3_path,
            "timestamp": self.timestamp,
            "embedding_count": self.embedding_count,
            "model_size_bytes": self.model_size_bytes,
            "config": self.config,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        """Create from dictionary."""
        return cls(
            s3_path=data["s3_path"],
            timestamp=data["timestamp"],
            embedding_count=data["embedding_count"],
            model_size_bytes=data["model_size_bytes"],
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GradientBatch:
    """Batch of gradients for efficient transfer."""
    
    weight_gradients: Dict[str, Any]  # layer_name -> gradient array
    embedding_gradients: Dict[int, Any]  # token_id -> gradient array
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return not self.weight_gradients and not self.embedding_gradients


@dataclass
class PullRequest:
    """Request to pull parameters."""
    
    layer_names: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    include_optimizer_state: bool = False


@dataclass
class DecayRequest:
    """Request for embedding decay operation."""
    
    method: str  # "multiply" or "prune"
    factor: float = 0.99
    min_count: int = 5
    target_tables: List[str] = field(default_factory=list)  # Empty means all

