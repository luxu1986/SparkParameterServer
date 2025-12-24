"""Barrier synchronization for distributed training."""

import threading
import time
from typing import Dict, List, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor

from pyspark_ps.communication.protocol import MessageType, PSMessage, ServerInfo
from pyspark_ps.communication.rpc_handler import RPCClient
from pyspark_ps.communication.serialization import Serializer
from pyspark_ps.utils.logging import get_logger


class BarrierCoordinator:
    """
    Coordinates barrier synchronization across distributed workers.
    
    Uses the PS servers as coordination points. All servers maintain
    barrier state, and clients can check any server for status.
    """
    
    def __init__(
        self,
        servers: List[ServerInfo],
        timeout_seconds: float = 300.0
    ):
        """
        Initialize barrier coordinator.
        
        Args:
            servers: List of PS server info
            timeout_seconds: Default timeout for barrier operations
        """
        self.servers = servers
        self.timeout_seconds = timeout_seconds
        
        self.logger = get_logger("barrier_coordinator")
        self.serializer = Serializer()
        self._client = RPCClient(timeout=timeout_seconds)
        
        # Use first server as coordinator
        self._coordinator = servers[0]
    
    def create(self, name: str, num_workers: int):
        """
        Create a named barrier.
        
        Must be called before workers attempt to enter the barrier.
        
        Args:
            name: Barrier name
            num_workers: Number of workers that must enter
        """
        message = PSMessage(
            msg_type=MessageType.BARRIER_CREATE,
            client_id="coordinator",
            payload=self.serializer.serialize({
                "name": name,
                "num_workers": num_workers,
            })
        )
        
        response = self._client.call(
            self._coordinator.host,
            self._coordinator.port,
            message
        )
        
        if response.msg_type == MessageType.RESPONSE_ERROR:
            error = self.serializer.deserialize(response.payload)
            raise RuntimeError(f"Failed to create barrier: {error}")
        
        self.logger.debug(f"Created barrier '{name}' for {num_workers} workers")
    
    def enter(
        self,
        name: str,
        client_id: str,
        timeout_seconds: Optional[float] = None
    ) -> bool:
        """
        Enter a barrier and wait for release.
        
        Blocks until the barrier is released by the coordinator.
        
        Args:
            name: Barrier name
            client_id: Unique identifier for this client
            timeout_seconds: Timeout (uses default if not specified)
            
        Returns:
            True if barrier was released, False if timeout
        """
        timeout = timeout_seconds or self.timeout_seconds
        start_time = time.time()
        
        # First, register entry
        message = PSMessage(
            msg_type=MessageType.BARRIER_ENTER,
            client_id=client_id,
            payload=self.serializer.serialize({"name": name})
        )
        
        response = self._client.call(
            self._coordinator.host,
            self._coordinator.port,
            message
        )
        
        if response.msg_type == MessageType.RESPONSE_ERROR:
            error = self.serializer.deserialize(response.payload)
            raise RuntimeError(f"Failed to enter barrier: {error}")
        
        # Poll for release
        poll_interval = 0.1
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self.logger.warning(f"Barrier '{name}' timeout after {elapsed:.1f}s")
                return False
            
            status = self._get_status(name)
            
            if status.get("released", False):
                self.logger.debug(f"Barrier '{name}' released for client {client_id}")
                return True
            
            time.sleep(min(poll_interval, timeout - elapsed))
            poll_interval = min(poll_interval * 1.5, 1.0)  # Exponential backoff
    
    def wait(self, name: str, timeout_seconds: Optional[float] = None) -> bool:
        """
        Wait for all workers to enter the barrier.
        
        Called by the coordinator (main client) to wait for all workers.
        
        Args:
            name: Barrier name
            timeout_seconds: Timeout
            
        Returns:
            True if all workers entered, False if timeout
        """
        timeout = timeout_seconds or self.timeout_seconds
        start_time = time.time()
        poll_interval = 0.1
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self.logger.warning(f"Barrier wait '{name}' timeout after {elapsed:.1f}s")
                return False
            
            status = self._get_status(name)
            
            if not status.get("exists", False):
                raise RuntimeError(f"Barrier '{name}' does not exist")
            
            if status.get("num_entered", 0) >= status.get("num_workers", 0):
                self.logger.debug(f"All workers entered barrier '{name}'")
                return True
            
            time.sleep(min(poll_interval, timeout - elapsed))
            poll_interval = min(poll_interval * 1.5, 1.0)
    
    def release(self, name: str):
        """
        Release all workers waiting at a barrier.
        
        Called by the coordinator after all workers have entered.
        
        Args:
            name: Barrier name
        """
        message = PSMessage(
            msg_type=MessageType.BARRIER_RELEASE,
            client_id="coordinator",
            payload=self.serializer.serialize({"name": name})
        )
        
        response = self._client.call(
            self._coordinator.host,
            self._coordinator.port,
            message
        )
        
        if response.msg_type == MessageType.RESPONSE_ERROR:
            error = self.serializer.deserialize(response.payload)
            raise RuntimeError(f"Failed to release barrier: {error}")
        
        self.logger.debug(f"Released barrier '{name}'")
    
    def _get_status(self, name: str) -> Dict[str, Any]:
        """Get barrier status from coordinator."""
        message = PSMessage(
            msg_type=MessageType.BARRIER_STATUS,
            client_id="coordinator",
            payload=self.serializer.serialize({"name": name})
        )
        
        response = self._client.call(
            self._coordinator.host,
            self._coordinator.port,
            message
        )
        
        if response.msg_type == MessageType.RESPONSE_ERROR:
            return {"exists": False}
        
        return self.serializer.deserialize(response.payload)
    
    def close(self):
        """Close coordinator resources."""
        self._client.close()


class LocalBarrier:
    """
    Local barrier for single-machine testing.
    
    Uses threading primitives instead of distributed coordination.
    """
    
    def __init__(self, name: str, num_workers: int):
        """
        Initialize local barrier.
        
        Args:
            name: Barrier name
            num_workers: Number of workers
        """
        self.name = name
        self.num_workers = num_workers
        
        self._barrier = threading.Barrier(num_workers)
        self._released = threading.Event()
        self._entered = set()
        self._lock = threading.Lock()
    
    def enter(self, client_id: str, timeout: float = 300.0) -> bool:
        """Enter the barrier and wait."""
        with self._lock:
            self._entered.add(client_id)
        
        try:
            self._barrier.wait(timeout=timeout)
            self._released.wait(timeout=timeout)
            return True
        except threading.BrokenBarrierError:
            return False
    
    def wait(self, timeout: float = 300.0) -> bool:
        """Wait for all workers to enter."""
        start = time.time()
        while len(self._entered) < self.num_workers:
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True
    
    def release(self):
        """Release waiting workers."""
        self._released.set()
    
    def reset(self):
        """Reset the barrier for reuse."""
        self._barrier.reset()
        self._released.clear()
        self._entered.clear()

