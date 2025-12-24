"""RPC communication layer for Parameter Server."""

import socket
import struct
import threading
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import selectors

from pyspark_ps.communication.protocol import PSMessage, MessageType
from pyspark_ps.communication.serialization import Serializer
from pyspark_ps.utils.logging import get_logger


class RPCServer:
    """
    High-performance RPC server for PS nodes.
    
    Features:
    - Async request handling with thread pool
    - Connection management
    - Graceful shutdown
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int = 4,
        max_message_size: int = 100 * 1024 * 1024
    ):
        """
        Initialize RPC server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            num_workers: Number of worker threads
            max_message_size: Maximum message size in bytes
        """
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.max_message_size = max_message_size
        
        self.logger = get_logger(f"rpc_server:{port}")
        self.serializer = Serializer()
        
        self._handlers: Dict[MessageType, Callable] = {}
        self._socket: Optional[socket.socket] = None
        self._selector: Optional[selectors.DefaultSelector] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._connections: Dict[socket.socket, Dict] = {}
        self._lock = threading.Lock()
    
    def register_handler(self, msg_type: MessageType, handler: Callable):
        """
        Register a handler for a message type.
        
        Args:
            msg_type: Message type to handle
            handler: Function(PSMessage) -> PSMessage
        """
        self._handlers[msg_type] = handler
    
    def start(self):
        """Start the RPC server."""
        if self._running:
            return
        
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(128)
        self._socket.setblocking(False)
        
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._socket, selectors.EVENT_READ, data=None)
        
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._running = True
        
        self._server_thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._server_thread.start()
        
        self.logger.info(f"RPC server started on {self.host}:{self.port}")
    
    def stop(self, timeout: float = 5.0):
        """Stop the RPC server."""
        self._running = False
        
        if self._server_thread:
            self._server_thread.join(timeout=timeout)
        
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
        
        with self._lock:
            for conn in list(self._connections.keys()):
                try:
                    self._selector.unregister(conn)
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
        
        if self._socket:
            try:
                self._selector.unregister(self._socket)
                self._socket.close()
            except Exception:
                pass
        
        if self._selector:
            self._selector.close()
        
        self.logger.info("RPC server stopped")
    
    def _serve_loop(self):
        """Main server loop."""
        while self._running:
            try:
                events = self._selector.select(timeout=0.1)
                
                for key, mask in events:
                    if key.data is None:
                        # Accept new connection
                        self._accept_connection(key.fileobj)
                    else:
                        # Handle existing connection
                        self._handle_connection(key, mask)
                        
            except Exception as e:
                if self._running:
                    self.logger.error(f"Server loop error: {e}")
    
    def _accept_connection(self, sock: socket.socket):
        """Accept a new connection."""
        try:
            conn, addr = sock.accept()
            conn.setblocking(False)
            
            with self._lock:
                self._connections[conn] = {
                    "addr": addr,
                    "buffer": b"",
                    "pending_size": None,
                }
            
            self._selector.register(
                conn,
                selectors.EVENT_READ,
                data={"addr": addr}
            )
            
            self.logger.debug(f"Accepted connection from {addr}")
            
        except Exception as e:
            self.logger.error(f"Accept error: {e}")
    
    def _handle_connection(self, key: selectors.SelectorKey, mask: int):
        """Handle data on a connection."""
        conn = key.fileobj
        
        if mask & selectors.EVENT_READ:
            try:
                data = conn.recv(65536)
                
                if not data:
                    # Connection closed
                    self._close_connection(conn)
                    return
                
                with self._lock:
                    conn_data = self._connections.get(conn)
                    if not conn_data:
                        return
                    
                    conn_data["buffer"] += data
                
                # Process complete messages
                self._process_buffer(conn)
                
            except ConnectionResetError:
                self._close_connection(conn)
            except Exception as e:
                self.logger.error(f"Read error: {e}")
                self._close_connection(conn)
    
    def _process_buffer(self, conn: socket.socket):
        """Process buffered data and extract complete messages."""
        with self._lock:
            conn_data = self._connections.get(conn)
            if not conn_data:
                return
            
            buffer = conn_data["buffer"]
            
            while True:
                # Need at least 4 bytes for message size
                if len(buffer) < 4:
                    break
                
                # Read message size
                msg_size = struct.unpack("<I", buffer[:4])[0]
                
                if msg_size > self.max_message_size:
                    self.logger.error(f"Message too large: {msg_size}")
                    self._close_connection(conn)
                    return
                
                # Check if we have the complete message
                if len(buffer) < 4 + msg_size:
                    break
                
                # Extract message
                msg_data = buffer[4:4 + msg_size]
                buffer = buffer[4 + msg_size:]
                
                # Submit for processing
                self._executor.submit(self._handle_message, conn, msg_data)
            
            conn_data["buffer"] = buffer
    
    def _handle_message(self, conn: socket.socket, msg_data: bytes):
        """Handle a complete message."""
        try:
            # Deserialize message
            msg_dict = self.serializer.deserialize(msg_data)
            message = PSMessage.from_dict(msg_dict)
            
            # Find handler
            handler = self._handlers.get(message.msg_type)
            
            if handler:
                response = handler(message)
            else:
                response = message.create_response(
                    MessageType.RESPONSE_ERROR,
                    payload=self.serializer.serialize({"error": "Unknown message type"})
                )
            
            # Send response
            self._send_response(conn, response)
            
        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
            try:
                error_response = PSMessage(
                    msg_type=MessageType.RESPONSE_ERROR,
                    client_id="server",
                    request_id="",
                    payload=self.serializer.serialize({"error": str(e)})
                )
                self._send_response(conn, error_response)
            except Exception:
                pass
    
    def _send_response(self, conn: socket.socket, response: PSMessage):
        """Send a response message."""
        try:
            response_data = self.serializer.serialize(response.to_dict())
            size_header = struct.pack("<I", len(response_data))
            
            with self._lock:
                conn.sendall(size_header + response_data)
                
        except Exception as e:
            self.logger.error(f"Send error: {e}")
    
    def _close_connection(self, conn: socket.socket):
        """Close a connection."""
        with self._lock:
            if conn in self._connections:
                try:
                    self._selector.unregister(conn)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
                del self._connections[conn]


class RPCClient:
    """
    RPC client for communicating with PS servers.
    
    Features:
    - Connection pooling
    - Async requests with futures
    - Automatic reconnection
    - Request batching
    """
    
    def __init__(
        self,
        timeout: float = 60.0,
        max_message_size: int = 100 * 1024 * 1024,
        pool_size: int = 4
    ):
        """
        Initialize RPC client.
        
        Args:
            timeout: Request timeout in seconds
            max_message_size: Maximum message size
            pool_size: Connection pool size per server
        """
        self.timeout = timeout
        self.max_message_size = max_message_size
        self.pool_size = pool_size
        
        self.logger = get_logger("rpc_client")
        self.serializer = Serializer()
        
        self._pools: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._closed = False
    
    def _get_connection(self, host: str, port: int) -> socket.socket:
        """Get a connection from the pool or create a new one."""
        key = f"{host}:{port}"
        
        with self._lock:
            if key not in self._pools:
                self._pools[key] = queue.Queue(maxsize=self.pool_size)
        
        pool = self._pools[key]
        
        try:
            # Try to get an existing connection
            conn = pool.get_nowait()
            
            # Check if connection is still alive
            try:
                conn.setblocking(False)
                conn.recv(1, socket.MSG_PEEK)
                conn.setblocking(True)
                return conn
            except BlockingIOError:
                # No data available, connection is fine
                conn.setblocking(True)
                return conn
            except Exception:
                # Connection is dead, create a new one
                try:
                    conn.close()
                except Exception:
                    pass
                    
        except queue.Empty:
            pass
        
        # Create new connection
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(self.timeout)
        conn.connect((host, port))
        return conn
    
    def _return_connection(self, host: str, port: int, conn: socket.socket):
        """Return a connection to the pool."""
        key = f"{host}:{port}"
        
        with self._lock:
            pool = self._pools.get(key)
            if pool:
                try:
                    pool.put_nowait(conn)
                    return
                except queue.Full:
                    pass
        
        # Pool is full, close connection
        try:
            conn.close()
        except Exception:
            pass
    
    def call(
        self,
        host: str,
        port: int,
        message: PSMessage
    ) -> PSMessage:
        """
        Make a synchronous RPC call.
        
        Args:
            host: Server host
            port: Server port
            message: Request message
            
        Returns:
            Response message
        """
        if self._closed:
            raise RuntimeError("Client is closed")
        
        conn = None
        try:
            conn = self._get_connection(host, port)
            
            # Serialize and send
            msg_data = self.serializer.serialize(message.to_dict())
            size_header = struct.pack("<I", len(msg_data))
            conn.sendall(size_header + msg_data)
            
            # Receive response
            response_data = self._receive_message(conn)
            response_dict = self.serializer.deserialize(response_data)
            
            self._return_connection(host, port, conn)
            
            return PSMessage.from_dict(response_dict)
            
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            raise
    
    def call_async(
        self,
        host: str,
        port: int,
        message: PSMessage,
        executor: ThreadPoolExecutor
    ) -> Future:
        """
        Make an async RPC call.
        
        Args:
            host: Server host
            port: Server port
            message: Request message
            executor: Thread pool executor
            
        Returns:
            Future containing response message
        """
        return executor.submit(self.call, host, port, message)
    
    def call_batch(
        self,
        requests: List[Tuple[str, int, PSMessage]],
        executor: Optional[ThreadPoolExecutor] = None
    ) -> List[PSMessage]:
        """
        Make multiple RPC calls in parallel.
        
        Args:
            requests: List of (host, port, message) tuples
            executor: Optional executor (creates one if not provided)
            
        Returns:
            List of response messages in same order
        """
        if not requests:
            return []
        
        own_executor = executor is None
        if own_executor:
            executor = ThreadPoolExecutor(max_workers=len(requests))
        
        try:
            futures = [
                self.call_async(host, port, msg, executor)
                for host, port, msg in requests
            ]
            
            return [f.result(timeout=self.timeout) for f in futures]
            
        finally:
            if own_executor:
                executor.shutdown(wait=True)
    
    def _receive_message(self, conn: socket.socket) -> bytes:
        """Receive a complete message from connection."""
        # Read size header
        size_data = self._recv_exact(conn, 4)
        msg_size = struct.unpack("<I", size_data)[0]
        
        if msg_size > self.max_message_size:
            raise ValueError(f"Message too large: {msg_size}")
        
        # Read message data
        return self._recv_exact(conn, msg_size)
    
    def _recv_exact(self, conn: socket.socket, size: int) -> bytes:
        """Receive exactly size bytes."""
        data = b""
        while len(data) < size:
            chunk = conn.recv(min(size - len(data), 65536))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data
    
    def close(self):
        """Close all connections."""
        self._closed = True
        
        with self._lock:
            for key, pool in self._pools.items():
                while True:
                    try:
                        conn = pool.get_nowait()
                        try:
                            conn.close()
                        except Exception:
                            pass
                    except queue.Empty:
                        break
            self._pools.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class MultiServerClient:
    """
    Client for communicating with multiple PS servers.
    
    Handles routing requests to appropriate servers based on sharding.
    """
    
    def __init__(
        self,
        servers: List[Tuple[str, int]],
        timeout: float = 60.0
    ):
        """
        Initialize multi-server client.
        
        Args:
            servers: List of (host, port) tuples
            timeout: Request timeout
        """
        self.servers = servers
        self.client = RPCClient(timeout=timeout)
        self._executor = ThreadPoolExecutor(max_workers=len(servers) * 2)
    
    def broadcast(self, message: PSMessage) -> List[PSMessage]:
        """Send message to all servers."""
        requests = [
            (host, port, message)
            for host, port in self.servers
        ]
        return self.client.call_batch(requests, self._executor)
    
    def send_to_server(self, server_id: int, message: PSMessage) -> PSMessage:
        """Send message to specific server."""
        host, port = self.servers[server_id]
        return self.client.call(host, port, message)
    
    def send_to_servers(
        self,
        server_messages: Dict[int, PSMessage]
    ) -> Dict[int, PSMessage]:
        """Send different messages to specific servers."""
        requests = [
            (self.servers[sid][0], self.servers[sid][1], msg)
            for sid, msg in server_messages.items()
        ]
        
        responses = self.client.call_batch(requests, self._executor)
        
        return {
            sid: resp
            for sid, resp in zip(server_messages.keys(), responses)
        }
    
    def close(self):
        """Close client resources."""
        self._executor.shutdown(wait=True)
        self.client.close()

