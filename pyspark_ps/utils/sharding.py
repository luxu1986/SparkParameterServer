"""Consistent hashing for token-to-server mapping."""

import hashlib
from typing import List, Dict, Tuple
from bisect import bisect_right


class ConsistentHashRing:
    """
    Consistent hashing ring for distributing tokens across servers.
    
    Uses virtual nodes to ensure balanced distribution and minimal
    redistribution when servers are added/removed.
    """
    
    def __init__(self, num_servers: int, virtual_nodes: int = 150):
        """
        Initialize the hash ring.
        
        Args:
            num_servers: Number of physical server nodes
            virtual_nodes: Number of virtual nodes per server (higher = more balanced)
        """
        if num_servers < 1:
            raise ValueError("num_servers must be at least 1")
        
        self.num_servers = num_servers
        self.virtual_nodes = virtual_nodes
        self._ring: List[Tuple[int, int]] = []  # (hash_value, server_id)
        self._sorted_hashes: List[int] = []
        
        self._build_ring()
    
    def _hash(self, key: str) -> int:
        """Generate a consistent hash value for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _build_ring(self) -> None:
        """Build the hash ring with virtual nodes."""
        self._ring = []
        
        for server_id in range(self.num_servers):
            for vn in range(self.virtual_nodes):
                key = f"server_{server_id}_vn_{vn}"
                hash_value = self._hash(key)
                self._ring.append((hash_value, server_id))
        
        # Sort by hash value for binary search
        self._ring.sort(key=lambda x: x[0])
        self._sorted_hashes = [h for h, _ in self._ring]
    
    def get_server(self, token_id: int) -> int:
        """
        Get the server index for a given token.
        
        Args:
            token_id: Integer token ID
            
        Returns:
            Server index (0 to num_servers-1)
        """
        if not self._ring:
            raise RuntimeError("Hash ring is empty")
        
        # Hash the token ID
        hash_value = self._hash(str(token_id))
        
        # Find the first server with hash >= token hash
        idx = bisect_right(self._sorted_hashes, hash_value)
        
        # Wrap around if necessary
        if idx >= len(self._ring):
            idx = 0
        
        return self._ring[idx][1]
    
    def get_servers_batch(self, token_ids: List[int]) -> Dict[int, List[int]]:
        """
        Batch lookup of server assignments for multiple tokens.
        
        Args:
            token_ids: List of integer token IDs
            
        Returns:
            Dict mapping server_id -> list of token IDs assigned to that server
        """
        result: Dict[int, List[int]] = {i: [] for i in range(self.num_servers)}
        
        for token_id in token_ids:
            server_id = self.get_server(token_id)
            result[server_id].append(token_id)
        
        return result
    
    def get_server_for_weight(self, layer_name: str, chunk_idx: int = 0) -> int:
        """
        Get the server for a weight tensor or chunk.
        
        Args:
            layer_name: Name of the weight layer
            chunk_idx: Chunk index for sharded weights
            
        Returns:
            Server index
        """
        key = f"weight_{layer_name}_{chunk_idx}"
        hash_value = self._hash(key)
        idx = bisect_right(self._sorted_hashes, hash_value)
        if idx >= len(self._ring):
            idx = 0
        return self._ring[idx][1]
    
    def get_all_servers_for_weight(self, layer_name: str) -> List[int]:
        """
        Get all servers that should store parts of a weight tensor.
        
        For replicated weights, returns all servers.
        For sharded weights, returns assigned servers.
        
        Args:
            layer_name: Name of the weight layer
            
        Returns:
            List of server indices
        """
        # By default, weights are replicated across all servers
        return list(range(self.num_servers))
    
    def rebalance(self, new_num_servers: int) -> Dict[int, List[int]]:
        """
        Calculate token redistribution for a new number of servers.
        
        Args:
            new_num_servers: New number of servers
            
        Returns:
            Dict mapping old_server_id -> list of tokens that need to move
            
        Note: This is for planning purposes; actual redistribution must
        be coordinated across the cluster.
        """
        old_ring = ConsistentHashRing(self.num_servers, self.virtual_nodes)
        new_ring = ConsistentHashRing(new_num_servers, self.virtual_nodes)
        
        # This would need actual token tracking, which we don't have here
        # Return empty dict as placeholder
        return {}


class ModuloSharding:
    """
    Simple modulo-based sharding for predictable distribution.
    
    Faster than consistent hashing but requires redistribution when
    the number of servers changes.
    """
    
    def __init__(self, num_servers: int):
        self.num_servers = num_servers
    
    def get_server(self, token_id: int) -> int:
        """Get server for a token using simple modulo."""
        return token_id % self.num_servers
    
    def get_servers_batch(self, token_ids: List[int]) -> Dict[int, List[int]]:
        """Batch lookup using modulo sharding."""
        result: Dict[int, List[int]] = {i: [] for i in range(self.num_servers)}
        for token_id in token_ids:
            result[token_id % self.num_servers].append(token_id)
        return result

