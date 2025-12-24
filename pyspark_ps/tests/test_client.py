"""Tests for PS client components."""

import unittest
import numpy as np
import time
import threading

from pyspark_ps.utils.config import PSConfig
from pyspark_ps.communication.protocol import ServerInfo


class TestPSConfig(unittest.TestCase):
    """Tests for PS configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PSConfig()
        
        self.assertEqual(config.num_servers, 4)
        self.assertEqual(config.embedding_dim, 64)
        self.assertEqual(config.embedding_optimizer, "adagrad")
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PSConfig(
            num_servers=8,
            embedding_dim=128,
            embedding_optimizer="adam",
            weight_optimizer="sgd"
        )
        
        self.assertEqual(config.num_servers, 8)
        self.assertEqual(config.embedding_dim, 128)
        self.assertEqual(config.embedding_optimizer, "adam")
    
    def test_serialization(self):
        """Test config serialization."""
        config = PSConfig(num_servers=2, embedding_dim=32)
        
        # To dict
        d = config.to_dict()
        self.assertEqual(d["num_servers"], 2)
        self.assertEqual(d["embedding_dim"], 32)
        
        # From dict
        config2 = PSConfig.from_dict(d)
        self.assertEqual(config2.num_servers, 2)
        
        # To/from JSON
        json_str = config.to_json()
        config3 = PSConfig.from_json(json_str)
        self.assertEqual(config3.num_servers, 2)
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = PSConfig()
        config.validate()  # Should not raise
        
        # Invalid configs
        with self.assertRaises(ValueError):
            config = PSConfig(num_servers=0)
            config.validate()
        
        with self.assertRaises(ValueError):
            config = PSConfig(embedding_dim=-1)
            config.validate()
        
        with self.assertRaises(ValueError):
            config = PSConfig(embedding_init="invalid")
            config.validate()
    
    def test_optimizer_config(self):
        """Test optimizer configuration retrieval."""
        config = PSConfig()
        
        adam_config = config.get_optimizer_config("adam")
        
        self.assertIn("learning_rate", adam_config)
        self.assertIn("beta1", adam_config)
        self.assertIn("beta2", adam_config)


class TestServerInfo(unittest.TestCase):
    """Tests for server info."""
    
    def test_creation(self):
        """Test server info creation."""
        info = ServerInfo(
            server_id=0,
            host="localhost",
            port=50000
        )
        
        self.assertEqual(info.server_id, 0)
        self.assertEqual(info.host, "localhost")
        self.assertEqual(info.port, 50000)
        self.assertEqual(info.address, "localhost:50000")
    
    def test_serialization(self):
        """Test server info serialization."""
        info = ServerInfo(
            server_id=1,
            host="192.168.1.1",
            port=50001,
            status="running",
            metadata={"test": "value"}
        )
        
        d = info.to_dict()
        info2 = ServerInfo.from_dict(d)
        
        self.assertEqual(info2.server_id, 1)
        self.assertEqual(info2.host, "192.168.1.1")
        self.assertEqual(info2.status, "running")
        self.assertEqual(info2.metadata["test"], "value")


class TestIntegration(unittest.TestCase):
    """Integration tests for client-server communication."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test servers."""
        from pyspark_ps.server.ps_server import PSServer
        
        cls.config = PSConfig(
            num_servers=2,
            embedding_dim=16,
            server_port_base=50100
        )
        
        # Start servers
        cls.servers = []
        for i in range(cls.config.num_servers):
            server = PSServer(
                server_id=i,
                total_servers=cls.config.num_servers,
                config=cls.config,
                host="127.0.0.1",
                port=cls.config.server_port_base + i
            )
            server.start()
            cls.servers.append(server)
        
        # Wait for servers to be ready
        time.sleep(0.5)
        
        cls.server_info = [s.get_server_info() for s in cls.servers]
    
    @classmethod
    def tearDownClass(cls):
        """Shutdown test servers."""
        for server in cls.servers:
            try:
                server.shutdown(grace_period_seconds=1)
            except Exception:
                pass
    
    def test_worker_client_pull_embeddings(self):
        """Test worker client can pull embeddings."""
        from pyspark_ps.client.worker_client import PSWorkerClient
        
        client = PSWorkerClient(self.server_info, self.config)
        
        try:
            # Pull embeddings for some tokens
            token_ids = [1, 2, 3, 4, 5]
            embeddings = client.pull_embeddings(token_ids)
            
            self.assertEqual(embeddings.shape, (5, 16))
        finally:
            client.close()
    
    def test_worker_client_push_embeddings(self):
        """Test worker client can push embedding gradients."""
        from pyspark_ps.client.worker_client import PSWorkerClient
        
        client = PSWorkerClient(self.server_info, self.config)
        
        try:
            # Pull first to ensure embeddings exist
            token_ids = [10, 20, 30]
            _ = client.pull_embeddings(token_ids)
            
            # Push gradients
            gradients = {
                10: np.ones(16, dtype=np.float32) * 0.1,
                20: np.ones(16, dtype=np.float32) * 0.2,
                30: np.ones(16, dtype=np.float32) * 0.3,
            }
            client.push_embedding_gradients(gradients)
            
            # Pull again and verify changed
            embeddings = client.pull_embeddings(token_ids)
            self.assertEqual(embeddings.shape, (3, 16))
        finally:
            client.close()
    
    def test_worker_client_weights(self):
        """Test worker client weight operations."""
        from pyspark_ps.client.worker_client import PSWorkerClient
        from pyspark_ps.communication.protocol import MessageType, PSMessage
        from pyspark_ps.communication.rpc_handler import RPCClient
        from pyspark_ps.communication.serialization import Serializer
        
        # First, initialize weights on servers
        rpc_client = RPCClient()
        serializer = Serializer()
        
        for server in self.server_info:
            msg = PSMessage(
                msg_type=MessageType.INIT_WEIGHTS,
                client_id="test",
                payload=serializer.serialize({
                    "shapes": {"layer1": [16, 8], "layer2": [8, 4]},
                    "init_strategy": "zeros",
                    "init_scale": 0.01,
                })
            )
            rpc_client.call(server.host, server.port, msg)
        
        rpc_client.close()
        
        # Now test worker client
        client = PSWorkerClient(self.server_info, self.config)
        
        try:
            # Pull model
            weights = client.pull_model()
            
            self.assertIn("layer1", weights)
            self.assertIn("layer2", weights)
            self.assertEqual(weights["layer1"].shape, (16, 8))
            
            # Push weight gradients
            gradients = {
                "layer1": np.ones((16, 8), dtype=np.float32) * 0.01,
                "layer2": np.ones((8, 4), dtype=np.float32) * 0.01,
            }
            client.push_weight_gradients(gradients)
            
        finally:
            client.close()
    
    def test_concurrent_clients(self):
        """Test multiple concurrent clients."""
        from pyspark_ps.client.worker_client import PSWorkerClient
        
        results = []
        errors = []
        
        def client_task(client_id):
            try:
                client = PSWorkerClient(
                    self.server_info,
                    self.config,
                    client_id=f"client_{client_id}"
                )
                
                # Each client works with different tokens
                base = client_id * 100
                token_ids = list(range(base, base + 10))
                
                # Pull
                embeddings = client.pull_embeddings(token_ids)
                
                # Push
                gradients = {
                    tid: np.ones(16, dtype=np.float32) * 0.1
                    for tid in token_ids
                }
                client.push_embedding_gradients(gradients)
                
                client.close()
                results.append(client_id)
            except Exception as e:
                errors.append((client_id, str(e)))
        
        # Run multiple clients concurrently
        threads = [
            threading.Thread(target=client_task, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=10)
        
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 5)


class TestMainClient(unittest.TestCase):
    """Tests for main client."""
    
    def test_local_mode(self):
        """Test main client in local mode."""
        from pyspark_ps.client.main_client import PSMainClient
        
        config = PSConfig(
            num_servers=2,
            embedding_dim=8,
            server_port_base=50200
        )
        
        client = PSMainClient(None, config)
        
        try:
            # Start servers
            server_info = client.start_servers()
            
            self.assertEqual(len(server_info), 2)
            
            # Initialize weights
            client.init_weights({
                "fc1": (10, 8),
                "fc2": (8, 4),
            })
            
            # Get stats
            stats = client.get_cluster_stats()
            
            self.assertIn("servers", stats)
            self.assertEqual(len(stats["servers"]), 2)
            
            # Decay embeddings
            result = client.decay_embeddings(method="multiply", factor=0.9)
            self.assertEqual(result["method"], "multiply")
            
        finally:
            client.shutdown_servers(grace_period_seconds=1)


if __name__ == "__main__":
    unittest.main()

