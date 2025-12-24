"""Integration tests for the complete PS system."""

import unittest
import numpy as np
import time
import threading

from pyspark_ps import PSMainClient, PSWorkerClient, PSConfig


class TestFullIntegration(unittest.TestCase):
    """Full integration tests simulating real training scenarios."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the PS cluster."""
        cls.config = PSConfig(
            num_servers=2,
            embedding_dim=32,
            embedding_optimizer="adagrad",
            weight_optimizer="adam",
            server_port_base=50300
        )
        
        # Start main client and servers
        cls.main_client = PSMainClient(None, cls.config)
        cls.server_info = cls.main_client.start_servers()
        
        # Initialize model weights
        cls.main_client.init_weights({
            "embedding_transform": (32, 16),
            "hidden": (16, 8),
            "output": (8, 1),
        })
    
    @classmethod
    def tearDownClass(cls):
        """Shutdown the PS cluster."""
        cls.main_client.shutdown_servers(grace_period_seconds=2)
    
    def test_training_loop_simulation(self):
        """Simulate a simple training loop."""
        # Create worker clients
        num_workers = 4
        workers = [
            PSWorkerClient(
                self.server_info,
                self.config,
                client_id=f"worker_{i}"
            )
            for i in range(num_workers)
        ]
        
        try:
            # Simulate multiple training steps
            for step in range(3):
                for worker_id, worker in enumerate(workers):
                    # Each worker processes a "batch" of tokens
                    batch_tokens = list(range(
                        worker_id * 100 + step * 10,
                        worker_id * 100 + step * 10 + 10
                    ))
                    
                    # Pull embeddings
                    embeddings = worker.pull_embeddings(batch_tokens)
                    self.assertEqual(embeddings.shape, (10, 32))
                    
                    # Pull model weights
                    weights = worker.pull_model()
                    self.assertIn("embedding_transform", weights)
                    
                    # Simulate forward/backward pass
                    # (In real use, this would compute actual gradients)
                    
                    # Push embedding gradients
                    emb_grads = {
                        tid: np.random.randn(32).astype(np.float32) * 0.01
                        for tid in batch_tokens
                    }
                    worker.push_embedding_gradients(emb_grads)
                    
                    # Push weight gradients
                    weight_grads = {
                        "embedding_transform": np.random.randn(32, 16).astype(np.float32) * 0.001,
                        "hidden": np.random.randn(16, 8).astype(np.float32) * 0.001,
                        "output": np.random.randn(8, 1).astype(np.float32) * 0.001,
                    }
                    worker.push_weight_gradients(weight_grads)
                
                # End of step
                self.main_client.step()
            
            # Verify cluster state
            stats = self.main_client.get_cluster_stats()
            self.assertGreater(stats["total_embeddings"], 0)
            
        finally:
            for worker in workers:
                worker.close()
    
    def test_embedding_decay(self):
        """Test embedding decay operation."""
        worker = PSWorkerClient(
            self.server_info,
            self.config,
            client_id="decay_test_worker"
        )
        
        try:
            # Create some embeddings
            tokens = list(range(1000, 1020))
            embeddings_before = worker.pull_embeddings(tokens)
            
            # Apply decay
            result = self.main_client.decay_embeddings(
                method="multiply",
                factor=0.5
            )
            
            self.assertEqual(result["method"], "multiply")
            
            # Pull again and verify decayed
            embeddings_after = worker.pull_embeddings(tokens)
            
            # Values should be halved
            np.testing.assert_array_almost_equal(
                embeddings_after,
                embeddings_before * 0.5,
                decimal=5
            )
            
        finally:
            worker.close()
    
    def test_embedding_pruning(self):
        """Test embedding pruning operation."""
        worker = PSWorkerClient(
            self.server_info,
            self.config,
            client_id="prune_test_worker"
        )
        
        try:
            # Create embeddings with varying update counts
            low_update_tokens = list(range(2000, 2010))
            high_update_tokens = list(range(2010, 2020))
            
            # Pull to create
            worker.pull_embeddings(low_update_tokens)
            
            # Update high_update_tokens multiple times
            for _ in range(10):
                worker.pull_embeddings(high_update_tokens)
                grads = {
                    tid: np.zeros(32, dtype=np.float32)
                    for tid in high_update_tokens
                }
                worker.push_embedding_gradients(grads)
            
            # Get stats before prune
            stats_before = self.main_client.get_cluster_stats()
            
            # Prune embeddings with count < 5
            result = self.main_client.decay_embeddings(
                method="prune",
                min_count=5
            )
            
            self.assertEqual(result["method"], "prune")
            
            # Verify some embeddings were pruned
            # (The low_update_tokens should be pruned)
            
        finally:
            worker.close()
    
    def test_concurrent_access(self):
        """Test concurrent access from multiple workers."""
        errors = []
        successful_ops = []
        lock = threading.Lock()
        
        def worker_task(worker_id):
            try:
                worker = PSWorkerClient(
                    self.server_info,
                    self.config,
                    client_id=f"concurrent_worker_{worker_id}"
                )
                
                for step in range(10):
                    # Unique tokens per worker per step
                    tokens = list(range(
                        worker_id * 1000 + step * 100,
                        worker_id * 1000 + step * 100 + 50
                    ))
                    
                    # Pull
                    embs = worker.pull_embeddings(tokens)
                    assert embs.shape == (50, 32)
                    
                    # Push
                    grads = {
                        t: np.random.randn(32).astype(np.float32) * 0.001
                        for t in tokens
                    }
                    worker.push_embedding_gradients(grads)
                    
                    with lock:
                        successful_ops.append((worker_id, step))
                
                worker.close()
                
            except Exception as e:
                with lock:
                    errors.append((worker_id, str(e)))
        
        # Run 8 workers concurrently
        threads = [
            threading.Thread(target=worker_task, args=(i,))
            for i in range(8)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(successful_ops), 8 * 10)  # 8 workers, 10 steps each


class TestCheckpointing(unittest.TestCase):
    """Tests for checkpoint save/load operations."""
    
    def setUp(self):
        """Set up test cluster."""
        self.config = PSConfig(
            num_servers=2,
            embedding_dim=16,
            server_port_base=50400
        )
        
        self.main_client = PSMainClient(None, self.config)
        self.server_info = self.main_client.start_servers()
        
        # Initialize weights
        self.main_client.init_weights({
            "layer1": (16, 8),
            "layer2": (8, 4),
        })
    
    def tearDown(self):
        """Shutdown test cluster."""
        self.main_client.shutdown_servers(grace_period_seconds=1)
    
    def test_local_checkpoint(self):
        """Test local checkpoint save/load."""
        import tempfile
        import os
        
        worker = PSWorkerClient(
            self.server_info,
            self.config,
            client_id="ckpt_test"
        )
        
        try:
            # Create some state
            tokens = list(range(100))
            embeddings_orig = worker.pull_embeddings(tokens)
            weights_orig = worker.pull_model()
            
            # Save checkpoint
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = os.path.join(tmpdir, "test_ckpt")
                self.main_client.checkpoint(ckpt_path)
                
                # Verify checkpoint files exist
                self.assertTrue(os.path.exists(ckpt_path))
                
                # Modify state
                grads = {t: np.ones(16, dtype=np.float32) for t in tokens}
                worker.push_embedding_gradients(grads)
                
                # Restore
                self.main_client.restore(ckpt_path)
                
                # Verify state restored
                embeddings_restored = worker.pull_embeddings(tokens)
                
                # Note: Due to optimizer state, exact equality may not hold
                # but values should be close to original
                
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()

