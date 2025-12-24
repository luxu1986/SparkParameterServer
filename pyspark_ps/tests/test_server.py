"""Tests for PS server components."""

import unittest
import numpy as np
import time

from pyspark_ps.server.update_counter import UpdateCounter
from pyspark_ps.server.embedding_store import EmbeddingStore
from pyspark_ps.server.weight_store import WeightStore
from pyspark_ps.server.shard_manager import ShardManager
from pyspark_ps.utils.config import PSConfig


class TestUpdateCounter(unittest.TestCase):
    """Tests for update counter."""
    
    def test_increment(self):
        """Test incrementing counts."""
        counter = UpdateCounter()
        
        counter.increment(1)
        counter.increment(1)
        counter.increment(2)
        
        self.assertEqual(counter.get_count(1), 2)
        self.assertEqual(counter.get_count(2), 1)
        self.assertEqual(counter.get_count(3), 0)
    
    def test_batch_increment(self):
        """Test batch increment."""
        counter = UpdateCounter()
        
        counter.increment_batch([1, 2, 3, 1, 1])
        
        self.assertEqual(counter.get_count(1), 3)
        self.assertEqual(counter.get_count(2), 1)
        self.assertEqual(counter.get_count(3), 1)
    
    def test_decay(self):
        """Test decay operation."""
        counter = UpdateCounter()
        counter.increment_batch([1, 2, 3])
        counter.increment(1)  # count = 2
        
        counter.decay(0.5)
        
        self.assertEqual(counter.get_count(1), 1.0)
        self.assertEqual(counter.get_count(2), 0.5)
    
    def test_prune(self):
        """Test pruning low-count entries."""
        counter = UpdateCounter()
        
        for i in range(10):
            for _ in range(i + 1):
                counter.increment(i)
        
        # Prune entries with count < 5
        pruned = counter.prune(5)
        
        # Tokens 0-3 should be pruned (counts 1-4)
        self.assertEqual(len(pruned), 4)
        self.assertEqual(len(counter), 6)
    
    def test_stats(self):
        """Test statistics."""
        counter = UpdateCounter()
        
        for i in range(10):
            counter.increment(i)
        
        stats = counter.get_stats()
        
        self.assertEqual(stats["num_entries"], 10)
        self.assertEqual(stats["min_count"], 1)
        self.assertEqual(stats["max_count"], 1)


class TestEmbeddingStore(unittest.TestCase):
    """Tests for embedding store."""
    
    def test_get_creates_embedding(self):
        """Test that get creates embedding if missing."""
        store = EmbeddingStore(embedding_dim=64, init_strategy="zeros")
        
        emb = store.get(1)
        
        self.assertIsNotNone(emb)
        self.assertEqual(emb.shape, (64,))
        np.testing.assert_array_equal(emb, np.zeros(64))
    
    def test_get_batch(self):
        """Test batch get."""
        store = EmbeddingStore(embedding_dim=32)
        
        embeddings, found_ids = store.get_batch([1, 2, 3, 4, 5])
        
        self.assertEqual(embeddings.shape, (5, 32))
        self.assertEqual(len(found_ids), 5)
    
    def test_update(self):
        """Test gradient update."""
        store = EmbeddingStore(
            embedding_dim=8,
            init_strategy="zeros",
            optimizer_name="sgd",
            optimizer_config={"learning_rate": 1.0, "momentum": 0.0}
        )
        
        # Get initial embedding (zeros)
        _ = store.get(1)
        
        # Update with gradient
        gradient = np.ones(8, dtype=np.float32)
        store.update(1, gradient)
        
        # Check update was applied
        emb = store.get(1)
        np.testing.assert_array_almost_equal(emb, -np.ones(8))
    
    def test_update_batch(self):
        """Test batch gradient update."""
        store = EmbeddingStore(embedding_dim=4, init_strategy="zeros")
        
        # Initialize
        store.get_batch([1, 2, 3])
        
        # Batch update
        gradients = {
            1: np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            2: np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        }
        store.update_batch(gradients)
        
        # Verify updates were applied
        stats = store.get_stats()
        self.assertEqual(stats["total_updates"], 2)
    
    def test_decay(self):
        """Test embedding decay."""
        store = EmbeddingStore(embedding_dim=4, init_strategy="normal")
        
        # Create some embeddings
        store.get_batch([1, 2, 3])
        
        # Store original values
        orig = {i: store.get(i).copy() for i in [1, 2, 3]}
        
        # Decay by 0.5
        store.decay(0.5)
        
        # Check values halved
        for i in [1, 2, 3]:
            np.testing.assert_array_almost_equal(
                store.get(i),
                orig[i] * 0.5
            )
    
    def test_prune(self):
        """Test embedding pruning."""
        store = EmbeddingStore(embedding_dim=4)
        
        # Create embeddings with different update counts
        for i in range(10):
            store.get(i)
            for _ in range(i):
                store.update(i, np.zeros(4))
        
        # Prune
        pruned = store.prune(min_count=5)
        
        self.assertGreater(pruned, 0)
        self.assertLess(len(store), 10)
    
    def test_state_save_restore(self):
        """Test saving and restoring state."""
        store1 = EmbeddingStore(embedding_dim=8)
        
        # Add some embeddings
        store1.get_batch([1, 2, 3])
        store1.update(1, np.ones(8) * 0.1)
        
        # Save state
        state = store1.get_state()
        
        # Restore to new store
        store2 = EmbeddingStore(embedding_dim=8)
        store2.set_state(state)
        
        # Verify
        np.testing.assert_array_almost_equal(
            store1.get(1),
            store2.get(1)
        )


class TestWeightStore(unittest.TestCase):
    """Tests for weight store."""
    
    def test_init_weights(self):
        """Test weight initialization."""
        store = WeightStore()
        
        store.init_weights("layer1", (10, 20), init_strategy="zeros")
        store.init_weights("layer2", (20, 5), init_strategy="normal")
        
        w1 = store.get("layer1")
        w2 = store.get("layer2")
        
        self.assertEqual(w1.shape, (10, 20))
        self.assertEqual(w2.shape, (20, 5))
        np.testing.assert_array_equal(w1, np.zeros((10, 20)))
    
    def test_update(self):
        """Test weight update."""
        store = WeightStore(
            optimizer_name="sgd",
            optimizer_config={"learning_rate": 1.0, "momentum": 0.0}
        )
        
        store.init_weights("test", (5, 5), init_strategy="zeros")
        
        gradient = np.ones((5, 5), dtype=np.float32)
        store.update("test", gradient)
        
        weights = store.get("test")
        np.testing.assert_array_almost_equal(weights, -np.ones((5, 5)))
    
    def test_version_tracking(self):
        """Test version tracking."""
        store = WeightStore()
        
        store.init_weights("test", (3, 3))
        
        initial_version = store.get_version()
        
        store.update("test", np.ones((3, 3)))
        
        self.assertEqual(store.get_version(), initial_version + 1)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        store = WeightStore(
            optimizer_name="sgd",
            optimizer_config={"learning_rate": 1.0, "momentum": 0.0}
        )
        
        store.init_weights("test", (3,), init_strategy="zeros")
        
        # Accumulate gradients
        store.accumulate_gradient("test", np.array([1.0, 0.0, 0.0]))
        store.accumulate_gradient("test", np.array([0.0, 1.0, 0.0]))
        store.accumulate_gradient("test", np.array([0.0, 0.0, 1.0]))
        
        # Apply with averaging
        store.apply_accumulated_gradients(average=True)
        
        weights = store.get("test")
        # Average gradient is [1/3, 1/3, 1/3], update is -lr * avg
        expected = np.array([-1/3, -1/3, -1/3])
        np.testing.assert_array_almost_equal(weights, expected)


class TestShardManager(unittest.TestCase):
    """Tests for shard manager."""
    
    def setUp(self):
        self.config = PSConfig(
            num_servers=4,
            embedding_dim=16,
            embedding_optimizer="sgd",
            weight_optimizer="sgd"
        )
    
    def test_token_ownership(self):
        """Test token ownership determination."""
        manager = ShardManager(0, 4, self.config)
        
        # Test ownership
        owns_count = sum(1 for i in range(100) if manager.owns_token(i))
        
        # Should own roughly 1/4 of tokens
        self.assertGreater(owns_count, 10)
        self.assertLess(owns_count, 40)
    
    def test_partition_tokens(self):
        """Test token partitioning."""
        manager = ShardManager(0, 4, self.config)
        
        tokens = list(range(100))
        partitions = manager.partition_tokens(tokens)
        
        # All 4 servers should have some tokens
        for i in range(4):
            self.assertIn(i, partitions)
        
        # Total should equal input
        total = sum(len(p) for p in partitions.values())
        self.assertEqual(total, 100)
    
    def test_embedding_operations(self):
        """Test embedding get/update through shard manager."""
        manager = ShardManager(0, 4, self.config)
        
        # Get embeddings for tokens owned by this shard
        all_tokens = list(range(100))
        owned = manager.filter_owned_tokens(all_tokens)
        
        # Get embeddings
        embeddings, found = manager.get_embeddings(owned)
        
        self.assertEqual(len(found), len(owned))
        self.assertEqual(embeddings.shape[0], len(owned))
        self.assertEqual(embeddings.shape[1], 16)
    
    def test_weight_operations(self):
        """Test weight initialization and updates."""
        manager = ShardManager(0, 4, self.config)
        
        shapes = {
            "layer1": (32, 16),
            "layer2": (16, 8),
        }
        manager.init_weights(shapes)
        
        weights = manager.get_weights()
        
        self.assertIn("layer1", weights)
        self.assertIn("layer2", weights)
        self.assertEqual(weights["layer1"].shape, (32, 16))


if __name__ == "__main__":
    unittest.main()

