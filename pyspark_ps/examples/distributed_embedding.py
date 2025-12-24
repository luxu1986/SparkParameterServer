"""
Distributed embedding training example using PySpark Parameter Server.

This example demonstrates:
1. Using PS with Spark RDD operations
2. Multiple worker clients in parallel
3. Barrier synchronization between epochs
4. Embedding decay and pruning
5. S3 checkpointing (simulated with local storage)

This example requires PySpark to be installed:
    pip install pyspark

Run this example:
    python -m pyspark_ps.examples.distributed_embedding
"""

import numpy as np
import time
import threading
from typing import Iterator, Tuple, Dict, Any

# Import PS components
from pyspark_ps import PSMainClient, PSWorkerClient, PSConfig
from pyspark_ps.communication.protocol import ServerInfo


def simulate_partition_data(partition_id: int, num_samples: int = 100):
    """Simulate training data for a partition."""
    np.random.seed(partition_id * 1000)
    
    samples = []
    for i in range(num_samples):
        # Random token IDs (simulating user-item interactions)
        num_tokens = np.random.randint(3, 10)
        token_ids = np.random.randint(0, 100000, size=num_tokens).tolist()
        
        # Random label
        label = np.random.rand()
        
        samples.append({
            'token_ids': token_ids,
            'label': label,
        })
    
    return samples


def train_partition(
    partition_id: int,
    data: Iterator[Dict[str, Any]],
    server_info_list: list,
    config_dict: dict,
    embedding_dim: int
) -> Iterator[Tuple[int, float]]:
    """
    Train on a partition of data.
    
    This function runs on each Spark executor.
    """
    # Reconstruct config and server info
    config = PSConfig(**{k: v for k, v in config_dict.items() 
                        if k in PSConfig.__dataclass_fields__})
    server_info = [ServerInfo(**s) for s in server_info_list]
    
    # Create worker client for this partition
    worker = PSWorkerClient(
        server_info,
        config,
        client_id=f"partition_{partition_id}"
    )
    
    try:
        total_loss = 0.0
        num_samples = 0
        
        for sample in data:
            token_ids = sample['token_ids']
            label = sample['label']
            
            # Pull embeddings
            embeddings = worker.pull_embeddings(token_ids)
            
            # Simple mean pooling
            pooled = embeddings.mean(axis=0)
            
            # Simple linear prediction
            prediction = np.tanh(pooled.sum() / embedding_dim)
            
            # MSE loss
            loss = (prediction - label) ** 2
            total_loss += loss
            num_samples += 1
            
            # Compute gradients (simplified)
            d_prediction = 2 * (prediction - label)
            d_pooled = d_prediction * (1 - prediction ** 2) / embedding_dim
            
            # Scatter gradients to embeddings
            embedding_grads = {
                tid: np.full(embedding_dim, d_pooled / len(token_ids), dtype=np.float32)
                for tid in token_ids
            }
            
            # Push gradients
            worker.push_embedding_gradients(embedding_grads)
        
        avg_loss = total_loss / max(num_samples, 1)
        yield (partition_id, avg_loss)
        
    finally:
        worker.close()


def run_with_spark():
    """Run example with actual PySpark."""
    try:
        from pyspark import SparkContext, SparkConf
    except ImportError:
        print("PySpark not available. Running simulated version instead.")
        return run_simulated()
    
    # Spark configuration
    conf = SparkConf() \
        .setAppName("PSDistributedEmbedding") \
        .setMaster("local[4]")
    
    sc = SparkContext(conf=conf)
    
    try:
        # PS configuration
        config = PSConfig(
            num_servers=2,
            embedding_dim=32,
            embedding_optimizer="adagrad",
            server_port_base=50600,
        )
        
        print("\n" + "=" * 60)
        print("Distributed Embedding Training with PySpark")
        print("=" * 60)
        
        # Start PS cluster
        print("\n[1] Starting PS cluster...")
        main_client = PSMainClient(sc, config)
        server_info = main_client.start_servers()
        
        # Prepare broadcast variables
        server_info_list = [s.to_dict() for s in server_info]
        config_dict = config.to_dict()
        
        server_info_bc = sc.broadcast(server_info_list)
        config_bc = sc.broadcast(config_dict)
        
        # Number of partitions (simulating workers)
        num_partitions = 4
        
        # Training loop
        num_epochs = 3
        print(f"\n[2] Training for {num_epochs} epochs with {num_partitions} partitions...")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Create RDD with simulated data
            partition_data = [
                simulate_partition_data(p, num_samples=50)
                for p in range(num_partitions)
            ]
            
            rdd = sc.parallelize(
                [(i, d) for i, data in enumerate(partition_data) for d in data],
                numSlices=num_partitions
            )
            
            # Train on each partition
            def process_partition(iterator):
                items = list(iterator)
                if not items:
                    return
                
                partition_id = items[0][0]
                data = [item[1] for item in items]
                
                yield from train_partition(
                    partition_id,
                    iter(data),
                    server_info_bc.value,
                    config_bc.value,
                    config.embedding_dim
                )
            
            results = rdd.mapPartitions(process_partition).collect()
            
            epoch_time = time.time() - epoch_start
            avg_loss = np.mean([r[1] for r in results])
            
            print(f"    Epoch {epoch + 1}: avg_loss={avg_loss:.6f}, time={epoch_time:.2f}s")
            
            # Decay embeddings
            main_client.decay_embeddings(method="multiply", factor=0.99)
        
        # Final stats
        print("\n[3] Training complete!")
        stats = main_client.get_cluster_stats()
        print(f"    Total embeddings: {stats['total_embeddings']}")
        
        # Cleanup
        print("\n[4] Shutting down...")
        main_client.shutdown_servers()
        
    finally:
        sc.stop()
    
    print("Done!")


def run_simulated():
    """Run simulated version without Spark."""
    print("\n" + "=" * 60)
    print("Distributed Embedding Training (Simulated)")
    print("=" * 60)
    
    # PS configuration
    config = PSConfig(
        num_servers=2,
        embedding_dim=32,
        embedding_optimizer="adagrad",
        server_port_base=50700,
    )
    
    # Start PS cluster
    print("\n[1] Starting PS cluster...")
    main_client = PSMainClient(None, config)
    server_info = main_client.start_servers()
    
    # Prepare data
    server_info_list = [s.to_dict() for s in server_info]
    config_dict = config.to_dict()
    
    # Simulate partitions with threads
    num_partitions = 4
    num_epochs = 3
    
    print(f"\n[2] Training for {num_epochs} epochs with {num_partitions} workers...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        results = []
        results_lock = threading.Lock()
        
        def worker_thread(partition_id):
            data = simulate_partition_data(partition_id, num_samples=50)
            
            for result in train_partition(
                partition_id,
                iter(data),
                server_info_list,
                config_dict,
                config.embedding_dim
            ):
                with results_lock:
                    results.append(result)
        
        # Run workers in parallel
        threads = [
            threading.Thread(target=worker_thread, args=(i,))
            for i in range(num_partitions)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean([r[1] for r in results])
        
        print(f"    Epoch {epoch + 1}: avg_loss={avg_loss:.6f}, time={epoch_time:.2f}s")
        
        # Decay embeddings
        main_client.decay_embeddings(method="multiply", factor=0.99)
    
    # Final stats
    print("\n[3] Training complete!")
    stats = main_client.get_cluster_stats()
    print(f"    Total embeddings: {stats['total_embeddings']}")
    
    # Prune low-frequency embeddings
    print("\n[4] Pruning low-frequency embeddings...")
    prune_result = main_client.decay_embeddings(method="prune", min_count=2)
    total_pruned = sum(r.get('pruned', 0) for r in prune_result['results'])
    print(f"    Pruned {total_pruned} embeddings")
    
    # Final stats after pruning
    stats = main_client.get_cluster_stats()
    print(f"    Remaining embeddings: {stats['total_embeddings']}")
    
    # Cleanup
    print("\n[5] Shutting down...")
    main_client.shutdown_servers()
    print("Done!")


def main():
    """Main entry point."""
    # Try to use Spark if available and working
    try:
        from pyspark import SparkContext, SparkConf
        # Test if Spark is actually usable (needs Java)
        conf = SparkConf().setAppName("test").setMaster("local[1]")
        sc = SparkContext(conf=conf)
        sc.stop()
        run_with_spark()
    except Exception as e:
        print(f"Note: PySpark not available ({type(e).__name__}), running simulated version.")
        run_simulated()


if __name__ == "__main__":
    main()

