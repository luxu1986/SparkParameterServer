# Databricks notebook source
"""
Simple training example using PySpark Parameter Server.

This example demonstrates:
1. Setting up the PS cluster
2. Creating worker clients
3. Basic pull/push operations
4. Training loop synchronization
5. Checkpointing

Run this example:
    python -m pyspark_ps.examples.simple_training
"""

import numpy as np
import time

from pyspark_ps import PSMainClient, PSWorkerClient, PSConfig


def simple_forward(embeddings, weights, labels):
    """Simple forward pass simulation."""
    # embeddings: (batch, emb_dim)
    # weights['fc1']: (emb_dim, hidden)
    # weights['fc2']: (hidden, 1)
    
    hidden = np.tanh(embeddings @ weights['fc1'])
    output = hidden @ weights['fc2']
    
    # Simple MSE loss
    loss = np.mean((output.flatten() - labels) ** 2)
    
    return loss, hidden, output


def simple_backward(embeddings, hidden, output, labels, weights):
    """Simple backward pass simulation."""
    batch_size = embeddings.shape[0]
    
    # Gradient of MSE loss
    d_output = 2 * (output.flatten() - labels) / batch_size
    d_output = d_output.reshape(-1, 1)
    
    # fc2 gradients
    d_fc2 = hidden.T @ d_output
    d_hidden = d_output @ weights['fc2'].T
    
    # tanh gradient
    d_hidden = d_hidden * (1 - hidden ** 2)
    
    # fc1 gradients
    d_fc1 = embeddings.T @ d_hidden
    
    # embedding gradients
    d_embeddings = d_hidden @ weights['fc1'].T
    
    return {
        'fc1': d_fc1,
        'fc2': d_fc2,
    }, d_embeddings


def simulate_batch():
    """Simulate a batch of training data."""
    batch_size = 32
    num_tokens = 10
    
    # Random token IDs (simulating categorical features)
    token_ids = np.random.randint(0, 10000, size=(batch_size, num_tokens))
    
    # Random labels
    labels = np.random.randn(batch_size).astype(np.float32)
    
    return token_ids, labels


def main():
    """Main training function."""
    print("=" * 60)
    print("PySpark Parameter Server - Simple Training Example")
    print("=" * 60)
    
    # Configuration
    config = PSConfig(
        num_servers=2,
        embedding_dim=32,
        embedding_optimizer="adagrad",
        weight_optimizer="adam",
        server_port_base=50500,
        auto_decay=False,
    )
    
    print(f"\nConfiguration:")
    print(f"  - Number of servers: {config.num_servers}")
    print(f"  - Embedding dimension: {config.embedding_dim}")
    print(f"  - Embedding optimizer: {config.embedding_optimizer}")
    print(f"  - Weight optimizer: {config.weight_optimizer}")
    
    # Initialize main client (on driver)
    print("\n[1] Starting PS servers...")
    main_client = PSMainClient(None, config)
    server_info = main_client.start_servers()
    print(f"    Started {len(server_info)} servers")
    
    # Initialize model weights
    print("\n[2] Initializing model weights...")
    weight_shapes = {
        'fc1': (32, 16),  # embedding_dim -> hidden
        'fc2': (16, 1),   # hidden -> output
    }
    main_client.init_weights(weight_shapes, init_strategy='xavier')
    print(f"    Initialized {len(weight_shapes)} weight tensors")
    
    # Create worker client (simulating one executor)
    print("\n[3] Creating worker client...")
    worker = PSWorkerClient(server_info, config, client_id="worker_0")
    
    # Training loop
    print("\n[4] Starting training loop...")
    num_epochs = 3
    steps_per_epoch = 10
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for step in range(steps_per_epoch):
            # Simulate batch data
            token_ids, labels = simulate_batch()
            unique_tokens = list(set(token_ids.flatten().tolist()))
            
            # Pull embeddings
            all_embeddings = worker.pull_embeddings(unique_tokens)
            token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
            
            # Reconstruct batch embeddings
            batch_embeddings = np.zeros(
                (token_ids.shape[0], config.embedding_dim),
                dtype=np.float32
            )
            for i in range(token_ids.shape[0]):
                for j in range(token_ids.shape[1]):
                    batch_embeddings[i] += all_embeddings[token_to_idx[token_ids[i, j]]]
                batch_embeddings[i] /= token_ids.shape[1]  # Mean pooling
            
            # Pull model weights
            weights = worker.pull_model()
            
            # Forward pass
            loss, hidden, output = simple_forward(batch_embeddings, weights, labels)
            epoch_loss += loss
            
            # Backward pass
            weight_grads, d_embeddings = simple_backward(
                batch_embeddings, hidden, output, labels, weights
            )
            
            # Compute embedding gradients (scatter back)
            embedding_grads = {}
            for i in range(token_ids.shape[0]):
                for j in range(token_ids.shape[1]):
                    tid = token_ids[i, j]
                    grad = d_embeddings[i] / token_ids.shape[1]
                    if tid in embedding_grads:
                        embedding_grads[tid] += grad
                    else:
                        embedding_grads[tid] = grad.copy()
            
            # Push gradients
            worker.push_gradients(weight_grads, embedding_grads)
            
            total_steps += 1
            main_client.step()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / steps_per_epoch
        
        print(f"    Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, time={epoch_time:.2f}s")
        
        # Decay embeddings at end of epoch
        main_client.decay_embeddings(method="multiply", factor=0.99)
    
    # Final statistics
    print("\n[5] Training completed!")
    stats = main_client.get_cluster_stats()
    print(f"    Total embeddings: {stats['total_embeddings']}")
    print(f"    Total steps: {total_steps}")
    
    # Cleanup
    print("\n[6] Shutting down...")
    worker.close()
    main_client.shutdown_servers()
    print("    Done!")


if __name__ == "__main__":
    main()

