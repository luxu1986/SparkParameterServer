"""
Parameter Server implementation for distributed PyTorch training.
Handles both driver (main model) and worker (embedding shards) parameter servers.
"""

import threading
import logging
import pickle
import json
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from threading import Lock
import numpy as np

from models.base_model import MainModel, EmbeddingConfig
from utils.hashing import consistent_hash


logger = logging.getLogger(__name__)


class BaseParameterServer:
    """Base class for parameter servers with common functionality."""
    
    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0, port: int = 5000):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.port = port
        self.app = Flask(__name__)
        self.lock = Lock()
        self.running = False
        self.server_thread = None
        
        # Setup Flask routes
        self._setup_routes()
        
        # Disable Flask logging in production
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    def _setup_routes(self):
        """Setup Flask routes for the parameter server."""
        self.app.add_url_rule('/health', 'health', self._health_check, methods=['GET'])
        self.app.add_url_rule('/shutdown', 'shutdown', self._shutdown, methods=['POST'])
    
    def _health_check(self):
        """Health check endpoint."""
        return jsonify({"status": "healthy", "server_type": self.__class__.__name__})
    
    def _shutdown(self):
        """Shutdown endpoint."""
        self.stop()
        return jsonify({"status": "shutting down"})
    
    def start(self):
        """Start the parameter server."""
        if self.running:
            logger.warning("Parameter server already running")
            return
        
        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"Parameter server started on port {self.port}")
    
    def stop(self):
        """Stop the parameter server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        logger.info("Parameter server stopped")
    
    def _run_server(self):
        """Run the Flask server."""
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


class DriverParameterServer(BaseParameterServer):
    """Parameter server running on the driver node, managing the main model."""
    
    def __init__(self, model_config: Dict[str, Any], learning_rate: float = 0.01, 
                 weight_decay: float = 0.0, port: int = 5000):
        super().__init__(learning_rate, weight_decay, port)
        
        # Initialize main model
        self.model = MainModel(**model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Thread pool for handling concurrent requests
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Setup additional routes
        self._setup_driver_routes()
        
        logger.info(f"Driver parameter server initialized with model: {self.model}")
    
    def _setup_driver_routes(self):
        """Setup driver-specific routes."""
        self.app.add_url_rule('/get_model', 'get_model', self._get_model, methods=['GET'])
        self.app.add_url_rule('/update_gradients', 'update_gradients', 
                            self._update_gradients, methods=['POST'])
        self.app.add_url_rule('/get_model_state', 'get_model_state', 
                            self._get_model_state, methods=['GET'])
    
    def _get_model(self):
        """Return the current model state dict."""
        with self.lock:
            try:
                # Serialize model state dict
                state_dict = self.model.state_dict()
                # Convert tensors to numpy for JSON serialization
                serialized_state = {}
                for key, tensor in state_dict.items():
                    serialized_state[key] = tensor.cpu().numpy().tolist()
                
                return jsonify({
                    "status": "success",
                    "model_state": serialized_state
                })
            except Exception as e:
                logger.error(f"Error getting model: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500
    
    def _update_gradients(self):
        """Update model parameters with received gradients."""
        try:
            data = request.get_json()
            gradients = data.get('gradients', {})
            
            with self.lock:
                # Apply gradients to model parameters
                self.optimizer.zero_grad()
                
                for name, param in self.model.named_parameters():
                    if name in gradients:
                        grad_data = torch.tensor(gradients[name], dtype=param.dtype)
                        if param.grad is None:
                            param.grad = grad_data
                        else:
                            param.grad += grad_data
                
                # Perform optimizer step
                self.optimizer.step()
                
                return jsonify({"status": "success", "message": "Gradients applied"})
                
        except Exception as e:
            logger.error(f"Error updating gradients: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    def _get_model_state(self):
        """Return model statistics and metadata."""
        with self.lock:
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() 
                                     if p.requires_grad)
                
                return jsonify({
                    "status": "success",
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay
                })
            except Exception as e:
                logger.error(f"Error getting model state: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        with self.lock:
            try:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {str(e)}")
                raise


class WorkerParameterServer(BaseParameterServer):
    """Parameter server running on worker nodes, managing embedding shards."""
    
    def __init__(self, worker_id: int, num_workers: int, 
                 embedding_config: EmbeddingConfig, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, port: int = 5001):
        super().__init__(learning_rate, weight_decay, port)
        
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.embedding_config = embedding_config
        
        # Local embedding storage: feature_id -> embedding_tensor
        self.embeddings: Dict[int, torch.Tensor] = {}
        
        # Embedding optimizer (per-parameter)
        self.embedding_optimizers: Dict[int, torch.optim.SGD] = {}
        
        # Thread pool for handling concurrent requests
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Setup worker-specific routes
        self._setup_worker_routes()
        
        logger.info(f"Worker parameter server {worker_id} initialized")
    
    def _setup_worker_routes(self):
        """Setup worker-specific routes."""
        self.app.add_url_rule('/get_embeddings', 'get_embeddings', 
                            self._get_embeddings, methods=['POST'])
        self.app.add_url_rule('/update_embedding_gradients', 'update_embedding_gradients',
                            self._update_embedding_gradients, methods=['POST'])
        self.app.add_url_rule('/get_shard_info', 'get_shard_info',
                            self._get_shard_info, methods=['GET'])
    
    def _get_embeddings(self):
        """Return embeddings for requested feature IDs."""
        try:
            data = request.get_json()
            feature_ids = data.get('feature_ids', [])
            
            embeddings = {}
            with self.lock:
                for feature_id in feature_ids:
                    # Check if this feature belongs to this worker
                    if self._owns_feature(feature_id):
                        if feature_id not in self.embeddings:
                            # Initialize new embedding
                            self._initialize_embedding(feature_id)
                        
                        embeddings[feature_id] = self.embeddings[feature_id].tolist()
            
            return jsonify({
                "status": "success",
                "embeddings": embeddings,
                "worker_id": self.worker_id
            })
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    def _update_embedding_gradients(self):
        """Update embeddings with received gradients."""
        try:
            data = request.get_json()
            gradients = data.get('gradients', {})
            
            with self.lock:
                for feature_id_str, grad_data in gradients.items():
                    feature_id = int(feature_id_str)
                    
                    if self._owns_feature(feature_id) and feature_id in self.embeddings:
                        # Get or create optimizer for this embedding
                        if feature_id not in self.embedding_optimizers:
                            self.embedding_optimizers[feature_id] = torch.optim.Adam(
                                [self.embeddings[feature_id]], lr=self.learning_rate, weight_decay=self.weight_decay
                            )
                        
                        # Apply gradient
                        optimizer = self.embedding_optimizers[feature_id]
                        optimizer.zero_grad()
                        
                        grad_tensor = torch.tensor(grad_data, dtype=self.embeddings[feature_id].dtype)
                        self.embeddings[feature_id].grad = grad_tensor
                        
                        optimizer.step()
            
            return jsonify({"status": "success", "message": "Embedding gradients applied"})
            
        except Exception as e:
            logger.error(f"Error updating embedding gradients: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    def _get_shard_info(self):
        """Return information about this worker's embedding shard."""
        with self.lock:
            try:
                return jsonify({
                    "status": "success",
                    "worker_id": self.worker_id,
                    "num_embeddings": len(self.embeddings),
                    "embedding_dim": self.embedding_config.embedding_dim,
                    "total_parameters": len(self.embeddings) * self.embedding_config.embedding_dim
                })
            except Exception as e:
                logger.error(f"Error getting shard info: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500
    
    def _owns_feature(self, feature_id: int) -> bool:
        """Check if this worker owns the given feature ID."""
        return consistent_hash(feature_id, self.num_workers) == self.worker_id
    
    def _initialize_embedding(self, feature_id: int):
        """Initialize a new embedding vector for the given feature ID."""
        if feature_id in self.embeddings:
            return
        
        # Initialize with Xavier uniform initialization
        embedding = torch.empty(self.embedding_config.embedding_dim)
        nn.init.xavier_uniform_(embedding.unsqueeze(0))  # Need 2D for xavier_uniform
        embedding = embedding.squeeze(0)  # Back to 1D
        embedding.requires_grad_(True)
        
        self.embeddings[feature_id] = embedding
        logger.debug(f"Initialized embedding for feature {feature_id}")
    
    def save_shard_checkpoint(self, checkpoint_path: str):
        """Save embedding shard checkpoint."""
        with self.lock:
            try:
                checkpoint = {
                    'worker_id': self.worker_id,
                    'embeddings': {fid: emb.detach().clone() for fid, emb in self.embeddings.items()},
                    'embedding_config': self.embedding_config.__dict__,
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Shard checkpoint saved to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving shard checkpoint: {str(e)}")
                raise
    
    def load_shard_checkpoint(self, checkpoint_path: str):
        """Load embedding shard from checkpoint."""
        with self.lock:
            try:
                checkpoint = torch.load(checkpoint_path)
                self.embeddings = checkpoint['embeddings']
                
                # Reinitialize optimizers
                self.embedding_optimizers = {}
                for feature_id in self.embeddings:
                    self.embedding_optimizers[feature_id] = torch.optim.Adam(
                        [self.embeddings[feature_id]], lr=self.learning_rate, weight_decay=self.weight_decay
                    )
                
                logger.info(f"Shard checkpoint loaded from {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error loading shard checkpoint: {str(e)}")
                raise 