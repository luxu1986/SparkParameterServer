"""
PyTorch model definitions for distributed training.
Contains main model architecture and embedding handling.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding tables."""
    embedding_dim: int
    max_features: Optional[int] = None  # Maximum number of unique features
    init_std: float = 0.1  # Standard deviation for initialization
    
    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")


class MainModel(nn.Module):
    """
    Main neural network model that processes dense features and embeddings.
    This model will be stored on the driver parameter server.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float = 0.1, 
                 activation: str = "relu", **kwargs):
        """
        Initialize the main model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            activation: Activation function name
        """
        super(MainModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"MainModel initialized: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the main model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_param_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class CombinedModel(nn.Module):
    """
    Combined model that handles both dense features and embeddings.
    This is used by workers for local computation.
    """
    
    def __init__(self, main_model_config: Dict[str, Any], 
                 embedding_config: Dict[str, Any]):
        """
        Initialize the combined model.
        
        Args:
            main_model_config: Configuration for the main model
            embedding_config: Configuration for embeddings
        """
        super(CombinedModel, self).__init__()
        
        self.embedding_config = EmbeddingConfig(**embedding_config)
        
        # Calculate total input dimension (dense features + embeddings)
        dense_dim = main_model_config["input_dim"]
        embedding_dim = self.embedding_config.embedding_dim
        
        # Adjust main model input dimension to include embeddings
        adjusted_config = main_model_config.copy()
        adjusted_config["input_dim"] = dense_dim + embedding_dim
        
        # Initialize main model
        self.main_model = MainModel(**adjusted_config)
        
        # Store original dense feature dimension
        self.dense_dim = dense_dim
        
        logger.info(f"CombinedModel initialized: dense_dim={dense_dim}, "
                   f"embedding_dim={embedding_dim}")
    
    def forward_with_embeddings(self, dense_features: torch.Tensor, 
                               embeddings: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with dense features and embeddings.
        
        Args:
            dense_features: Dense feature tensor of shape (batch_size, dense_dim)
            embeddings: Dictionary mapping feature IDs to embedding tensors
            
        Returns:
            Output tensor from the main model
        """
        batch_size = dense_features.size(0)
        
        # Aggregate embeddings (mean pooling for simplicity)
        if embeddings:
            embedding_tensors = list(embeddings.values())
            stacked_embeddings = torch.stack(embedding_tensors, dim=0)  # (num_embeddings, embedding_dim)
            
            # Mean pooling across embeddings
            pooled_embeddings = torch.mean(stacked_embeddings, dim=0)  # (embedding_dim,)
            
            # Repeat for batch
            batch_embeddings = pooled_embeddings.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, embedding_dim)
        else:
            # No embeddings - use zeros
            batch_embeddings = torch.zeros(batch_size, self.embedding_config.embedding_dim)
        
        # Concatenate dense features and embeddings
        combined_input = torch.cat([dense_features, batch_embeddings], dim=1)
        
        # Forward through main model
        return self.main_model(combined_input)
    
    def load_main_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Load main model parameters from state dict."""
        self.main_model.load_state_dict(state_dict, strict=False)
    
    def get_main_model_gradients(self) -> Dict[str, torch.Tensor]:
        """Extract gradients from the main model."""
        gradients = {}
        for name, param in self.main_model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients
    
    def get_embedding_gradients(self, feature_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Extract gradients for embeddings used in the forward pass.
        
        Note: This is a simplified implementation. In practice, you would need
        to track which embeddings were used and their gradients.
        """
        # This is a placeholder - in a real implementation, you would
        # need to track embedding usage during forward pass
        embedding_gradients = {}
        
        # For now, return dummy gradients
        for feature_id in feature_ids:
            embedding_gradients[feature_id] = torch.randn(self.embedding_config.embedding_dim)
        
        return embedding_gradients


class EmbeddingLayer(nn.Module):
    """
    Embedding layer for categorical features.
    This is used for reference but embeddings are actually stored in parameter servers.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 padding_idx: Optional[int] = None):
        """
        Initialize embedding layer.
        
        Args:
            num_embeddings: Size of the dictionary of embeddings
            embedding_dim: Size of each embedding vector
            padding_idx: If given, pads the output with the embedding vector at padding_idx
        """
        super(EmbeddingLayer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding layer.
        
        Args:
            input_ids: Tensor of feature IDs
            
        Returns:
            Embedding tensor
        """
        return self.embedding(input_ids)


class DeepFM(nn.Module):
    """
    Example of a more complex model architecture (DeepFM).
    This demonstrates how to extend the base model for specific use cases.
    """
    
    def __init__(self, dense_dim: int, embedding_dim: int, 
                 hidden_dims: List[int], output_dim: int = 1):
        """
        Initialize DeepFM model.
        
        Args:
            dense_dim: Dimension of dense features
            embedding_dim: Dimension of embeddings
            hidden_dims: Hidden layer dimensions for deep component
            output_dim: Output dimension
        """
        super(DeepFM, self).__init__()
        
        self.dense_dim = dense_dim
        self.embedding_dim = embedding_dim
        
        # FM component (factorization machine)
        self.fm_linear = nn.Linear(dense_dim + embedding_dim, 1)
        
        # Deep component
        deep_input_dim = dense_dim + embedding_dim
        deep_layers = []
        prev_dim = deep_input_dim
        
        for hidden_dim in hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        deep_layers.append(nn.Linear(prev_dim, output_dim))
        self.deep_network = nn.Sequential(*deep_layers)
        
        # Final combination layer
        self.final_layer = nn.Linear(2, output_dim)
        
    def forward(self, dense_features: torch.Tensor, 
                embeddings: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through DeepFM.
        
        Args:
            dense_features: Dense feature tensor
            embeddings: Dictionary of embeddings
            
        Returns:
            Model predictions
        """
        batch_size = dense_features.size(0)
        
        # Aggregate embeddings
        if embeddings:
            embedding_tensors = list(embeddings.values())
            stacked_embeddings = torch.stack(embedding_tensors, dim=0)
            pooled_embeddings = torch.mean(stacked_embeddings, dim=0)
            batch_embeddings = pooled_embeddings.unsqueeze(0).repeat(batch_size, 1)
        else:
            batch_embeddings = torch.zeros(batch_size, self.embedding_dim)
        
        # Combine features
        combined_features = torch.cat([dense_features, batch_embeddings], dim=1)
        
        # FM component
        fm_output = self.fm_linear(combined_features)
        
        # Deep component
        deep_output = self.deep_network(combined_features)
        
        # Combine FM and deep outputs
        combined_output = torch.cat([fm_output, deep_output], dim=1)
        final_output = self.final_layer(combined_output)
        
        return final_output


# Factory function for creating models
def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create models based on type.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == "main":
        return MainModel(**kwargs)
    elif model_type == "combined":
        return CombinedModel(**kwargs)
    elif model_type == "deepfm":
        return DeepFM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model registry for easy extension
MODEL_REGISTRY = {
    "main": MainModel,
    "combined": CombinedModel,
    "deepfm": DeepFM,
    "embedding": EmbeddingLayer
}


def register_model(name: str, model_class: type):
    """Register a new model class."""
    MODEL_REGISTRY[name.lower()] = model_class


def get_model_class(name: str) -> type:
    """Get model class by name."""
    return MODEL_REGISTRY.get(name.lower()) 