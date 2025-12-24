"""TensorFlow 2.0 model wrapper for numpy <-> tensor conversion.

This module bridges the gap between TensorFlow Keras models (which work with tensors)
and the Parameter Server (which uses numpy arrays for network transfer).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class TFModelWrapper:
    """
    Wrapper for TensorFlow 2.0 models handling numpy <-> tensor conversion.
    
    TensorFlow Keras models work with tensors internally, but PS uses numpy arrays
    for network transfer. This wrapper bridges the two:
    
    - get_weights() -> Dict[str, np.ndarray]: Extract weights for PS storage
    - set_weights(Dict[str, np.ndarray]): Load weights from PS into model
    - compute_gradients() -> Dict[str, np.ndarray]: Get gradients as numpy for PS
    
    Conversion flow:
    - model.get_weights() returns List[np.ndarray] (already numpy!)
    - GradientTape.gradient() returns List[tf.Tensor], convert via .numpy()
    - model.set_weights() accepts List[np.ndarray] (already numpy!)
    
    We use Dict[str, np.ndarray] with layer names as keys for clarity.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        model_builder: Optional[Callable[[], Any]] = None
    ):
        """
        Initialize the wrapper.
        
        Args:
            model: Optional existing tf.keras.Model
            model_builder: Optional function that returns a tf.keras.Model
        """
        self._model = model
        self._model_builder = model_builder
        self._weight_names: Optional[List[str]] = None
        self._tf = None  # Lazy import
        
        if model is not None:
            self._init_weight_names()
    
    def _ensure_tf(self):
        """Lazy import TensorFlow."""
        if self._tf is None:
            import tensorflow as tf
            self._tf = tf
        return self._tf
    
    def _init_weight_names(self):
        """Initialize weight name mapping from model."""
        if self._model is None:
            return
        
        # Build unique names using layer.name/weight.name format
        # This ensures uniqueness since model.weights can have duplicate short names
        self._weight_names = []
        for layer in self._model.layers:
            for weight in layer.weights:
                # Use format: layer_name/weight_name for uniqueness
                unique_name = f"{layer.name}/{weight.name}"
                self._weight_names.append(unique_name)
    
    def build_model(self) -> Any:
        """
        Build model using builder function.
        
        Returns:
            The built tf.keras.Model
        """
        if self._model_builder is None:
            raise ValueError("No model_builder provided")
        
        self._model = self._model_builder()
        self._init_weight_names()
        return self._model
    
    def get_model(self) -> Any:
        """
        Get underlying tf.keras.Model.
        
        Returns:
            The tf.keras.Model instance
        """
        if self._model is None:
            if self._model_builder is not None:
                return self.build_model()
            raise ValueError("No model available")
        return self._model
    
    def set_model(self, model: Any):
        """
        Set the model directly.
        
        Args:
            model: tf.keras.Model instance
        """
        self._model = model
        self._init_weight_names()
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract model weights as Dict[layer_name, np.ndarray].
        
        Used for: initializing PS weight_store, checkpointing.
        
        Internally: model.get_weights() returns List[np.ndarray],
        we convert to dict with layer names as keys.
        
        Returns:
            Dictionary mapping weight names to numpy arrays
        """
        model = self.get_model()
        weights_list = model.get_weights()
        
        if self._weight_names is None:
            self._init_weight_names()
        
        return {
            name: weight
            for name, weight in zip(self._weight_names, weights_list)
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """
        Load weights from dict into model.
        
        Used for: loading weights pulled from PS.
        
        Internally: convert dict back to list, call model.set_weights().
        
        Args:
            weights: Dictionary mapping weight names to numpy arrays
        """
        model = self.get_model()
        
        if self._weight_names is None:
            self._init_weight_names()
        
        # Convert dict to list in correct order
        weights_list = [weights[name] for name in self._weight_names]
        model.set_weights(weights_list)
    
    def get_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get shapes of all model weights.
        
        Used for: initializing PS weight_store.
        
        Returns:
            Dictionary mapping weight names to shape tuples
        """
        weights = self.get_weights()
        return {name: tuple(w.shape) for name, w in weights.items()}
    
    def get_weight_names(self) -> List[str]:
        """
        Get list of weight names in order.
        
        Returns:
            List of weight names
        """
        if self._weight_names is None:
            self._init_weight_names()
        return self._weight_names.copy() if self._weight_names else []
    
    def compute_gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        loss_fn: Optional[Any] = None,
        sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute loss and gradients, return as numpy arrays.
        
        Process:
        1. Convert inputs from numpy to tensor
        2. Forward pass with GradientTape
        3. Compute loss (with sample weights if provided)
        4. Compute gradients (returns List[tf.Tensor])
        5. Convert gradients to Dict[str, np.ndarray]
        
        Args:
            inputs: Input features as numpy array
            targets: Target values as numpy array
            loss_fn: Loss function (defaults to MSE)
            sample_weights: Optional sample weights
            
        Returns:
            Tuple of (loss_value, gradients_dict) - gradients ready for PS push
        """
        tf = self._ensure_tf()
        model = self.get_model()
        
        # Default loss function
        if loss_fn is None:
            loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Convert to tensors
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        targets_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
        
        if sample_weights is not None:
            weights_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        else:
            weights_tensor = None
        
        # Forward pass with gradient tape
        with tf.GradientTape() as tape:
            predictions = model(inputs_tensor, training=True)
            
            # Compute per-sample loss
            per_sample_loss = loss_fn(targets_tensor, predictions)
            
            # Handle different loss shapes
            if len(per_sample_loss.shape) > 1:
                # Multi-output loss - reduce across outputs first
                per_sample_loss = tf.reduce_mean(per_sample_loss, axis=-1)
            
            # Apply sample weights if provided
            if weights_tensor is not None and len(per_sample_loss.shape) > 0:
                loss = tf.reduce_mean(per_sample_loss * weights_tensor)
            else:
                loss = tf.reduce_mean(per_sample_loss)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Convert to dict with numpy arrays
        if self._weight_names is None:
            self._init_weight_names()
        
        # Map gradients to trainable variable names
        trainable_names = [v.name for v in model.trainable_variables]
        gradients_dict = {}
        
        for name, grad in zip(trainable_names, gradients):
            if grad is not None:
                gradients_dict[name] = grad.numpy()
            else:
                # Zero gradient for non-differentiable weights
                for w in model.trainable_variables:
                    if w.name == name:
                        gradients_dict[name] = np.zeros(w.shape, dtype=np.float32)
                        break
        
        return float(loss.numpy()), gradients_dict
    
    def compute_gradients_with_embeddings(
        self,
        dense_inputs: np.ndarray,
        sparse_embeddings: Dict[str, np.ndarray],
        targets: np.ndarray,
        loss_fn: Optional[Any] = None,
        sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute gradients for both model weights and embeddings.
        
        This method handles the case where embeddings are pulled from PS
        and we need gradients for both the model and the embeddings.
        
        Args:
            dense_inputs: Dense feature inputs
            sparse_embeddings: Dict mapping feature_name -> embeddings array
            targets: Target values
            loss_fn: Loss function
            sample_weights: Optional sample weights
            
        Returns:
            Tuple of (loss, model_gradients, embedding_gradients)
        """
        tf = self._ensure_tf()
        model = self.get_model()
        
        if loss_fn is None:
            loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Convert to tensors
        dense_tensor = tf.convert_to_tensor(dense_inputs, dtype=tf.float32)
        targets_tensor = tf.convert_to_tensor(targets, dtype=tf.float32)
        
        # Convert embeddings to tensors (as Variables to track gradients)
        embedding_tensors = {}
        for name, emb in sparse_embeddings.items():
            embedding_tensors[name] = tf.Variable(
                emb, trainable=True, dtype=tf.float32
            )
        
        if sample_weights is not None:
            weights_tensor = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        else:
            weights_tensor = None
        
        # Concatenate embeddings and dense features
        all_embeddings = [embedding_tensors[name] for name in sorted(embedding_tensors.keys())]
        
        with tf.GradientTape() as tape:
            # Watch embedding variables
            for emb_var in embedding_tensors.values():
                tape.watch(emb_var)
            
            # Concatenate inputs
            if all_embeddings:
                concat_embeddings = tf.concat(all_embeddings, axis=-1)
                inputs_tensor = tf.concat([concat_embeddings, dense_tensor], axis=-1)
            else:
                inputs_tensor = dense_tensor
            
            # Forward pass
            predictions = model(inputs_tensor, training=True)
            
            # Compute loss
            if weights_tensor is not None:
                per_sample_loss = loss_fn(targets_tensor, predictions)
                if len(per_sample_loss.shape) > 0:
                    loss = tf.reduce_mean(per_sample_loss * weights_tensor)
                else:
                    loss = per_sample_loss
            else:
                loss = loss_fn(targets_tensor, predictions)
        
        # Compute model gradients
        model_grads = tape.gradient(loss, model.trainable_variables)
        
        # Compute embedding gradients
        embedding_grads = tape.gradient(
            loss,
            list(embedding_tensors.values())
        )
        
        # Convert model gradients to dict
        trainable_names = [v.name for v in model.trainable_variables]
        model_gradients_dict = {}
        for name, grad in zip(trainable_names, model_grads):
            if grad is not None:
                model_gradients_dict[name] = grad.numpy()
        
        # Convert embedding gradients to dict
        embedding_gradients_dict = {}
        for (name, _), grad in zip(
            sorted(embedding_tensors.items()),
            embedding_grads
        ):
            if grad is not None:
                embedding_gradients_dict[name] = grad.numpy()
        
        return float(loss.numpy()), model_gradients_dict, embedding_gradients_dict
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run inference on inputs.
        
        Args:
            inputs: Input features as numpy array
            
        Returns:
            Predictions as numpy array
        """
        tf = self._ensure_tf()
        model = self.get_model()
        
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        predictions = model(inputs_tensor, training=False)
        
        return predictions.numpy()
    
    def save_architecture(self, path: str):
        """
        Save model architecture to JSON file.
        
        Args:
            path: Path to save the JSON file
        """
        model = self.get_model()
        model_json = model.to_json()
        
        with open(path, 'w') as f:
            f.write(model_json)
    
    @classmethod
    def load_architecture(cls, path: str) -> "TFModelWrapper":
        """
        Load model architecture from JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            TFModelWrapper with the loaded model (no weights)
        """
        import tensorflow as tf
        
        with open(path, 'r') as f:
            model_json = f.read()
        
        model = tf.keras.models.model_from_json(model_json)
        return cls(model=model)
    
    def clone_architecture(self) -> "TFModelWrapper":
        """
        Create a new wrapper with the same model architecture but fresh weights.
        
        Returns:
            New TFModelWrapper with cloned architecture
        """
        tf = self._ensure_tf()
        model = self.get_model()
        
        # Clone via JSON
        model_json = model.to_json()
        cloned_model = tf.keras.models.model_from_json(model_json)
        
        return TFModelWrapper(model=cloned_model)

