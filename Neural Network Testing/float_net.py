
"""
Traditional Floating-Point Neural Network Implementation
Identical architecture to the fractional network for comparison.
"""

import numpy as np
import random
from typing import List, Tuple

class FloatingPointNeuralNetwork:
    """A traditional neural network using floating-point arithmetic."""
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01, seed: int = None):
        """
        Initialize the network with given layer sizes.
        
        Args:
            layer_sizes: List of integers specifying the size of each layer
            learning_rate: Learning rate
            seed: Random seed for reproducible initialization
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize weights and biases using Xavier initialization
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # Xavier initialization
            std = np.sqrt(2.0 / (fan_in + fan_out))
            
            weight_matrix = np.random.normal(0, std, (fan_out, fan_in))
            self.weights.append(weight_matrix)
            
            bias_vector = np.zeros(fan_out)
            self.biases.append(bias_vector)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input array of shape (input_size,)
            
        Returns:
            output: Network output
            activations: List of activations at each layer
        """
        activations = [x]
        current = x
        
        for i in range(self.num_layers - 1):
            # Linear transformation: z = W @ a + b
            z = self.weights[i] @ current + self.biases[i]
            
            # Apply activation function
            if i < self.num_layers - 2:  # Hidden layers use ReLU
                current = np.maximum(0, z)
            else:  # Output layer uses softmax
                # Softmax with numerical stability
                exp_z = np.exp(z - np.max(z))
                current = exp_z / np.sum(exp_z)
            
            activations.append(current)
        
        return current, activations
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Network predictions (softmax output)
            targets: One-hot encoded targets
            
        Returns:
            Cross-entropy loss
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        loss = -np.sum(targets * np.log(predictions))
        return loss
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray, 
                activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass to compute gradients.
        
        Args:
            predictions: Network output
            targets: True targets (one-hot)
            activations: Activations from forward pass
            
        Returns:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        weight_gradients = []
        bias_gradients = []
        
        # Start with output layer error
        delta = predictions - targets
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            prev_activation = activations[i]
            
            # Weight gradient
            weight_grad = np.outer(delta, prev_activation)
            weight_gradients.insert(0, weight_grad)
            
            # Bias gradient
            bias_gradients.insert(0, delta.copy())
            
            # Compute delta for previous layer (if not input layer)
            if i > 0:
                delta_prev = self.weights[i].T @ delta
                
                # Apply activation derivative (ReLU derivative)
                relu_derivative = (activations[i] > 0).astype(float)
                delta = delta_prev * relu_derivative
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], 
                         bias_gradients: List[np.ndarray]):
        """Update weights and biases using gradients."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Perform one training step.
        
        Args:
            x: Input data
            y: Target labels (one-hot)
            
        Returns:
            Loss value
        """
        # Forward pass
        predictions, activations = self.forward(x)
        
        # Compute loss
        loss = self.compute_loss(predictions, y)
        
        # Backward pass
        weight_grads, bias_grads = self.backward(predictions, y, activations)
        
        # Update parameters
        self.update_parameters(weight_grads, bias_grads)
        
        return loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make a prediction without updating parameters."""
        predictions, _ = self.forward(x)
        return predictions
    
    def get_parameters_summary(self) -> dict:
        """Get a summary of network parameters for analysis."""
        total_params = sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
        
        return {
            'total_parameters': total_params,
            'learning_rate': self.learning_rate,
            'layer_sizes': self.layer_sizes
        }
