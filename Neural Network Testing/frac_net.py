
"""
Exact Fractional Neural Network Implementation
A feedforward neural network using exact fraction arithmetic.
"""

from fractions import Fraction
from fraction_tensor import FractionTensor
import random
import copy
from typing import List, Tuple

class FractionalNeuralNetwork:
    """A neural network that uses exact fraction arithmetic throughout."""
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01, seed: int = None):
        """
        Initialize the network with given layer sizes.
        
        Args:
            layer_sizes: List of integers specifying the size of each layer
            learning_rate: Learning rate as a float (will be converted to Fraction)
            seed: Random seed for reproducible initialization
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = Fraction(learning_rate).limit_denominator(10000)
        self.num_layers = len(layer_sizes)
        
        if seed is not None:
            random.seed(seed)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization adapted for fractions
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # Use a simple rational approximation for Xavier initialization
            # std = sqrt(2 / (fan_in + fan_out)) ≈ rational approximation
            std_approx = Fraction(1, max(1, int((fan_in + fan_out) / 2)))
            
            # Initialize weights with small random fractions
            weight_matrix = []
            for j in range(fan_out):
                row = []
                for k in range(fan_in):
                    # Random value between -std_approx and +std_approx
                    rand_val = Fraction(random.randint(-1000, 1000), 10000) * std_approx
                    row.append(rand_val)
                weight_matrix.append(row)
            
            self.weights.append(FractionTensor(weight_matrix))
            
            # Initialize biases to zero
            bias_vector = [Fraction(0) for _ in range(fan_out)]
            self.biases.append(FractionTensor(bias_vector))
    
    def forward(self, x: FractionTensor) -> Tuple[FractionTensor, List[FractionTensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (input_size,)
            
        Returns:
            output: Network output
            activations: List of activations at each layer (for backprop)
        """
        activations = [x]
        current = x
        
        for i in range(self.num_layers - 1):
            # Linear transformation: z = W @ a + b
            z = self.weights[i].matmul(current.transpose() if len(current.shape) == 2 else 
                                     FractionTensor([[val] for val in current.data]))
            
            # Add bias
            if len(z.shape) == 2:
                # z is (output_size, 1), convert to 1D and add bias
                z_1d = FractionTensor([z.data[j][0] for j in range(len(z.data))])
            else:
                z_1d = z
            
            z_with_bias = z_1d + self.biases[i]
            
            # Apply activation function
            if i < self.num_layers - 2:  # Hidden layers use ReLU
                current = z_with_bias.relu()
            else:  # Output layer uses softmax
                current = z_with_bias.softmax()
            
            activations.append(current)
        
        return current, activations
    
    def compute_loss(self, predictions: FractionTensor, targets: FractionTensor) -> FractionTensor:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Network predictions (softmax output)
            targets: One-hot encoded targets
            
        Returns:
            Cross-entropy loss
        """
        # Cross-entropy: -sum(y * log(p))
        # Use rational approximation for log: log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x near 1
        
        loss = Fraction(0)
        for i in range(len(predictions.data)):
            if targets.data[i] > Fraction(0):  # Only compute for true class
                p = predictions.data[i]
                
                # Rational log approximation (Taylor series around 1)
                if p > Fraction(1, 1000):  # Avoid log(0)
                    x_minus_1 = p - Fraction(1)
                    log_approx = x_minus_1 - x_minus_1*x_minus_1/Fraction(2) + x_minus_1*x_minus_1*x_minus_1/Fraction(3)
                    loss -= targets.data[i] * log_approx
                else:
                    # Large penalty for very small probabilities
                    loss += Fraction(10)
        
        return FractionTensor(loss)
    
    def backward(self, predictions: FractionTensor, targets: FractionTensor, 
                activations: List[FractionTensor]) -> Tuple[List[FractionTensor], List[FractionTensor]]:
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
        # For softmax + cross-entropy: dL/dz = predictions - targets
        delta = predictions - targets
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients for current layer
            prev_activation = activations[i]
            
            # Weight gradient: dL/dW = delta @ prev_activation^T
            if len(prev_activation.shape) == 1:
                # Convert to column vector for matrix multiplication
                prev_act_col = FractionTensor([[val] for val in prev_activation.data])
                delta_row = FractionTensor([delta.data])
                weight_grad = delta_row.transpose().matmul(prev_act_col.transpose())
            else:
                weight_grad = delta.matmul(prev_activation.transpose())
            
            weight_gradients.insert(0, weight_grad)
            
            # Bias gradient: dL/db = delta
            bias_gradients.insert(0, delta.copy())
            
            # Compute delta for previous layer (if not input layer)
            if i > 0:
                # delta_prev = W^T @ delta * activation_derivative
                delta_prev = self.weights[i].transpose().matmul(
                    FractionTensor([[val] for val in delta.data])
                )
                
                # Convert back to 1D
                delta_prev_1d = FractionTensor([delta_prev.data[j][0] for j in range(len(delta_prev.data))])
                
                # Apply activation derivative (ReLU derivative for hidden layers)
                if i > 0:
                    # For ReLU: derivative is 1 if input > 0, else 0
                    # We need the pre-activation values, but we'll approximate using post-activation
                    relu_deriv = activations[i].relu_derivative()
                    delta = delta_prev_1d * relu_deriv
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[FractionTensor], 
                         bias_gradients: List[FractionTensor]):
        """Update weights and biases using gradients."""
        for i in range(len(self.weights)):
            # Update weights: W = W - learning_rate * dW
            self.weights[i] = self.weights[i] - weight_gradients[i] * self.learning_rate
            
            # Update biases: b = b - learning_rate * db
            self.biases[i] = self.biases[i] - bias_gradients[i] * self.learning_rate
    
    def train_step(self, x: FractionTensor, y: FractionTensor) -> FractionTensor:
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
    
    def predict(self, x: FractionTensor) -> FractionTensor:
        """Make a prediction without updating parameters."""
        predictions, _ = self.forward(x)
        return predictions
    
    def get_parameters_summary(self) -> dict:
        """Get a summary of network parameters for analysis."""
        total_params = 0
        max_denominator = 1
        
        for weight_matrix in self.weights:
            if len(weight_matrix.shape) == 2:
                total_params += weight_matrix.shape[0] * weight_matrix.shape[1]
                for row in weight_matrix.data:
                    for val in row:
                        max_denominator = max(max_denominator, val.denominator)
        
        for bias_vector in self.biases:
            total_params += len(bias_vector.data)
            for val in bias_vector.data:
                max_denominator = max(max_denominator, val.denominator)
        
        return {
            'total_parameters': total_params,
            'max_denominator': max_denominator,
            'learning_rate': self.learning_rate,
            'layer_sizes': self.layer_sizes
        }
