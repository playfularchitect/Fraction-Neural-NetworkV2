
"""
Training Script for Exact Fractional vs Floating-Point Neural Networks
Trains both networks on MNIST and logs comprehensive results.
"""

import argparse
import json
import time
import os
from fractions import Fraction
import random
import numpy as np
from typing import List, Dict, Any

# Import our custom networks
from frac_net import FractionalNeuralNetwork
from float_net import FloatingPointNeuralNetwork
from fraction_tensor import FractionTensor

# Import PyTorch for MNIST data loading
import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist_data(num_samples: int = 5000, seed: int = 42):
    """
    Load and preprocess MNIST data.
    
    Args:
        num_samples: Number of samples to use (for faster training)
        seed: Random seed for reproducible data selection
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Convert to numpy arrays and select subset
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # Process training data
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    
    for i in indices[:num_samples]:
        image, label = train_dataset[i]
        # Flatten image and normalize to [0, 1]
        flattened = image.view(-1).numpy()
        train_data.append(flattened)
        
        # One-hot encode labels
        one_hot = np.zeros(10)
        one_hot[label] = 1
        train_labels.append(one_hot)
    
    # Process test data (smaller subset)
    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)
    
    for i in test_indices[:num_samples // 5]:  # Use 1/5 of training size for testing
        image, label = test_dataset[i]
        flattened = image.view(-1).numpy()
        test_data.append(flattened)
        
        one_hot = np.zeros(10)
        one_hot[label] = 1
        test_labels.append(one_hot)
    
    return (np.array(train_data), np.array(train_labels), 
            np.array(test_data), np.array(test_labels))

def convert_to_fractions(data: np.ndarray) -> List[FractionTensor]:
    """Convert numpy arrays to FractionTensor format."""
    result = []
    for sample in data:
        # Convert each float to a fraction with reasonable denominator
        frac_data = [Fraction(float(val)).limit_denominator(10000) for val in sample]
        result.append(FractionTensor(frac_data))
    return result

def train_fractional_network(train_data: List[FractionTensor], 
                           train_labels: List[FractionTensor],
                           test_data: List[FractionTensor],
                           test_labels: List[FractionTensor],
                           epochs: int = 5,
                           learning_rate: float = 0.01,
                           seed: int = 42) -> Dict[str, Any]:
    """Train the fractional neural network and return results."""
    
    # Initialize network
    network = FractionalNeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        learning_rate=learning_rate,
        seed=seed
    )
    
    results = {
        'network_type': 'fractional',
        'epochs': epochs,
        'learning_rate': learning_rate,
        'seed': seed,
        'train_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'epoch_times': [],
        'parameter_summary': network.get_parameters_summary()
    }
    
    print(f"Training Fractional Network (seed={seed})...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        correct_predictions = 0
        
        # Shuffle training data
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        # Training loop
        for i, idx in enumerate(indices):
            x = train_data[idx]
            y = train_labels[idx]
            
            # Training step
            loss = network.train_step(x, y)
            epoch_losses.append(float(loss.data))
            
            # Check prediction accuracy
            prediction = network.predict(x)
            pred_class = np.argmax(prediction.to_float())
            true_class = np.argmax(y.to_float())
            
            if pred_class == true_class:
                correct_predictions += 1
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, Sample {i+1}/{len(indices)}, "
                      f"Loss: {np.mean(epoch_losses[-100:]):.6f}")
        
        # Calculate metrics
        avg_loss = np.mean(epoch_losses)
        train_accuracy = correct_predictions / len(train_data)
        
        # Test accuracy
        test_correct = 0
        for i in range(len(test_data)):
            prediction = network.predict(test_data[i])
            pred_class = np.argmax(prediction.to_float())
            true_class = np.argmax(test_labels[i].to_float())
            
            if pred_class == true_class:
                test_correct += 1
        
        test_accuracy = test_correct / len(test_data)
        epoch_time = time.time() - epoch_start
        
        # Store results
        results['train_losses'].append(avg_loss)
        results['train_accuracies'].append(train_accuracy)
        results['test_accuracies'].append(test_accuracy)
        results['epoch_times'].append(epoch_time)
        
        print(f"  Epoch {epoch+1} completed: Loss={avg_loss:.6f}, "
              f"Train Acc={train_accuracy:.4f}, Test Acc={test_accuracy:.4f}, "
              f"Time={epoch_time:.2f}s")
    
    # Final parameter analysis
    final_summary = network.get_parameters_summary()
    results['final_parameter_summary'] = final_summary
    
    print(f"Fractional network training completed!")
    print(f"Max denominator: {final_summary['max_denominator']}")
    
    return results

def train_floating_network(train_data: np.ndarray, 
                         train_labels: np.ndarray,
                         test_data: np.ndarray,
                         test_labels: np.ndarray,
                         epochs: int = 5,
                         learning_rate: float = 0.01,
                         seed: int = 42) -> Dict[str, Any]:
    """Train the floating-point neural network and return results."""
    
    # Initialize network
    network = FloatingPointNeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        learning_rate=learning_rate,
        seed=seed
    )
    
    results = {
        'network_type': 'floating_point',
        'epochs': epochs,
        'learning_rate': learning_rate,
        'seed': seed,
        'train_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'epoch_times': [],
        'parameter_summary': network.get_parameters_summary()
    }
    
    print(f"Training Floating-Point Network (seed={seed})...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        correct_predictions = 0
        
        # Shuffle training data
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        # Training loop
        for i, idx in enumerate(indices):
            x = train_data[idx]
            y = train_labels[idx]
            
            # Training step
            loss = network.train_step(x, y)
            epoch_losses.append(loss)
            
            # Check prediction accuracy
            prediction = network.predict(x)
            pred_class = np.argmax(prediction)
            true_class = np.argmax(y)
            
            if pred_class == true_class:
                correct_predictions += 1
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, Sample {i+1}/{len(indices)}, "
                      f"Loss: {np.mean(epoch_losses[-100:]):.6f}")
        
        # Calculate metrics
        avg_loss = np.mean(epoch_losses)
        train_accuracy = correct_predictions / len(train_data)
        
        # Test accuracy
        test_correct = 0
        for i in range(len(test_data)):
            prediction = network.predict(test_data[i])
            pred_class = np.argmax(prediction)
            true_class = np.argmax(test_labels[i])
            
            if pred_class == true_class:
                test_correct += 1
        
        test_accuracy = test_correct / len(test_data)
        epoch_time = time.time() - epoch_start
        
        # Store results
        results['train_losses'].append(avg_loss)
        results['train_accuracies'].append(train_accuracy)
        results['test_accuracies'].append(test_accuracy)
        results['epoch_times'].append(epoch_time)
        
        print(f"  Epoch {epoch+1} completed: Loss={avg_loss:.6f}, "
              f"Train Acc={train_accuracy:.4f}, Test Acc={test_accuracy:.4f}, "
              f"Time={epoch_time:.2f}s")
    
    print(f"Floating-point network training completed!")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train exact fractional vs floating-point neural networks')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--runs', type=int, default=3, help='Number of independent runs')
    parser.add_argument('--samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    
    print("=== Exact Fractional vs Floating-Point Neural Network Comparison ===")
    print(f"Configuration: {args.epochs} epochs, {args.runs} runs, {args.samples} samples")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data once
    print("Loading MNIST data...")
    train_data_np, train_labels_np, test_data_np, test_labels_np = load_mnist_data(
        num_samples=args.samples, seed=42
    )
    
    # Convert to fractions for fractional network
    print("Converting data to fractions...")
    train_data_frac = convert_to_fractions(train_data_np)
    train_labels_frac = convert_to_fractions(train_labels_np)
    test_data_frac = convert_to_fractions(test_data_np)
    test_labels_frac = convert_to_fractions(test_labels_np)
    
    all_results = []
    
    # Run multiple experiments
    for run in range(args.runs):
        print(f"\n--- Run {run + 1}/{args.runs} ---")
        
        # Use different seeds for each run to test reproducibility
        base_seed = 42 + run * 1000
        
        # Train fractional network
        frac_results = train_fractional_network(
            train_data_frac, train_labels_frac,
            test_data_frac, test_labels_frac,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=base_seed
        )
        frac_results['run'] = run
        all_results.append(frac_results)
        
        # Train floating-point network
        float_results = train_floating_network(
            train_data_np, train_labels_np,
            test_data_np, test_labels_np,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=base_seed
        )
        float_results['run'] = run
        all_results.append(float_results)
        
        # Save individual run results
        with open(f'results/run_{run}_fractional.json', 'w') as f:
            json.dump(frac_results, f, indent=2, default=str)
        
        with open(f'results/run_{run}_floating.json', 'w') as f:
            json.dump(float_results, f, indent=2, default=str)
    
    # Save combined results
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n=== Training Complete ===")
    print("Results saved to results/ directory")
    print("Run 'python analysis.py' to generate plots and analysis report")

if __name__ == "__main__":
    main()
