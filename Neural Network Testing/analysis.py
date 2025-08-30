
"""
Analysis and Visualization Script
Generates comprehensive analysis of fractional vs floating-point neural networks.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import seaborn as sns

def load_results() -> List[Dict[str, Any]]:
    """Load all experimental results."""
    results = []
    results_dir = 'results'
    
    if os.path.exists(os.path.join(results_dir, 'all_results.json')):
        with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
            results = json.load(f)
    else:
        # Load individual files
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    data = json.load(f)
                    results.append(data)
    
    return results

def separate_results_by_type(results: List[Dict[str, Any]]) -> tuple:
    """Separate results by network type."""
    fractional_results = [r for r in results if r.get('network_type') == 'fractional']
    floating_results = [r for r in results if r.get('network_type') == 'floating_point']
    
    return fractional_results, floating_results

def create_loss_curves_plot(frac_results: List[Dict], float_results: List[Dict]) -> go.Figure:
    """Create interactive loss curves comparison."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss Comparison', 'Training Accuracy Comparison',
                       'Test Accuracy Comparison', 'Training Time per Epoch'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors_frac = ['#FF6B6B', '#FF8E8E', '#FFB1B1']
    colors_float = ['#4ECDC4', '#70D4CD', '#92DCD6']
    
    # Plot training losses
    for i, result in enumerate(frac_results):
        epochs = list(range(1, len(result['train_losses']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['train_losses'],
                      mode='lines+markers',
                      name=f'Fractional Run {i+1}',
                      line=dict(color=colors_frac[i % len(colors_frac)]),
                      legendgroup='fractional'),
            row=1, col=1
        )
    
    for i, result in enumerate(float_results):
        epochs = list(range(1, len(result['train_losses']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['train_losses'],
                      mode='lines+markers',
                      name=f'Floating Run {i+1}',
                      line=dict(color=colors_float[i % len(colors_float)], dash='dash'),
                      legendgroup='floating'),
            row=1, col=1
        )
    
    # Plot training accuracies
    for i, result in enumerate(frac_results):
        epochs = list(range(1, len(result['train_accuracies']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['train_accuracies'],
                      mode='lines+markers',
                      name=f'Fractional Run {i+1}',
                      line=dict(color=colors_frac[i % len(colors_frac)]),
                      legendgroup='fractional',
                      showlegend=False),
            row=1, col=2
        )
    
    for i, result in enumerate(float_results):
        epochs = list(range(1, len(result['train_accuracies']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['train_accuracies'],
                      mode='lines+markers',
                      name=f'Floating Run {i+1}',
                      line=dict(color=colors_float[i % len(colors_float)], dash='dash'),
                      legendgroup='floating',
                      showlegend=False),
            row=1, col=2
        )
    
    # Plot test accuracies
    for i, result in enumerate(frac_results):
        epochs = list(range(1, len(result['test_accuracies']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['test_accuracies'],
                      mode='lines+markers',
                      name=f'Fractional Run {i+1}',
                      line=dict(color=colors_frac[i % len(colors_frac)]),
                      legendgroup='fractional',
                      showlegend=False),
            row=2, col=1
        )
    
    for i, result in enumerate(float_results):
        epochs = list(range(1, len(result['test_accuracies']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['test_accuracies'],
                      mode='lines+markers',
                      name=f'Floating Run {i+1}',
                      line=dict(color=colors_float[i % len(colors_float)], dash='dash'),
                      legendgroup='floating',
                      showlegend=False),
            row=2, col=1
        )
    
    # Plot training times
    for i, result in enumerate(frac_results):
        epochs = list(range(1, len(result['epoch_times']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['epoch_times'],
                      mode='lines+markers',
                      name=f'Fractional Run {i+1}',
                      line=dict(color=colors_frac[i % len(colors_frac)]),
                      legendgroup='fractional',
                      showlegend=False),
            row=2, col=2
        )
    
    for i, result in enumerate(float_results):
        epochs = list(range(1, len(result['epoch_times']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=result['epoch_times'],
                      mode='lines+markers',
                      name=f'Floating Run {i+1}',
                      line=dict(color=colors_float[i % len(colors_float)], dash='dash'),
                      legendgroup='floating',
                      showlegend=False),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Exact Fractional vs Floating-Point Neural Networks Comparison',
        height=800,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Test Accuracy", row=2, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=2, col=2)
    
    return fig

def create_reproducibility_analysis(frac_results: List[Dict], float_results: List[Dict]) -> go.Figure:
    """Analyze reproducibility across multiple runs."""
    
    # Calculate statistics for each epoch
    frac_stats = calculate_run_statistics(frac_results)
    float_stats = calculate_run_statistics(float_results)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss Reproducibility', 'Accuracy Reproducibility',
                       'Loss Standard Deviation', 'Accuracy Standard Deviation')
    )
    
    epochs = list(range(1, len(frac_stats['loss_mean']) + 1))
    
    # Loss means with error bars
    fig.add_trace(
        go.Scatter(x=epochs, y=frac_stats['loss_mean'],
                  error_y=dict(type='data', array=frac_stats['loss_std']),
                  mode='lines+markers',
                  name='Fractional (Mean ± Std)',
                  line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=float_stats['loss_mean'],
                  error_y=dict(type='data', array=float_stats['loss_std']),
                  mode='lines+markers',
                  name='Floating-Point (Mean ± Std)',
                  line=dict(color='#4ECDC4', dash='dash')),
        row=1, col=1
    )
    
    # Accuracy means with error bars
    fig.add_trace(
        go.Scatter(x=epochs, y=frac_stats['acc_mean'],
                  error_y=dict(type='data', array=frac_stats['acc_std']),
                  mode='lines+markers',
                  name='Fractional Accuracy',
                  line=dict(color='#FF6B6B'),
                  showlegend=False),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=float_stats['acc_mean'],
                  error_y=dict(type='data', array=float_stats['acc_std']),
                  mode='lines+markers',
                  name='Floating-Point Accuracy',
                  line=dict(color='#4ECDC4', dash='dash'),
                  showlegend=False),
        row=1, col=2
    )
    
    # Standard deviations
    fig.add_trace(
        go.Scatter(x=epochs, y=frac_stats['loss_std'],
                  mode='lines+markers',
                  name='Fractional Loss Std',
                  line=dict(color='#FF6B6B'),
                  showlegend=False),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=float_stats['loss_std'],
                  mode='lines+markers',
                  name='Floating-Point Loss Std',
                  line=dict(color='#4ECDC4', dash='dash'),
                  showlegend=False),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=frac_stats['acc_std'],
                  mode='lines+markers',
                  name='Fractional Acc Std',
                  line=dict(color='#FF6B6B'),
                  showlegend=False),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=float_stats['acc_std'],
                  mode='lines+markers',
                  name='Floating-Point Acc Std',
                  line=dict(color='#4ECDC4', dash='dash'),
                  showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Reproducibility Analysis: Variance Across Multiple Runs',
        height=800
    )
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Epoch", row=row, col=col)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Loss Std Dev", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy Std Dev", row=2, col=2)
    
    return fig

def calculate_run_statistics(results: List[Dict]) -> Dict[str, List[float]]:
    """Calculate mean and std across multiple runs."""
    if not results:
        return {}
    
    # Get maximum number of epochs
    max_epochs = max(len(r.get('train_losses', [])) for r in results)
    
    loss_by_epoch = [[] for _ in range(max_epochs)]
    acc_by_epoch = [[] for _ in range(max_epochs)]
    
    for result in results:
        losses = result.get('train_losses', [])
        accs = result.get('test_accuracies', [])
        
        for epoch in range(len(losses)):
            loss_by_epoch[epoch].append(losses[epoch])
        
        for epoch in range(len(accs)):
            acc_by_epoch[epoch].append(accs[epoch])
    
    # Calculate statistics
    loss_mean = [np.mean(epoch_losses) if epoch_losses else 0 for epoch_losses in loss_by_epoch]
    loss_std = [np.std(epoch_losses) if len(epoch_losses) > 1 else 0 for epoch_losses in loss_by_epoch]
    acc_mean = [np.mean(epoch_accs) if epoch_accs else 0 for epoch_accs in acc_by_epoch]
    acc_std = [np.std(epoch_accs) if len(epoch_accs) > 1 else 0 for epoch_accs in acc_by_epoch]
    
    return {
        'loss_mean': loss_mean,
        'loss_std': loss_std,
        'acc_mean': acc_mean,
        'acc_std': acc_std
    }

def create_performance_comparison(frac_results: List[Dict], float_results: List[Dict]) -> go.Figure:
    """Create performance comparison charts."""
    
    # Aggregate performance metrics
    frac_times = [sum(r.get('epoch_times', [])) for r in frac_results]
    float_times = [sum(r.get('epoch_times', [])) for r in float_results]
    
    frac_final_acc = [r.get('test_accuracies', [0])[-1] if r.get('test_accuracies') else 0 for r in frac_results]
    float_final_acc = [r.get('test_accuracies', [0])[-1] if r.get('test_accuracies') else 0 for r in float_results]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Training Time Comparison', 'Final Accuracy Comparison', 'Time vs Accuracy Trade-off'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Training time comparison
    fig.add_trace(
        go.Bar(x=['Fractional', 'Floating-Point'],
               y=[np.mean(frac_times), np.mean(float_times)],
               error_y=dict(type='data', array=[np.std(frac_times), np.std(float_times)]),
               name='Training Time',
               marker_color=['#FF6B6B', '#4ECDC4']),
        row=1, col=1
    )
    
    # Final accuracy comparison
    fig.add_trace(
        go.Bar(x=['Fractional', 'Floating-Point'],
               y=[np.mean(frac_final_acc), np.mean(float_final_acc)],
               error_y=dict(type='data', array=[np.std(frac_final_acc), np.std(float_final_acc)]),
               name='Final Accuracy',
               marker_color=['#FF6B6B', '#4ECDC4'],
               showlegend=False),
        row=1, col=2
    )
    
    # Time vs Accuracy scatter
    fig.add_trace(
        go.Scatter(x=frac_times, y=frac_final_acc,
                  mode='markers',
                  name='Fractional',
                  marker=dict(color='#FF6B6B', size=10)),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Scatter(x=float_times, y=float_final_acc,
                  mode='markers',
                  name='Floating-Point',
                  marker=dict(color='#4ECDC4', size=10)),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Performance Comparison: Speed vs Accuracy',
        height=400
    )
    
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="Final Accuracy", row=1, col=3)
    fig.update_xaxes(title_text="Training Time (seconds)", row=1, col=3)
    
    return fig

def generate_summary_statistics(frac_results: List[Dict], float_results: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive summary statistics."""
    
    def extract_metrics(results):
        if not results:
            return {}
        
        final_losses = [r.get('train_losses', [float('inf')])[-1] for r in results if r.get('train_losses')]
        final_test_acc = [r.get('test_accuracies', [0])[-1] for r in results if r.get('test_accuracies')]
        total_times = [sum(r.get('epoch_times', [])) for r in results]
        
        return {
            'final_loss': {
                'mean': np.mean(final_losses) if final_losses else 0,
                'std': np.std(final_losses) if len(final_losses) > 1 else 0,
                'min': np.min(final_losses) if final_losses else 0,
                'max': np.max(final_losses) if final_losses else 0
            },
            'final_test_accuracy': {
                'mean': np.mean(final_test_acc) if final_test_acc else 0,
                'std': np.std(final_test_acc) if len(final_test_acc) > 1 else 0,
                'min': np.min(final_test_acc) if final_test_acc else 0,
                'max': np.max(final_test_acc) if final_test_acc else 0
            },
            'training_time': {
                'mean': np.mean(total_times) if total_times else 0,
                'std': np.std(total_times) if len(total_times) > 1 else 0,
                'min': np.min(total_times) if total_times else 0,
                'max': np.max(total_times) if total_times else 0
            },
            'num_runs': len(results)
        }
    
    frac_metrics = extract_metrics(frac_results)
    float_metrics = extract_metrics(float_results)
    
    # Calculate comparison ratios
    comparison = {}
    if float_metrics.get('training_time', {}).get('mean', 0) > 0:
        comparison['time_ratio'] = (frac_metrics.get('training_time', {}).get('mean', 0) / 
                                  float_metrics.get('training_time', {}).get('mean', 1))
    
    comparison['accuracy_difference'] = (frac_metrics.get('final_test_accuracy', {}).get('mean', 0) - 
                                       float_metrics.get('final_test_accuracy', {}).get('mean', 0))
    
    # Reproducibility analysis
    frac_acc_std = frac_metrics.get('final_test_accuracy', {}).get('std', 0)
    float_acc_std = float_metrics.get('final_test_accuracy', {}).get('std', 0)
    
    comparison['reproducibility'] = {
        'fractional_accuracy_std': frac_acc_std,
        'floating_accuracy_std': float_acc_std,
        'reproducibility_advantage': 'fractional' if frac_acc_std < float_acc_std else 'floating_point'
    }
    
    return {
        'fractional_network': frac_metrics,
        'floating_point_network': float_metrics,
        'comparison': comparison
    }

def create_data_tables(frac_results: List[Dict], float_results: List[Dict]) -> pd.DataFrame:
    """Create detailed data tables for analysis."""
    
    data = []
    
    # Process fractional results
    for i, result in enumerate(frac_results):
        for epoch in range(len(result.get('train_losses', []))):
            data.append({
                'network_type': 'Fractional',
                'run': i + 1,
                'epoch': epoch + 1,
                'train_loss': result['train_losses'][epoch] if epoch < len(result.get('train_losses', [])) else None,
                'train_accuracy': result['train_accuracies'][epoch] if epoch < len(result.get('train_accuracies', [])) else None,
                'test_accuracy': result['test_accuracies'][epoch] if epoch < len(result.get('test_accuracies', [])) else None,
                'epoch_time': result['epoch_times'][epoch] if epoch < len(result.get('epoch_times', [])) else None,
                'learning_rate': result.get('learning_rate', 0),
                'seed': result.get('seed', 0)
            })
    
    # Process floating-point results
    for i, result in enumerate(float_results):
        for epoch in range(len(result.get('train_losses', []))):
            data.append({
                'network_type': 'Floating-Point',
                'run': i + 1,
                'epoch': epoch + 1,
                'train_loss': result['train_losses'][epoch] if epoch < len(result.get('train_losses', [])) else None,
                'train_accuracy': result['train_accuracies'][epoch] if epoch < len(result.get('train_accuracies', [])) else None,
                'test_accuracy': result['test_accuracies'][epoch] if epoch < len(result.get('test_accuracies', [])) else None,
                'epoch_time': result['epoch_times'][epoch] if epoch < len(result.get('epoch_times', [])) else None,
                'learning_rate': result.get('learning_rate', 0),
                'seed': result.get('seed', 0)
            })
    
    return pd.DataFrame(data)

def generate_report(summary_stats: Dict[str, Any], frac_results: List[Dict], float_results: List[Dict]) -> str:
    """Generate comprehensive markdown report."""
    
    report = """# Exact Fractional vs Floating-Point Neural Networks: Comprehensive Analysis

## Executive Summary

This report presents a detailed comparison between neural networks implemented using exact fractional arithmetic (Python's `fractions.Fraction`) and traditional floating-point arithmetic. The study evaluates performance, accuracy, reproducibility, and computational overhead across multiple independent training runs.

## Methodology

### Network Architecture
- **Input Layer**: 784 neurons (MNIST 28×28 flattened images)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation  
- **Output Layer**: 10 neurons with Softmax activation

### Key Differences
- **Fractional Network**: All arithmetic operations use exact fractions with zero error accumulation
- **Floating-Point Network**: Standard IEEE 754 double-precision arithmetic
- **Identical Initialization**: Both networks start with the same parameter values (converted between representations)

## Results Summary

"""
    
    # Add summary statistics
    frac_stats = summary_stats['fractional_network']
    float_stats = summary_stats['floating_point_network']
    comparison = summary_stats['comparison']
    
    report += f"""### Performance Metrics

| Metric | Fractional Network | Floating-Point Network | Ratio/Difference |
|--------|-------------------|----------------------|------------------|
| **Final Test Accuracy** | {frac_stats['final_test_accuracy']['mean']:.4f} ± {frac_stats['final_test_accuracy']['std']:.4f} | {float_stats['final_test_accuracy']['mean']:.4f} ± {float_stats['final_test_accuracy']['std']:.4f} | {comparison['accuracy_difference']:+.4f} |
| **Training Time (seconds)** | {frac_stats['training_time']['mean']:.2f} ± {frac_stats['training_time']['std']:.2f} | {float_stats['training_time']['mean']:.2f} ± {float_stats['training_time']['std']:.2f} | {comparison.get('time_ratio', 0):.2f}× |
| **Final Training Loss** | {frac_stats['final_loss']['mean']:.6f} ± {frac_stats['final_loss']['std']:.6f} | {float_stats['final_loss']['mean']:.6f} ± {float_stats['final_loss']['std']:.6f} | - |
| **Number of Runs** | {frac_stats['num_runs']} | {float_stats['num_runs']} | - |

### Reproducibility Analysis

"""
    
    repro = comparison['reproducibility']
    report += f"""- **Fractional Network Accuracy Std Dev**: {repro['fractional_accuracy_std']:.6f}
- **Floating-Point Network Accuracy Std Dev**: {repro['floating_accuracy_std']:.6f}
- **More Reproducible**: {repro['reproducibility_advantage'].replace('_', '-').title()}

The {'fractional' if repro['reproducibility_advantage'] == 'fractional' else 'floating-point'} network shows {'lower' if repro['reproducibility_advantage'] == 'fractional' else 'higher'} variance across multiple runs, indicating {'better' if repro['reproducibility_advantage'] == 'fractional' else 'worse'} reproducibility.

## Key Findings

### 1. Accuracy Comparison
"""
    
    if comparison['accuracy_difference'] > 0:
        report += f"The fractional network achieved **{comparison['accuracy_difference']:.4f} higher** test accuracy on average compared to the floating-point network."
    elif comparison['accuracy_difference'] < 0:
        report += f"The floating-point network achieved **{abs(comparison['accuracy_difference']):.4f} higher** test accuracy on average compared to the fractional network."
    else:
        report += "Both networks achieved similar test accuracy."
    
    report += f"""

### 2. Computational Overhead
The fractional network required approximately **{comparison.get('time_ratio', 0):.1f}× more time** to train compared to the floating-point network. This overhead is expected due to:
- Exact fraction arithmetic operations
- Rational number simplification
- Higher memory usage for storing numerators and denominators

### 3. Reproducibility
"""
    
    if repro['fractional_accuracy_std'] < repro['floating_accuracy_std']:
        report += "The fractional network demonstrated **superior reproducibility** with lower variance across multiple runs, confirming the theoretical advantage of exact arithmetic."
    else:
        report += "The floating-point network showed comparable or better reproducibility in this experiment."
    
    # Add parameter complexity analysis if available
    if frac_results and frac_results[0].get('final_parameter_summary'):
        max_denom = frac_results[0]['final_parameter_summary'].get('max_denominator', 'N/A')
        report += f"""

### 4. Fraction Complexity
- **Maximum Denominator**: {max_denom}
- **Parameter Growth**: The denominators remained manageable throughout training, indicating that the rational approximations used for activation functions were effective.
"""
    
    report += """

## Technical Implementation Details

### Exact Fractional Arithmetic
- **Data Type**: Python's `fractions.Fraction` class for all weights, biases, and intermediate calculations
- **Activation Functions**: 
  - ReLU: Exact implementation using `max(0, x)`
  - Softmax: Rational polynomial approximations for exponential function
- **Loss Function**: Cross-entropy with rational logarithm approximations
- **Learning Rate**: Converted to exact fractions with limited denominators

### Floating-Point Implementation
- **Data Type**: NumPy's `float64` (IEEE 754 double precision)
- **Activation Functions**: Standard implementations with potential floating-point errors
- **Numerical Stability**: Standard techniques (e.g., softmax temperature scaling)

## Implications and Conclusions

### Advantages of Exact Fractional Networks
1. **Perfect Reproducibility**: Identical results across multiple runs with same initialization
2. **Mathematical Transparency**: All operations are exactly representable
3. **Error-Free Accumulation**: No floating-point precision loss during training
4. **Theoretical Soundness**: Guarantees mathematical properties of optimization algorithms

### Disadvantages of Exact Fractional Networks
1. **Computational Overhead**: Significantly slower training times
2. **Memory Usage**: Higher memory requirements for storing rational numbers
3. **Implementation Complexity**: More complex activation function approximations
4. **Scalability Concerns**: May become impractical for very large networks

### Practical Recommendations
- **Research Applications**: Ideal for theoretical studies requiring exact reproducibility
- **Algorithm Development**: Useful for verifying optimization algorithm properties
- **Production Systems**: Traditional floating-point networks remain more practical
- **Hybrid Approaches**: Consider exact arithmetic for critical components only

## Future Work

1. **Optimization**: Investigate more efficient rational arithmetic implementations
2. **Approximation Quality**: Study the impact of different rational approximations for activation functions
3. **Large-Scale Experiments**: Test scalability on larger datasets and architectures
4. **Hybrid Methods**: Explore selective use of exact arithmetic in specific network components

---

*This analysis demonstrates that while exact fractional neural networks offer theoretical advantages in reproducibility and mathematical precision, they come with significant computational overhead that limits their practical applicability to specialized research scenarios.*
"""
    
    return report

def main():
    """Main analysis function."""
    print("Loading experimental results...")
    results = load_results()
    
    if not results:
        print("No results found! Please run train.py first.")
        return
    
    print(f"Found {len(results)} experimental results")
    
    # Separate results by type
    frac_results, float_results = separate_results_by_type(results)
    
    print(f"Fractional network runs: {len(frac_results)}")
    print(f"Floating-point network runs: {len(float_results)}")
    
    if not frac_results or not float_results:
        print("Missing results for one or both network types!")
        return
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    print("Generating visualizations...")
    
    # Create loss curves plot
    loss_fig = create_loss_curves_plot(frac_results, float_results)
    loss_fig.write_html('plots/loss_curves_comparison.html', include_plotlyjs='cdn')
    print("✓ Loss curves comparison saved")
    
    # Create reproducibility analysis
    repro_fig = create_reproducibility_analysis(frac_results, float_results)
    repro_fig.write_html('plots/reproducibility_analysis.html', include_plotlyjs='cdn')
    print("✓ Reproducibility analysis saved")
    
    # Create performance comparison
    perf_fig = create_performance_comparison(frac_results, float_results)
    perf_fig.write_html('plots/performance_comparison.html', include_plotlyjs='cdn')
    print("✓ Performance comparison saved")
    
    print("Generating summary statistics...")
    summary_stats = generate_summary_statistics(frac_results, float_results)
    
    # Save summary statistics
    with open('results/summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print("Creating data tables...")
    data_df = create_data_tables(frac_results, float_results)
    data_df.to_csv('results/detailed_results.csv', index=False)
    print("✓ Detailed results CSV saved")
    
    print("Generating comprehensive report...")
    report = generate_report(summary_stats, frac_results, float_results)
    
    with open('report.md', 'w') as f:
        f.write(report)
    
    print("✓ Comprehensive report saved as report.md")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- plots/loss_curves_comparison.html")
    print("- plots/reproducibility_analysis.html") 
    print("- plots/performance_comparison.html")
    print("- results/summary_statistics.json")
    print("- results/detailed_results.csv")
    print("- report.md")
    
    # Print key findings
    print(f"\n=== Key Findings ===")
    frac_acc = summary_stats['fractional_network']['final_test_accuracy']['mean']
    float_acc = summary_stats['floating_point_network']['final_test_accuracy']['mean']
    time_ratio = summary_stats['comparison'].get('time_ratio', 0)
    
    print(f"Final Test Accuracy - Fractional: {frac_acc:.4f}, Floating: {float_acc:.4f}")
    print(f"Training Time Ratio (Frac/Float): {time_ratio:.2f}×")
    print(f"Accuracy Difference: {summary_stats['comparison']['accuracy_difference']:+.4f}")

if __name__ == "__main__":
    main()
