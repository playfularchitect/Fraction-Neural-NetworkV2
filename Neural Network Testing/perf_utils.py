
"""
Performance Utilities for Memory and Timing Analysis
"""

import time
import psutil
import os
from functools import wraps
from typing import Dict, Any, Callable
import json

class PerformanceMonitor:
    """Monitor performance metrics during training."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.metrics = []
    
    def start_monitoring(self):
        """Start monitoring performance."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def record_metric(self, label: str, additional_data: Dict[str, Any] = None):
        """Record a performance metric."""
        current_time = time.perf_counter()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        metric = {
            'label': label,
            'timestamp': current_time,
            'elapsed_time': current_time - self.start_time if self.start_time else 0,
            'memory_mb': current_memory,
            'memory_delta_mb': current_memory - self.start_memory if self.start_memory else 0
        }
        
        if additional_data:
            metric.update(additional_data)
        
        self.metrics.append(metric)
        return metric
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        total_time = self.metrics[-1]['elapsed_time']
        max_memory = max(m['memory_mb'] for m in self.metrics)
        memory_growth = self.metrics[-1]['memory_delta_mb']
        
        return {
            'total_time_seconds': total_time,
            'max_memory_mb': max_memory,
            'memory_growth_mb': memory_growth,
            'num_measurements': len(self.metrics),
            'detailed_metrics': self.metrics
        }
    
    def save_metrics(self, filename: str):
        """Save metrics to file."""
        with open(filename, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

def time_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        # Store timing info in result if it's a dict
        if isinstance(result, dict):
            result['execution_time'] = execution_time
        
        return result
    return wrapper

def memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def compare_performance_metrics(frac_results: Dict[str, Any], 
                              float_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare performance metrics between fractional and floating-point networks."""
    
    comparison = {
        'fractional_network': {
            'total_time': sum(frac_results.get('epoch_times', [])),
            'avg_epoch_time': sum(frac_results.get('epoch_times', [])) / len(frac_results.get('epoch_times', [1])),
            'final_accuracy': frac_results.get('test_accuracies', [0])[-1] if frac_results.get('test_accuracies') else 0,
            'parameter_summary': frac_results.get('final_parameter_summary', {})
        },
        'floating_point_network': {
            'total_time': sum(float_results.get('epoch_times', [])),
            'avg_epoch_time': sum(float_results.get('epoch_times', [])) / len(float_results.get('epoch_times', [1])),
            'final_accuracy': float_results.get('test_accuracies', [0])[-1] if float_results.get('test_accuracies') else 0,
            'parameter_summary': float_results.get('parameter_summary', {})
        }
    }
    
    # Calculate ratios
    frac_time = comparison['fractional_network']['total_time']
    float_time = comparison['floating_point_network']['total_time']
    
    comparison['performance_ratios'] = {
        'time_ratio_frac_to_float': frac_time / float_time if float_time > 0 else float('inf'),
        'accuracy_difference': (comparison['fractional_network']['final_accuracy'] - 
                              comparison['floating_point_network']['final_accuracy'])
    }
    
    return comparison

class FractionComplexityAnalyzer:
    """Analyze the complexity of fractions in the network."""
    
    @staticmethod
    def analyze_fraction_tensor(tensor):
        """Analyze fraction complexity in a tensor."""
        if hasattr(tensor, 'data'):
            if isinstance(tensor.data, list):
                fractions = FractionComplexityAnalyzer._flatten_fractions(tensor.data)
            else:
                fractions = [tensor.data]
        else:
            return {}
        
        if not fractions:
            return {}
        
        denominators = [f.denominator for f in fractions if hasattr(f, 'denominator')]
        numerators = [abs(f.numerator) for f in fractions if hasattr(f, 'numerator')]
        
        return {
            'num_fractions': len(fractions),
            'max_denominator': max(denominators) if denominators else 0,
            'avg_denominator': sum(denominators) / len(denominators) if denominators else 0,
            'max_numerator': max(numerators) if numerators else 0,
            'avg_numerator': sum(numerators) / len(numerators) if numerators else 0,
            'complexity_score': max(denominators) * max(numerators) if denominators and numerators else 0
        }
    
    @staticmethod
    def _flatten_fractions(data):
        """Recursively flatten nested lists to get all fractions."""
        result = []
        if isinstance(data, list):
            for item in data:
                result.extend(FractionComplexityAnalyzer._flatten_fractions(item))
        else:
            result.append(data)
        return result
    
    @staticmethod
    def analyze_network_complexity(network):
        """Analyze fraction complexity across entire network."""
        total_analysis = {
            'weights': [],
            'biases': [],
            'overall': {
                'max_denominator': 0,
                'max_numerator': 0,
                'total_fractions': 0,
                'avg_complexity': 0
            }
        }
        
        # Analyze weights
        for i, weight_matrix in enumerate(network.weights):
            analysis = FractionComplexityAnalyzer.analyze_fraction_tensor(weight_matrix)
            analysis['layer'] = i
            total_analysis['weights'].append(analysis)
            
            # Update overall stats
            total_analysis['overall']['max_denominator'] = max(
                total_analysis['overall']['max_denominator'],
                analysis.get('max_denominator', 0)
            )
            total_analysis['overall']['max_numerator'] = max(
                total_analysis['overall']['max_numerator'],
                analysis.get('max_numerator', 0)
            )
            total_analysis['overall']['total_fractions'] += analysis.get('num_fractions', 0)
        
        # Analyze biases
        for i, bias_vector in enumerate(network.biases):
            analysis = FractionComplexityAnalyzer.analyze_fraction_tensor(bias_vector)
            analysis['layer'] = i
            total_analysis['biases'].append(analysis)
            
            # Update overall stats
            total_analysis['overall']['max_denominator'] = max(
                total_analysis['overall']['max_denominator'],
                analysis.get('max_denominator', 0)
            )
            total_analysis['overall']['max_numerator'] = max(
                total_analysis['overall']['max_numerator'],
                analysis.get('max_numerator', 0)
            )
            total_analysis['overall']['total_fractions'] += analysis.get('num_fractions', 0)
        
        # Calculate average complexity
        if total_analysis['overall']['total_fractions'] > 0:
            total_analysis['overall']['avg_complexity'] = (
                total_analysis['overall']['max_denominator'] * 
                total_analysis['overall']['max_numerator']
            ) / total_analysis['overall']['total_fractions']
        
        return total_analysis
