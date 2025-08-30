#!/usr/bin/env python3
"""
Minimal Demonstration of Exact Fractional vs Floating-Point Arithmetic
Shows key concepts without heavy computation.
"""

import numpy as np
from fractions import Fraction
import time

def demonstrate_exact_vs_approximate():
    """Show the difference between exact and approximate arithmetic."""
    print("=== Exact vs Approximate Arithmetic ===")
    
    # Example 1: Simple fraction operations
    print("\n1. Basic Arithmetic:")
    a_frac = Fraction(1, 3)
    b_frac = Fraction(1, 6)
    result_frac = a_frac + b_frac
    
    a_float = 1/3
    b_float = 1/6
    result_float = a_float + b_float
    
    print(f"Exact:       1/3 + 1/6 = {result_frac} = {float(result_frac)}")
    print(f"Approximate: {a_float} + {b_float} = {result_float}")
    print(f"Difference:  {abs(float(result_frac) - result_float):.2e}")
    
    # Example 2: Accumulation of errors
    print("\n2. Error Accumulation:")
    sum_frac = Fraction(0)
    sum_float = 0.0
    
    for i in range(1000):
        val_frac = Fraction(1, 7)  # Exact 1/7
        val_float = 1/7            # Approximate 1/7
        
        sum_frac += val_frac
        sum_float += val_float
    
    expected = Fraction(1000, 7)
    print(f"Expected:    1000 × (1/7) = {expected} = {float(expected)}")
    print(f"Exact sum:   {sum_frac} = {float(sum_frac)}")
    print(f"Float sum:   {sum_float}")
    print(f"Exact error: {abs(float(expected) - float(sum_frac)):.2e}")
    print(f"Float error: {abs(float(expected) - sum_float):.2e}")

def demonstrate_reproducibility():
    """Show perfect reproducibility with exact arithmetic."""
    print("\n=== Reproducibility Demonstration ===")
    
    def compute_with_fractions(seed):
        np.random.seed(seed)
        values = [Fraction(x).limit_denominator(1000) for x in np.random.random(10)]
        
        # Simulate some computation
        result = Fraction(0)
        for val in values:
            result += val * val - val / Fraction(3)
        
        return result
    
    def compute_with_floats(seed):
        np.random.seed(seed)
        values = np.random.random(10)
        
        # Same computation with floats
        result = 0.0
        for val in values:
            result += val * val - val / 3.0
        
        return result
    
    print("Running identical computations with same seed...")
    
    # Multiple runs with fractions
    frac_results = []
    for run in range(3):
        result = compute_with_fractions(42)
        frac_results.append(result)
        print(f"Fraction run {run+1}: {float(result):.15f}")
    
    # Multiple runs with floats
    float_results = []
    for run in range(3):
        result = compute_with_floats(42)
        float_results.append(result)
        print(f"Float run {run+1}:    {result:.15f}")
    
    # Check reproducibility
    frac_identical = all(r == frac_results[0] for r in frac_results)
    float_identical = all(abs(r - float_results[0]) < 1e-15 for r in float_results)
    
    print(f"\nFraction reproducibility: {'PERFECT' if frac_identical else 'IMPERFECT'}")
    print(f"Float reproducibility:    {'PERFECT' if float_identical else 'IMPERFECT'}")

def demonstrate_network_operations():
    """Show exact vs approximate neural network operations."""
    print("\n=== Neural Network Operations ===")
    
    # Simple matrix multiplication
    print("Matrix multiplication example:")
    
    # Exact version
    W_frac = [[Fraction(1, 2), Fraction(1, 3)], 
              [Fraction(1, 4), Fraction(1, 5)]]
    x_frac = [Fraction(2, 3), Fraction(3, 4)]
    
    # Compute W @ x exactly
    result_frac = []
    for i in range(len(W_frac)):
        dot_product = Fraction(0)
        for j in range(len(x_frac)):
            dot_product += W_frac[i][j] * x_frac[j]
        result_frac.append(dot_product)
    
    # Approximate version
    W_float = [[0.5, 1/3], [0.25, 0.2]]
    x_float = [2/3, 0.75]
    
    result_float = []
    for i in range(len(W_float)):
        dot_product = 0.0
        for j in range(len(x_float)):
            dot_product += W_float[i][j] * x_float[j]
        result_float.append(dot_product)
    
    print("Exact result:")
    for i, val in enumerate(result_frac):
        print(f"  Output {i+1}: {val} = {float(val):.10f}")
    
    print("Approximate result:")
    for i, val in enumerate(result_float):
        print(f"  Output {i+1}: {val:.10f}")
    
    print("Differences:")
    for i in range(len(result_frac)):
        diff = abs(float(result_frac[i]) - result_float[i])
        print(f"  Output {i+1}: {diff:.2e}")

def demonstrate_activation_functions():
    """Show exact vs approximate activation functions."""
    print("\n=== Activation Functions ===")
    
    # ReLU is exact for both
    print("ReLU (exact for both):")
    test_vals_frac = [Fraction(-1, 2), Fraction(0), Fraction(3, 4)]
    test_vals_float = [-0.5, 0.0, 0.75]
    
    for i, (frac_val, float_val) in enumerate(zip(test_vals_frac, test_vals_float)):
        relu_frac = max(frac_val, Fraction(0))
        relu_float = max(float_val, 0.0)
        print(f"  Input {i+1}: {float(frac_val):.3f} -> Exact: {float(relu_frac):.3f}, Float: {relu_float:.3f}")
    
    # Softmax approximation
    print("\nSoftmax (approximate for fractions):")
    inputs_frac = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)]
    inputs_float = [0.5, 1/3, 0.25]
    
    # Simple rational approximation for exp(x) ≈ 1 + x + x²/2
    def rational_exp_approx(x):
        return Fraction(1) + x + x*x/Fraction(2)
    
    # Exact softmax approximation
    exp_frac = [rational_exp_approx(x) for x in inputs_frac]
    sum_exp_frac = sum(exp_frac)
    softmax_frac = [e / sum_exp_frac for e in exp_frac]
    
    # Float softmax
    exp_float = [np.exp(x) for x in inputs_float]
    sum_exp_float = sum(exp_float)
    softmax_float = [e / sum_exp_float for e in exp_float]
    
    print("Rational approximation:")
    for i, val in enumerate(softmax_frac):
        print(f"  Output {i+1}: {float(val):.6f}")
    
    print("Standard softmax:")
    for i, val in enumerate(softmax_float):
        print(f"  Output {i+1}: {val:.6f}")

def performance_analysis():
    """Analyze performance differences."""
    print("\n=== Performance Analysis ===")
    
    # Time fraction operations
    print("Timing arithmetic operations...")
    
    # Fraction operations
    start_time = time.time()
    result_frac = Fraction(0)
    for i in range(1000):
        a = Fraction(i, 1000)
        b = Fraction(i+1, 1000)
        result_frac += a * b + a / (b + Fraction(1))
    frac_time = time.time() - start_time
    
    # Float operations
    start_time = time.time()
    result_float = 0.0
    for i in range(1000):
        a = i / 1000.0
        b = (i + 1) / 1000.0
        result_float += a * b + a / (b + 1.0)
    float_time = time.time() - start_time
    
    print(f"Fraction operations: {frac_time:.4f} seconds")
    print(f"Float operations:    {float_time:.4f} seconds")
    print(f"Speed ratio:         {frac_time/float_time:.1f}× slower")
    
    # Memory usage approximation
    frac_val = Fraction(123456789, 987654321)
    float_val = 123456789 / 987654321
    
    print(f"\nMemory usage (approximate):")
    print(f"Fraction: {frac_val} (numerator + denominator)")
    print(f"Float:    {float_val} (8 bytes)")

def main():
    """Run all demonstrations."""
    print("Exact Fractional Neural Networks - Core Concepts Demonstration")
    print("=" * 70)
    
    demonstrate_exact_vs_approximate()
    demonstrate_reproducibility()
    demonstrate_network_operations()
    demonstrate_activation_functions()
    performance_analysis()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("✓ Exact arithmetic eliminates floating-point errors")
    print("✓ Perfect reproducibility across multiple runs")
    print("✓ Mathematical transparency in all operations")
    print("✓ Significant computational overhead (10-100× slower)")
    print("✓ Higher memory usage due to rational number storage")
    print("✓ Suitable for research requiring exact reproducibility")

if __name__ == "__main__":
    main()
