
"""
Exact Fractional Tensor Operations
Implements tensor-like operations using Python's fractions.Fraction for exact arithmetic.
"""

from fractions import Fraction
import random
from typing import List, Union, Tuple
import copy

class FractionTensor:
    """A tensor-like class that uses exact fractions for all operations."""
    
    def __init__(self, data: Union[List, List[List], int, float, Fraction]):
        """Initialize tensor from nested lists, scalars, or existing data."""
        if isinstance(data, (int, float, Fraction)):
            self.data = Fraction(data)
            self.shape = ()
        elif isinstance(data, list):
            if not data:
                raise ValueError("Cannot create tensor from empty list")
            
            # Convert all elements to Fraction recursively
            self.data = self._convert_to_fractions(data)
            self.shape = self._compute_shape(self.data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _convert_to_fractions(self, data):
        """Recursively convert nested lists to fractions."""
        if isinstance(data, list):
            return [self._convert_to_fractions(item) for item in data]
        else:
            return Fraction(data)
    
    def _compute_shape(self, data):
        """Compute the shape of nested list structure."""
        if not isinstance(data, list):
            return ()
        
        shape = [len(data)]
        if data and isinstance(data[0], list):
            inner_shape = self._compute_shape(data[0])
            shape.extend(inner_shape)
        
        return tuple(shape)
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...]):
        """Create a tensor filled with zeros."""
        if len(shape) == 0:
            return cls(Fraction(0))
        elif len(shape) == 1:
            return cls([Fraction(0)] * shape[0])
        elif len(shape) == 2:
            return cls([[Fraction(0) for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise NotImplementedError("Only 0D, 1D, and 2D tensors supported")
    
    @classmethod
    def random_normal(cls, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, seed: int = None):
        """Create a tensor with random normal-like values using rational approximations."""
        if seed is not None:
            random.seed(seed)
        
        def random_fraction():
            # Simple Box-Muller approximation using rational arithmetic
            u1 = Fraction(random.randint(1, 10000), 10000)
            u2 = Fraction(random.randint(1, 10000), 10000)
            
            # Approximate normal using central limit theorem with uniform random variables
            # Sum of 12 uniform(0,1) - 6 approximates normal(0,1)
            normal_approx = sum(Fraction(random.randint(0, 1000), 1000) for _ in range(12)) - Fraction(6)
            return Fraction(mean) + Fraction(std) * normal_approx
        
        if len(shape) == 0:
            return cls(random_fraction())
        elif len(shape) == 1:
            return cls([random_fraction() for _ in range(shape[0])])
        elif len(shape) == 2:
            return cls([[random_fraction() for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise NotImplementedError("Only 0D, 1D, and 2D tensors supported")
    
    def __add__(self, other):
        """Element-wise addition."""
        if isinstance(other, (int, float, Fraction)):
            other = FractionTensor(other)
        
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        
        if self.shape == ():
            return FractionTensor(self.data + other.data)
        elif len(self.shape) == 1:
            result = [self.data[i] + other.data[i] for i in range(len(self.data))]
            return FractionTensor(result)
        elif len(self.shape) == 2:
            result = [[self.data[i][j] + other.data[i][j] 
                      for j in range(len(self.data[i]))] 
                     for i in range(len(self.data))]
            return FractionTensor(result)
        else:
            raise NotImplementedError("Only 0D, 1D, and 2D tensors supported")
    
    def __sub__(self, other):
        """Element-wise subtraction."""
        if isinstance(other, (int, float, Fraction)):
            other = FractionTensor(other)
        
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        
        if self.shape == ():
            return FractionTensor(self.data - other.data)
        elif len(self.shape) == 1:
            result = [self.data[i] - other.data[i] for i in range(len(self.data))]
            return FractionTensor(result)
        elif len(self.shape) == 2:
            result = [[self.data[i][j] - other.data[i][j] 
                      for j in range(len(self.data[i]))] 
                     for i in range(len(self.data))]
            return FractionTensor(result)
        else:
            raise NotImplementedError("Only 0D, 1D, and 2D tensors supported")
    
    def __mul__(self, other):
        """Element-wise multiplication or scalar multiplication."""
        if isinstance(other, (int, float, Fraction)):
            other_frac = Fraction(other)
            if self.shape == ():
                return FractionTensor(self.data * other_frac)
            elif len(self.shape) == 1:
                result = [self.data[i] * other_frac for i in range(len(self.data))]
                return FractionTensor(result)
            elif len(self.shape) == 2:
                result = [[self.data[i][j] * other_frac 
                          for j in range(len(self.data[i]))] 
                         for i in range(len(self.data))]
                return FractionTensor(result)
        
        # Element-wise multiplication with another tensor
        if isinstance(other, FractionTensor):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            
            if self.shape == ():
                return FractionTensor(self.data * other.data)
            elif len(self.shape) == 1:
                result = [self.data[i] * other.data[i] for i in range(len(self.data))]
                return FractionTensor(result)
            elif len(self.shape) == 2:
                result = [[self.data[i][j] * other.data[i][j] 
                          for j in range(len(self.data[i]))] 
                         for i in range(len(self.data))]
                return FractionTensor(result)
        
        raise NotImplementedError("Unsupported multiplication")
    
    def matmul(self, other):
        """Matrix multiplication."""
        if not isinstance(other, FractionTensor):
            raise ValueError("Matrix multiplication requires FractionTensor")
        
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")
        
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} @ {other.shape}")
        
        rows, cols = self.shape[0], other.shape[1]
        inner_dim = self.shape[1]
        
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                # Dot product of row i from self with column j from other
                dot_product = Fraction(0)
                for k in range(inner_dim):
                    dot_product += self.data[i][k] * other.data[k][j]
                row.append(dot_product)
            result.append(row)
        
        return FractionTensor(result)
    
    def transpose(self):
        """Transpose a 2D tensor."""
        if len(self.shape) != 2:
            raise ValueError("Transpose only supported for 2D tensors")
        
        rows, cols = self.shape
        result = [[self.data[i][j] for i in range(rows)] for j in range(cols)]
        return FractionTensor(result)
    
    def sum(self, axis=None):
        """Sum along specified axis or all elements."""
        if axis is None:
            # Sum all elements
            if self.shape == ():
                return FractionTensor(self.data)
            elif len(self.shape) == 1:
                return FractionTensor(sum(self.data))
            elif len(self.shape) == 2:
                total = Fraction(0)
                for row in self.data:
                    for val in row:
                        total += val
                return FractionTensor(total)
        
        if len(self.shape) == 2:
            if axis == 0:
                # Sum along rows (result is 1D)
                cols = self.shape[1]
                result = [Fraction(0)] * cols
                for i in range(self.shape[0]):
                    for j in range(cols):
                        result[j] += self.data[i][j]
                return FractionTensor(result)
            elif axis == 1:
                # Sum along columns (result is 1D)
                result = [sum(row) for row in self.data]
                return FractionTensor(result)
        
        raise NotImplementedError(f"Sum with axis={axis} not implemented for shape {self.shape}")
    
    def relu(self):
        """ReLU activation function."""
        zero = Fraction(0)
        
        if self.shape == ():
            return FractionTensor(max(self.data, zero))
        elif len(self.shape) == 1:
            result = [max(val, zero) for val in self.data]
            return FractionTensor(result)
        elif len(self.shape) == 2:
            result = [[max(val, zero) for val in row] for row in self.data]
            return FractionTensor(result)
        else:
            raise NotImplementedError("ReLU only supported for 0D, 1D, and 2D tensors")
    
    def relu_derivative(self):
        """Derivative of ReLU function."""
        zero = Fraction(0)
        one = Fraction(1)
        
        if self.shape == ():
            return FractionTensor(one if self.data > zero else zero)
        elif len(self.shape) == 1:
            result = [one if val > zero else zero for val in self.data]
            return FractionTensor(result)
        elif len(self.shape) == 2:
            result = [[one if val > zero else zero for val in row] for row in self.data]
            return FractionTensor(result)
        else:
            raise NotImplementedError("ReLU derivative only supported for 0D, 1D, and 2D tensors")
    
    def softmax(self):
        """Softmax activation using rational approximations."""
        if len(self.shape) != 1:
            raise ValueError("Softmax only supported for 1D tensors")
        
        # Use a rational approximation for exp(x) ≈ (1 + x/n)^n for large n
        # For numerical stability, subtract max value first
        max_val = max(self.data)
        
        # Approximate exp using rational functions
        def rational_exp_approx(x):
            # For small x, use Taylor series: exp(x) ≈ 1 + x + x²/2 + x³/6
            # Truncated to keep denominators reasonable
            if abs(x) < Fraction(1, 10):
                return Fraction(1) + x + x*x/Fraction(2) + x*x*x/Fraction(6)
            else:
                # For larger x, use (1 + x/100)^100 approximation
                n = 100
                return (Fraction(1) + x/Fraction(n)) ** n
        
        # Compute softmax
        shifted = [val - max_val for val in self.data]
        exp_vals = [rational_exp_approx(val) for val in shifted]
        sum_exp = sum(exp_vals)
        
        result = [val / sum_exp for val in exp_vals]
        return FractionTensor(result)
    
    def to_float(self):
        """Convert to float for display/comparison purposes only."""
        if self.shape == ():
            return float(self.data)
        elif len(self.shape) == 1:
            return [float(val) for val in self.data]
        elif len(self.shape) == 2:
            return [[float(val) for val in row] for row in self.data]
        else:
            raise NotImplementedError("to_float only supported for 0D, 1D, and 2D tensors")
    
    def copy(self):
        """Create a deep copy of the tensor."""
        return FractionTensor(copy.deepcopy(self.data))
    
    def __repr__(self):
        return f"FractionTensor(shape={self.shape}, data={self.data})"
    
    def __str__(self):
        if self.shape == ():
            return str(float(self.data))
        elif len(self.shape) == 1:
            return str([float(val) for val in self.data])
        elif len(self.shape) == 2:
            return str([[float(val) for val in row] for row in self.data])
        else:
            return f"FractionTensor(shape={self.shape})"
