"""
Array operations for scaling that work across different backends (NumPy, CuPy, JAX).
This utility abstracts away backend-specific implementations.
"""

import numpy as np
import scipy.sparse as sp_cpu
import importlib.util
from typing import Union, Tuple, Any, Dict, Optional

# Check for optional backends
has_cupy = importlib.util.find_spec("cupy") is not None
has_jax = importlib.util.find_spec("jax") is not None

# Import backends if available
if has_cupy:
    import cupy as cp
    import cupyx.scipy.sparse as sp_gpu
else:
    cp = None
    sp_gpu = None

if has_jax:
    import jax
    import jax.numpy as jnp
else:
    jax = None
    jnp = None


class ArrayBackend:
    """Array operation backend that abstracts NumPy, CuPy, and JAX differences."""
    
    def __init__(self, backend='numpy'):
        """
        Initialize with the specified backend.
        
        Args:
            backend: 'numpy', 'cupy', or 'jax'
        """
        self.backend = backend.lower()
        
        # Fallback if requested backend isn't available
        if self.backend == 'cupy' and not has_cupy:
            print("Warning: CuPy requested but not available. Falling back to NumPy.")
            self.backend = 'numpy'
        
        if self.backend == 'jax' and not has_jax:
            print("Warning: JAX requested but not available. Falling back to NumPy.")
            self.backend = 'numpy'
    
    def get_array_module(self):
        """Get the appropriate array module for the current backend."""
        if self.backend == 'cupy' and has_cupy:
            return cp
        elif self.backend == 'jax' and has_jax:
            return jnp
        else:
            return np
    
    def get_sparse_module(self):
        """Get the appropriate sparse module for the current backend."""
        if self.backend == 'cupy' and has_cupy:
            return sp_gpu
        else:
            return sp_cpu
    
    def zeros(self, shape, dtype=None):
        """Create a zeros array with the current backend."""
        xp = self.get_array_module()
        return xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Create a ones array with the current backend."""
        xp = self.get_array_module()
        return xp.ones(shape, dtype=dtype)
    
    def max(self, a, axis=None):
        """Compute maximum with the current backend."""
        xp = self.get_array_module()
        return xp.max(a, axis=axis)
    
    def min(self, a, axis=None):
        """Compute minimum with the current backend."""
        xp = self.get_array_module()
        return xp.min(a, axis=axis)
    
    def abs(self, a):
        """Compute absolute value with the current backend."""
        xp = self.get_array_module()
        return xp.abs(a)
    
    def where(self, condition, x, y):
        """Compute where operation with the current backend."""
        xp = self.get_array_module()
        return xp.where(condition, x, y)
    
    def linalg_norm(self, a, ord=None, axis=None):
        """Compute norm with the current backend."""
        xp = self.get_array_module()
        return xp.linalg.norm(a, ord=ord, axis=axis)
    
    def isfinite(self, a):
        """Check if values are finite with the current backend."""
        xp = self.get_array_module()
        return xp.isfinite(a)
    
    def isnan(self, a):
        """Check if values are NaN with the current backend."""
        xp = self.get_array_module()
        return xp.isnan(a)
    
    def sqrt(self, a):
        """Compute square root with the current backend."""
        xp = self.get_array_module()
        return xp.sqrt(a)
    
    def power(self, a, exponent):
        """Compute power with the current backend."""
        xp = self.get_array_module()
        return xp.power(a, exponent)
    
    def mean(self, a, axis=None):
        """Compute mean with the current backend."""
        xp = self.get_array_module()
        return xp.mean(a, axis=axis)
    
    def sum(self, a, axis=None):
        """Compute sum with the current backend."""
        xp = self.get_array_module()
        return xp.sum(a, axis=axis)
    
    def diags(self, diagonal_array):
        """Create a diagonal sparse matrix with the current backend."""
        sp = self.get_sparse_module()
        return sp.diags(diagonal_array)
    
    def to_numpy(self, array):
        """Convert array to NumPy if needed."""
        if self.backend == 'numpy':
            return array
        elif self.backend == 'cupy' and has_cupy:
            return array.get() if hasattr(array, 'get') else array
        elif self.backend == 'jax' and has_jax:
            return np.array(array)
        return array
    
    def from_numpy(self, array):
        """Convert NumPy array to the current backend format."""
        if self.backend == 'numpy':
            return array
        elif self.backend == 'cupy' and has_cupy:
            return cp.array(array)
        elif self.backend == 'jax' and has_jax:
            return jnp.array(array)
        return array
    
    def copy_matrix(self, matrix):
        """Copy a matrix to the current backend format."""
        if hasattr(matrix, 'copy'):
            return matrix.copy()
        # Fall back to simpler copy if no copy method
        return matrix
