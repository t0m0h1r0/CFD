"""
Base class for matrix scaling implementations with array-agnostic operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseScaling(ABC):
    """Base class for matrix scaling techniques."""
    
    def __init__(self):
        """
        Initialize scaling implementation.
        """
        pass
    
    @abstractmethod
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Scale matrix A and right-hand side vector b.
        
        Args:
            A: System matrix (NumPy or CuPy)
            b: Right-hand side vector (NumPy or CuPy)
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        pass
    
    @abstractmethod
    def unscale(self, x, scale_info: Dict[str, Any]):
        """
        Unscale the solution vector using scaling information.
        
        Args:
            x: Solution vector (NumPy or CuPy)
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        pass
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """
        Scale only the right-hand side vector (optimization for multiple solves).
        
        Args:
            b: Right-hand side vector (NumPy or CuPy)
            scale_info: Scaling information
            
        Returns:
            scaled_b: Scaled right-hand side
        """
        # Default implementation: use row scale if available
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
            
        # Otherwise, each subclass should implement efficiently
        return b
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the scaling method."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of the scaling method."""
        pass
    
    # Utility methods to work with both NumPy and CuPy arrays
    
    def _get_array_module(self, x):
        """
        Get the array module (numpy or cupy) for the given array x.
        
        This allows for transparent handling of both NumPy and CuPy arrays.
        
        Args:
            x: NumPy or CuPy array
            
        Returns:
            module: NumPy or CuPy module
        """
        # Check if input is a CuPy array by looking for 'device' attribute
        if hasattr(x, 'device') and hasattr(x, 'get'):
            import cupy
            return cupy
        else:
            import numpy
            return numpy
    
    def _is_sparse(self, A):
        """
        Check if matrix A is sparse (in any format).
        
        Args:
            A: Matrix (NumPy or CuPy)
            
        Returns:
            bool: True if A is a sparse matrix
        """
        return hasattr(A, 'format')
    
    def _maximum(self, x, min_value):
        """
        Array-agnostic element-wise maximum.
        
        Args:
            x: NumPy or CuPy array
            min_value: Minimum value to enforce
            
        Returns:
            Array with element-wise maximum applied
        """
        xp = self._get_array_module(x)
        return xp.maximum(x, min_value)
    
    def _abs(self, x):
        """
        Array-agnostic absolute value.
        
        Args:
            x: NumPy or CuPy array
            
        Returns:
            Array with absolute values
        """
        xp = self._get_array_module(x)
        return xp.abs(x)
    
    def _ones_like(self, x):
        """
        Array-agnostic ones like.
        
        Args:
            x: NumPy or CuPy array
            
        Returns:
            Array of ones with same shape and type as x
        """
        xp = self._get_array_module(x)
        return xp.ones_like(x)
    
    def _zeros(self, shape, dtype=None, array_ref=None):
        """
        Array-agnostic zeros.
        
        Args:
            shape: Shape of the array
            dtype: Data type (optional)
            array_ref: Reference array to determine module (NumPy or CuPy)
            
        Returns:
            Array of zeros
        """
        if array_ref is not None:
            xp = self._get_array_module(array_ref)
        else:
            import numpy as xp
        
        return xp.zeros(shape, dtype=dtype)
    
    def _diags(self, v, shape=None):
        """
        Array-agnostic diagonal matrix.
        
        Args:
            v: Diagonal values (NumPy or CuPy array)
            shape: Shape of the resulting matrix (optional)
            
        Returns:
            Sparse diagonal matrix
        """
        xp = self._get_array_module(v)
        
        if hasattr(xp, 'sparse'):
            # CuPy or SciPy sparse
            return xp.sparse.diags(v)
        else:
            # Fall back to SciPy sparse for NumPy
            import scipy.sparse as sp
            return sp.diags(v)