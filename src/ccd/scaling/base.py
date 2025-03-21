"""
Base class for matrix scaling implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional
import numpy as np


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
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        pass
    
    @abstractmethod
    def unscale(self, x, scale_info: Dict[str, Any]):
        """
        Unscale the solution vector using scaling information.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        pass
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """
        Scale only the right-hand side vector (optimization for multiple solves).
        
        Args:
            b: Right-hand side vector
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
    
    # Utility methods for subclasses
    
    def to_numpy(self, arr):
        """
        Convert array to NumPy array.
        
        Args:
            arr: Input array (NumPy or CuPy)
            
        Returns:
            numpy_arr: NumPy array
        """
        if arr is None:
            return None
        
        # Check if it's a CuPy array
        if hasattr(arr, 'get'):
            return arr.get()
        
        # Check if it's a JAX array
        if str(type(arr)).find('jax') >= 0:
            return np.array(arr)
        
        # Already a NumPy array
        return arr
    
    def is_sparse(self, A):
        """
        Check if A is a sparse matrix.
        
        Args:
            A: Input matrix
            
        Returns:
            is_sparse: True if A is a sparse matrix
        """
        return hasattr(A, 'format') or hasattr(A, 'tocsr') or hasattr(A, 'tocsc')
    
    def compute_row_norms(self, A, norm_type="inf"):
        """
        Compute row norms of matrix A.
        
        Args:
            A: Matrix
            norm_type: Type of norm to use ("inf", "1", "2")
            
        Returns:
            row_norms: Array of row norms
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            return self.compute_sparse_row_norms(A, norm_type)
        else:
            # Handle dense matrix
            if norm_type == "inf":
                return np.max(np.abs(A), axis=1)
            elif norm_type == "1":
                return np.sum(np.abs(A), axis=1)
            else:  # default to "2"
                return np.sqrt(np.sum(A * A, axis=1))
    
    def compute_sparse_row_norms(self, A, norm_type="inf"):
        """
        Compute row norms of sparse matrix A.
        
        Args:
            A: Sparse matrix
            norm_type: Type of norm to use ("inf", "1", "2")
            
        Returns:
            row_norms: Array of row norms
        """
        # Convert to CSR if not already
        if hasattr(A, "tocsr"):
            A = A.tocsr()
        
        n_rows = A.shape[0]
        row_norms = np.zeros(n_rows)
        
        # For each row
        for i in range(n_rows):
            # Get row slice
            start, end = A.indptr[i], A.indptr[i+1]
            if start < end:
                row_data = A.data[start:end]
                if norm_type == "inf":
                    row_norms[i] = np.max(np.abs(row_data))
                elif norm_type == "1":
                    row_norms[i] = np.sum(np.abs(row_data))
                else:  # default to "2"
                    row_norms[i] = np.sqrt(np.sum(row_data * row_data))
        
        return row_norms