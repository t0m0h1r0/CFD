"""
Column scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class ColumnScaling(BaseScaling):
    """Column scaling using column norms."""
    
    def __init__(self, norm_type="inf"):
        """
        Initialize column scaling.
        
        Args:
            norm_type: Type of norm to use ("inf", "1", "2")
        """
        super().__init__()
        self.norm_type = norm_type
    
    def scale(self, A, b):
        """
        Scale matrix A by column norms. Unlike row scaling, this changes the solution vector.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Compute column norms
        col_norms = self._compute_column_norms(A, self.norm_type)
        
        # Create scaling factors (inverse of norms)
        # Avoid division by zero
        scale_factors = self._create_scale_factors(col_norms)
        
        # Scale matrix but NOT vector
        scaled_A = self._scale_matrix(A, scale_factors)
        
        # Return scaled matrix, unchanged vector, and scaling info
        return scaled_A, b, {"col_scale": scale_factors}
    
    def unscale(self, x, scale_info):
        """
        Unscale the solution vector using the column scaling factors.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        # With column scaling, we need to unscale the solution
        col_scale = scale_info.get("col_scale")
        if col_scale is not None:
            return x / col_scale
        return x
    
    def scale_b_only(self, b, scale_info):
        """
        Column scaling doesn't scale b, so return unchanged.
        
        Args:
            b: Right-hand side vector
            scale_info: Scaling information
            
        Returns:
            scaled_b: Original right-hand side (unchanged)
        """
        return b
    
    def _compute_column_norms(self, A, norm_type="inf"):
        """
        Compute column norms of matrix A.
        
        Args:
            A: Matrix
            norm_type: Type of norm to use ("inf", "1", "2")
            
        Returns:
            col_norms: Array of column norms
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            return self._compute_sparse_column_norms(A, norm_type)
        else:
            # Handle dense matrix
            if norm_type == "inf":
                return np.max(np.abs(A), axis=0)
            elif norm_type == "1":
                return np.sum(np.abs(A), axis=0)
            else:  # default to "2"
                return np.sqrt(np.sum(A * A, axis=0))
    
    def _compute_sparse_column_norms(self, A, norm_type="inf"):
        """
        Compute column norms of sparse matrix A.
        
        Args:
            A: Sparse matrix
            norm_type: Type of norm to use ("inf", "1", "2")
            
        Returns:
            col_norms: Array of column norms
        """
        # Convert to CSC for column operations
        if hasattr(A, "tocsc"):
            A = A.tocsc()
        
        n_cols = A.shape[1]
        col_norms = np.zeros(n_cols)
        
        # For each column
        for j in range(n_cols):
            # Get column slice
            start, end = A.indptr[j], A.indptr[j+1]
            if start < end:
                col_data = A.data[start:end]
                if norm_type == "inf":
                    col_norms[j] = np.max(np.abs(col_data))
                elif norm_type == "1":
                    col_norms[j] = np.sum(np.abs(col_data))
                else:  # default to "2"
                    col_norms[j] = np.sqrt(np.sum(col_data * col_data))
        
        return col_norms
    
    def _create_scale_factors(self, col_norms, epsilon=1e-10):
        """
        Create scaling factors from column norms.
        
        Args:
            col_norms: Array of column norms
            epsilon: Small value to avoid division by zero
            
        Returns:
            scale_factors: Array of scaling factors
        """
        # Avoid division by zero by adding a small epsilon
        scale_factors = 1.0 / (col_norms + epsilon)
        
        # If any column has all zeros, set scaling factor to 1.0
        scale_factors[col_norms == 0] = 1.0
        
        return scale_factors
    
    def _scale_matrix(self, A, scale_factors):
        """
        Scale matrix A by column scaling factors.
        
        Args:
            A: System matrix
            scale_factors: Array of scaling factors
            
        Returns:
            scaled_A: Scaled matrix
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            return self._scale_sparse_matrix(A, scale_factors)
        else:
            # Handle dense matrix
            # Create a diagonal matrix with scaling factors
            D = np.diag(scale_factors)
            return A @ D
    
    def _scale_sparse_matrix(self, A, scale_factors):
        """
        Scale sparse matrix A by column scaling factors.
        
        Args:
            A: Sparse matrix
            scale_factors: Array of scaling factors
            
        Returns:
            scaled_A: Scaled sparse matrix
        """
        # Convert to CSC for column operations
        if hasattr(A, "tocsc"):
            A = A.tocsc()
            
        # Copy the matrix (avoid modifying the original)
        scaled_A = A.copy()
        
        # For each column, scale the data
        for j in range(A.shape[1]):
            start, end = A.indptr[j], A.indptr[j+1]
            if start < end:
                scaled_A.data[start:end] *= scale_factors[j]
                
        return scaled_A
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"ColumnScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Column scaling using {self.norm_type}-norm of each column."