"""
Row scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class RowScaling(BaseScaling):
    """Row scaling using row norms."""
    
    def __init__(self, norm_type="inf"):
        """
        Initialize row scaling.
        
        Args:
            norm_type: Type of norm to use ("inf", "1", "2")
        """
        super().__init__()
        self.norm_type = norm_type
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b by row norms.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Compute row norms
        row_norms = self.compute_row_norms(A, self.norm_type)
        
        # Create scaling factors (inverse of norms)
        # Avoid division by zero
        scale_factors = self._create_scale_factors(row_norms)
        
        # Scale matrix and vector
        scaled_A = self._scale_matrix(A, scale_factors)
        scaled_b = b * scale_factors
        
        # Return scaled matrix, vector, and scaling info
        return scaled_A, scaled_b, {"row_scale": scale_factors}
    
    def unscale(self, x, scale_info):
        """
        Return x unchanged (row scaling doesn't affect x).
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Original solution (unchanged)
        """
        # Row scaling doesn't affect the solution vector
        return x
    
    def scale_b_only(self, b, scale_info):
        """
        Scale the right-hand side vector using the precomputed scaling factors.
        
        Args:
            b: Right-hand side vector
            scale_info: Scaling information
            
        Returns:
            scaled_b: Scaled right-hand side
        """
        row_scale = scale_info.get("row_scale")
        if row_scale is not None:
            return b * row_scale
        return b
    
    def _create_scale_factors(self, row_norms, epsilon=1e-10):
        """
        Create scaling factors from row norms.
        
        Args:
            row_norms: Array of row norms
            epsilon: Small value to avoid division by zero
            
        Returns:
            scale_factors: Array of scaling factors
        """
        # Avoid division by zero by adding a small epsilon
        scale_factors = 1.0 / (row_norms + epsilon)
        
        # If any row has all zeros, set scaling factor to 1.0
        scale_factors[row_norms == 0] = 1.0
        
        return scale_factors
    
    def _scale_matrix(self, A, scale_factors):
        """
        Scale matrix A by row scaling factors.
        
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
            return D @ A
    
    def _scale_sparse_matrix(self, A, scale_factors):
        """
        Scale sparse matrix A by row scaling factors.
        
        Args:
            A: Sparse matrix
            scale_factors: Array of scaling factors
            
        Returns:
            scaled_A: Scaled sparse matrix
        """
        # Convert to CSR if not already
        if hasattr(A, "tocsr"):
            A = A.tocsr()
            
        # Copy the matrix (avoid modifying the original)
        scaled_A = A.copy()
        
        # For each row, scale the data
        for i in range(A.shape[0]):
            start, end = A.indptr[i], A.indptr[i+1]
            if start < end:
                scaled_A.data[start:end] *= scale_factors[i]
                
        return scaled_A
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"RowScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Row scaling using {self.norm_type}-norm of each row."