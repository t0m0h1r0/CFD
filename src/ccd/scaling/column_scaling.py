"""
Column scaling implementation for linear systems.
"""

import numpy as np
from .base import BaseScaling

class ColumnScaling(BaseScaling):
    """
    Column scaling for linear systems.
    A' = A D^(-1)
    where D = diag(||A_j||) for each column j
    """
    
    def __init__(self):
        super().__init__()
        
    @property
    def name(self):
        return "ColumnScaling"
        
    @property
    def description(self):
        return "Column scaling using column norms"
    
    def scale(self, A, b):
        """
        Scale matrix A using column norms
        
        Args:
            A: System matrix
            b: Right-hand side vector (not scaled in column scaling)
            
        Returns:
            Tuple of (scaled_A, b, scaling_info)
        """
        # Get column norms
        if hasattr(A, 'transpose') and hasattr(A.transpose(), 'power'):
            # Sparse matrix
            A_t = A.transpose()
            col_norms = A_t.power(2).sum(axis=0)
            if hasattr(col_norms, 'toarray'):
                col_norms = col_norms.toarray().flatten()
            col_norms = np.sqrt(col_norms)
        else:
            # Dense matrix
            col_norms = np.sqrt(np.sum(A**2, axis=0))
        
        # Handle zero columns
        col_norms[col_norms < 1e-15] = 1.0
        
        # Compute D^(-1) for scaling
        col_scale = 1.0 / col_norms
        
        # Scale matrix: A' = A D^(-1)
        # For each col j: A[:,j] = A[:,j] / ||A_j||
        if hasattr(A, 'multiply'):
            # Create column scaling matrix (diagonal)
            from scipy import sparse
            D_inv = sparse.diags(col_scale)
            A_scaled = A @ D_inv
        else:
            # Scale dense matrix by columns
            A_scaled = A.copy()
            for j in range(A.shape[1]):
                A_scaled[:, j] = A_scaled[:, j] * col_scale[j]
        
        # Right-hand side is not scaled in column scaling
        
        # Store scaling information for unscaling
        scaling_info = {
            'col_scale': col_scale
        }
        
        return A_scaled, b, scaling_info
    
    def unscale(self, x, scale_info):
        """
        Unscale solution vector
        
        Args:
            x: Solution vector of the scaled system
            scale_info: Scaling information
            
        Returns:
            Unscaled solution vector
        """
        col_scale = scale_info.get('col_scale')
        if col_scale is None:
            return x
            
        # For column scaling, the solution must be unscaled by multiplying
        # with the inverse of the column scaling
        # Since we scaled as A' = A D^(-1), the solution x' of A'x' = b
        # is related to the original solution x as x = D^(-1) x'
        return x * col_scale
    
    def scale_b_only(self, b, scale_info):
        """
        Scale only right-hand side (for multiple solves)
        
        Args:
            b: Right-hand side vector
            scale_info: Scaling information
            
        Returns:
            Right-hand side (unchanged for column scaling)
        """
        # Column scaling doesn't affect b
        return b