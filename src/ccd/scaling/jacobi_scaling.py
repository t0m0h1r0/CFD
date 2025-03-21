"""
Jacobi scaling (diagonal scaling) implementation for linear systems.
"""

import numpy as np
from .base import BaseScaling

class JacobiScaling(BaseScaling):
    """
    Jacobi scaling (diagonal scaling) for linear systems.
    A' = D^(-1/2) A D^(-1/2)
    b' = D^(-1/2) b
    where D = diag(A)
    """
    
    def __init__(self):
        super().__init__()
        
    @property
    def name(self):
        return "JacobiScaling"
        
    @property
    def description(self):
        return "Jacobi scaling using diagonal elements"
    
    def scale(self, A, b):
        """
        Scale matrix A and vector b using Jacobi scaling
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Tuple of (scaled_A, scaled_b, scaling_info)
        """
        # Get diagonal elements
        if hasattr(A, 'diagonal'):
            # Sparse matrix
            diag = A.diagonal()
        else:
            # Dense matrix
            diag = np.diag(A)
        
        # Handle zeros in diagonal to avoid division by zero
        diag_abs = np.abs(diag)
        diag_abs[diag_abs < 1e-15] = 1.0
        
        # Compute D^(-1/2)
        D_sqrt_inv = 1.0 / np.sqrt(diag_abs)
        
        # Store original signs to handle negative diagonals correctly
        signs = np.ones_like(diag)
        signs[diag < 0] = -1.0
        
        D_sqrt_inv_with_sign = D_sqrt_inv * np.sqrt(signs)
        
        # Scale matrix: A' = D^(-1/2) A D^(-1/2)
        # For each row i: A[i,:] = A[i,:] * D_sqrt_inv
        # For each col j: A[:,j] = A[:,j] * D_sqrt_inv
        D_sqrt_inv_mat = D_sqrt_inv_with_sign[:, np.newaxis]
        
        if hasattr(A, 'multiply') and hasattr(A, 'transpose'):
            # Sparse matrix operations
            # Scale rows
            A_scaled = A.multiply(D_sqrt_inv_mat)
            # Scale columns
            A_scaled = A_scaled.multiply(D_sqrt_inv_with_sign)
        else:
            # Dense matrix operations
            A_scaled = A.copy()
            # Scale rows
            A_scaled = A_scaled * D_sqrt_inv_mat
            # Scale columns
            A_scaled = A_scaled * D_sqrt_inv_with_sign
            
        # Scale right-hand side: b' = D^(-1/2) b
        b_scaled = b * D_sqrt_inv_with_sign
        
        # Store scaling information for unscaling
        scaling_info = {
            'D_sqrt_inv': D_sqrt_inv_with_sign,
            'signs': signs
        }
        
        return A_scaled, b_scaled, scaling_info
    
    def unscale(self, x, scale_info):
        """
        Unscale solution vector
        
        Args:
            x: Solution vector of the scaled system
            scale_info: Scaling information
            
        Returns:
            Unscaled solution vector
        """
        D_sqrt_inv = scale_info.get('D_sqrt_inv')
        if D_sqrt_inv is None:
            return x
            
        # Apply correct unscaling: x = D^(-1/2) x'
        # Since we scaled as A' = D^(-1/2) A D^(-1/2) and b' = D^(-1/2) b
        # The original system Ax = b becomes A'x' = b' where x' = D^(1/2) x
        # Therefore x = D^(-1/2) x'
        return x * D_sqrt_inv
    
    def scale_b_only(self, b, scale_info):
        """
        Scale only right-hand side (for multiple solves)
        
        Args:
            b: Right-hand side vector
            scale_info: Scaling information
            
        Returns:
            Scaled right-hand side
        """
        D_sqrt_inv = scale_info.get('D_sqrt_inv')
        if D_sqrt_inv is None:
            return b
            
        return b * D_sqrt_inv