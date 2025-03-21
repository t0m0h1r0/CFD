"""
Jacobi scaling implementation (diagonal preconditioning).
"""

import numpy as np
from .base import BaseScaling


class JacobiScaling(BaseScaling):
    """Jacobi scaling using the diagonal elements of the matrix."""
    
    def __init__(self, lambda_min=1e-10):
        """
        Initialize Jacobi scaling.
        
        Args:
            lambda_min: Minimum value for diagonal elements to avoid division by zero
        """
        super().__init__()
        self.lambda_min = lambda_min
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b using Jacobi scaling.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Extract diagonal elements
        diag = self._extract_diagonal(A)
        
        # Create scaling factors (avoid division by zero)
        scale_factors = self._create_scale_factors(diag)
        
        # Scale matrix and vector
        scaled_A = self._scale_matrix(A, scale_factors)
        scaled_b = b * scale_factors
        
        # Return scaled matrix, vector, and scaling info
        return scaled_A, scaled_b, {"diag_scale": scale_factors}
    
    def unscale(self, x, scale_info):
        """
        Return x unchanged (Jacobi scaling doesn't affect x).
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Original solution (unchanged)
        """
        # Jacobi scaling doesn't affect the solution vector
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
        diag_scale = scale_info.get("diag_scale")
        if diag_scale is not None:
            return b * diag_scale
        return b
    
    def _extract_diagonal(self, A):
        """
        Extract diagonal elements of matrix A.
        
        Args:
            A: Matrix
            
        Returns:
            diag: Diagonal elements
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            if hasattr(A, "diagonal"):
                # Native diagonal extraction
                return A.diagonal()
            elif hasattr(A, "toarray"):
                # Convert to dense for diagonal
                return np.diag(A.toarray())
            else:
                # Extract manually
                n = min(A.shape)
                diag = np.zeros(n)
                for i in range(n):
                    diag[i] = A[i, i]
                return diag
        else:
            # Handle dense matrix
            return np.diag(A)
    
    def _create_scale_factors(self, diag):
        """
        Create scaling factors from diagonal elements.
        
        Args:
            diag: Diagonal elements
            
        Returns:
            scale_factors: Array of scaling factors
        """
        # Compute inverse square root of diagonal elements
        # Avoid division by zero by adding a small epsilon
        abs_diag = np.abs(diag)
        abs_diag = np.maximum(abs_diag, self.lambda_min)
        
        # Create scaling factors (1/sqrt(|diag|))
        scale_factors = 1.0 / np.sqrt(abs_diag)
        
        return scale_factors
    
    def _scale_matrix(self, A, scale_factors):
        """
        Scale matrix A using Jacobi scaling.
        
        Args:
            A: Matrix
            scale_factors: Scaling factors
            
        Returns:
            scaled_A: Scaled matrix
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently (D^(-1/2) * A * D^(-1/2))
            from scipy import sparse
            D = sparse.diags(scale_factors)
            
            # Need both CSR and CSC format for efficiency
            if hasattr(A, "tocsr"):
                A_csr = A.tocsr()
            else:
                A_csr = A
                
            # D * A * D
            return D @ A_csr @ D
        else:
            # Handle dense matrix
            D = np.diag(scale_factors)
            return D @ A @ D
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return "JacobiScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return "Jacobi scaling using diagonal elements of the matrix."