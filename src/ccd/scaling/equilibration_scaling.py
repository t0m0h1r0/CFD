"""
Equilibration scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class EquilibrationScaling(BaseScaling):
    """Equilibration scaling for numerical stability."""
    
    def __init__(self, epsilon=1e-10):
        """
        Initialize equilibration scaling.
        
        Args:
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b using equilibration.
        This scaling attempts to make all rows have similar norms,
        and all columns have similar norms, to improve the condition number.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Get matrix dimensions
        n_rows, n_cols = A.shape
        
        # Compute row and column norms
        row_norms = self.compute_row_norms(A, norm_type="inf")
        col_norms = self._compute_column_norms(A, norm_type="inf")
        
        # Create row and column scaling factors
        row_scale = self._create_scale_factors(row_norms)
        col_scale = self._create_scale_factors(col_norms)
        
        # Apply scaling: D_r * A * D_c where D_r and D_c are diagonal matrices
        scaled_A = self._scale_matrix(A, row_scale, col_scale)
        
        # Scale right-hand side: D_r * b
        scaled_b = b * row_scale
        
        # Return scaled matrix, vector, and scaling info
        return scaled_A, scaled_b, {"row_scale": row_scale, "col_scale": col_scale}
    
    def unscale(self, x, scale_info):
        """
        Unscale the solution vector.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        # Recover original solution: D_c^(-1) * x
        col_scale = scale_info.get("col_scale")
        if col_scale is not None:
            return x / col_scale
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
            # Convert to CSC for column operations
            if hasattr(A, "tocsc"):
                A_csc = A.tocsc()
            else:
                A_csc = A
                
            n_cols = A.shape[1]
            col_norms = np.zeros(n_cols)
            
            # For each column
            for j in range(n_cols):
                # Get column slice
                start, end = A_csc.indptr[j], A_csc.indptr[j+1]
                if start < end:
                    col_data = A_csc.data[start:end]
                    if norm_type == "inf":
                        col_norms[j] = np.max(np.abs(col_data))
                    elif norm_type == "1":
                        col_norms[j] = np.sum(np.abs(col_data))
                    else:  # default to "2"
                        col_norms[j] = np.sqrt(np.sum(col_data * col_data))
            
            return col_norms
        else:
            # Handle dense matrix
            if norm_type == "inf":
                return np.max(np.abs(A), axis=0)
            elif norm_type == "1":
                return np.sum(np.abs(A), axis=0)
            else:  # default to "2"
                return np.sqrt(np.sum(A * A, axis=0))
    
    def _create_scale_factors(self, norms):
        """
        Create scaling factors from norms.
        
        Args:
            norms: Array of norms
            
        Returns:
            scale_factors: Array of scaling factors
        """
        # Avoid division by zero
        scale_factors = 1.0 / (norms + self.epsilon)
        
        # If any norm is zero, set scaling factor to 1.0
        scale_factors[norms == 0] = 1.0
        
        return scale_factors
    
    def _scale_matrix(self, A, row_scale, col_scale):
        """
        Scale matrix A by row and column scaling factors.
        
        Args:
            A: Matrix
            row_scale: Row scaling factors
            col_scale: Column scaling factors
            
        Returns:
            scaled_A: Scaled matrix
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently (D_r * A * D_c)
            from scipy import sparse
            D_r = sparse.diags(row_scale)
            D_c = sparse.diags(col_scale)
            
            # Need both CSR and CSC format for efficiency
            if hasattr(A, "tocsr"):
                A_csr = A.tocsr()
            else:
                A_csr = A
                
            # D_r * A * D_c
            return D_r @ A_csr @ D_c
        else:
            # Handle dense matrix
            return row_scale.reshape(-1, 1) * A * col_scale
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return "EquilibrationScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return "Equilibration scaling for numerical stability."