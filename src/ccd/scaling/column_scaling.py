"""
Column scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class ColumnScaling(BaseScaling):
    """Column scaling using column norms."""
    
    def __init__(self, norm_type="inf", epsilon=1e-10):
        """
        Initialize column scaling.
        
        Args:
            norm_type: Type of norm to use ("inf", "1", "2")
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.norm_type = norm_type
        self.epsilon = epsilon
    
    def scale(self, A, b):
        """
        Scale matrix A by column norms. Unlike row scaling, this changes the solution vector.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Special case for CCD solver: Extract components from linear system
        n_rows, n_cols = A.shape
        
        # Determine block size based on matrix shape
        block_size = 4  # Default for 1D
        if n_rows == n_cols and n_rows % 7 == 0:
            block_size = 7  # For 2D
        
        # Check if matrix seems to be a CCD matrix (square with multiple of block_size)
        is_ccd = (n_rows == n_cols) and (n_rows % block_size == 0)
        
        if is_ccd:
            # Use block-wise scaling for CCD matrices
            return self._scale_ccd_system(A, b, block_size)
        else:
            # Use standard column scaling for general matrices
            return self._scale_standard(A, b)
    
    def _scale_ccd_system(self, A, b, block_size):
        """
        Scale a CCD system with block structure.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            block_size: Size of blocks (4 for 1D, 7 for 2D)
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        n_cols = A.shape[1]
        
        # Prepare scaling factors with proper size
        col_scale = np.ones(n_cols)
        
        # Apply scaling for each block separately
        n_blocks = n_cols // block_size
        for var_idx in range(block_size):
            # Extract the submatrix for this variable
            idx = list(range(var_idx, n_cols, block_size))
            
            # Get the submatrix
            if self.is_sparse(A):
                # For sparse matrices, we need to extract columns
                if hasattr(A, "tocsc"):
                    A_csc = A.tocsc()
                else:
                    A_csc = A
                submatrix = A_csc[:, idx]
            else:
                # For dense matrices, we can slice directly
                submatrix = A[:, idx]
            
            # Compute column norms for this submatrix
            sub_col_norms = self._compute_column_norms(submatrix, self.norm_type)
            
            # Create scaling factors for this block
            sub_scale = 1.0 / (sub_col_norms + self.epsilon)
            sub_scale[sub_col_norms == 0] = 1.0
            
            # Apply the scaling factors to the appropriate columns
            for i, scale in enumerate(sub_scale):
                col_scale[i * block_size + var_idx] = scale
        
        # Scale matrix but NOT vector
        scaled_A = self._scale_matrix(A, col_scale)
        
        return scaled_A, b, {"col_scale": col_scale, "block_size": block_size}
    
    def _scale_standard(self, A, b):
        """
        Standard column scaling for general matrices.
        
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
        scale_factors = 1.0 / (col_norms + self.epsilon)
        scale_factors[col_norms == 0] = 1.0
        
        # Limit extreme scaling factors to improve numerical stability
        max_scale = np.max(scale_factors[scale_factors < np.inf])
        min_scale = np.min(scale_factors[scale_factors > 0])
        
        # Cap the ratio to 1e8
        if max_scale / min_scale > 1e8:
            scale_factors = np.minimum(scale_factors, min_scale * 1e8)
        
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
            from scipy import sparse
            D = sparse.diags(scale_factors)
            
            # Convert to CSC for efficient column operations
            if hasattr(A, "tocsc"):
                A_csc = A.tocsc()
            else:
                A_csc = A
                
            # Scale columns: A * D
            return A_csc @ D
        else:
            # Handle dense matrix
            return A * scale_factors
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"ColumnScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Column scaling using {self.norm_type}-norm of each column, with CCD-aware block handling."