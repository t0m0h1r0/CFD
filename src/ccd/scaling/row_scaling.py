"""
Row scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class RowScaling(BaseScaling):
    """Row scaling using row norms."""
    
    def __init__(self, norm_type="inf", epsilon=1e-10):
        """
        Initialize row scaling.
        
        Args:
            norm_type: Type of norm to use ("inf", "1", "2")
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.norm_type = norm_type
        self.epsilon = epsilon
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b by row norms.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Special case for CCD solver: Extract components from linear system
        # This is needed because CCD combines different variables in one system
        # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
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
            # Use standard row scaling for general matrices
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
        n_rows = A.shape[0]
        
        # Prepare scaling factors with proper size
        row_scale = np.ones(n_rows)
        
        # Apply scaling for each block separately
        n_blocks = n_rows // block_size
        for var_idx in range(block_size):
            # Extract the submatrix for this variable
            idx = list(range(var_idx, n_rows, block_size))
            
            # Get the submatrix
            if self.is_sparse(A):
                # For sparse matrices, we need to extract rows
                submatrix = A[idx, :]
            else:
                # For dense matrices, we can slice directly
                submatrix = A[idx, :]
            
            # Compute row norms for this submatrix
            sub_row_norms = self.compute_row_norms(submatrix, self.norm_type)
            
            # Create scaling factors for this block
            sub_scale = 1.0 / (sub_row_norms + self.epsilon)
            sub_scale[sub_row_norms == 0] = 1.0
            
            # Apply the scaling factors to the appropriate rows
            for i, scale in enumerate(sub_scale):
                row_scale[i * block_size + var_idx] = scale
        
        # Scale matrix and right-hand side
        scaled_A = self._scale_matrix(A, row_scale)
        scaled_b = b * row_scale
        
        return scaled_A, scaled_b, {"row_scale": row_scale, "block_size": block_size}
    
    def _scale_standard(self, A, b):
        """
        Standard row scaling for general matrices.
        
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
        scale_factors = 1.0 / (row_norms + self.epsilon)
        scale_factors[row_norms == 0] = 1.0
        
        # Limit extreme scaling factors to improve numerical stability
        max_scale = np.max(scale_factors[scale_factors < np.inf])
        min_scale = np.min(scale_factors[scale_factors > 0])
        
        # Cap the ratio to 1e8
        if max_scale / min_scale > 1e8:
            scale_factors = np.minimum(scale_factors, min_scale * 1e8)
        
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
            from scipy import sparse
            D = sparse.diags(scale_factors)
            
            # Convert to CSR for efficient row operations
            if hasattr(A, "tocsr"):
                A_csr = A.tocsr()
            else:
                A_csr = A
                
            # Scale rows: D * A
            return D @ A_csr
        else:
            # Handle dense matrix
            return scale_factors.reshape(-1, 1) * A
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"RowScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Row scaling using {self.norm_type}-norm of each row, with CCD-aware block handling."