"""
Equilibration scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class EquilibrationScaling(BaseScaling):
    """Equilibration scaling for numerical stability."""
    
    def __init__(self, epsilon=1e-10, max_ratio=1e8):
        """
        Initialize equilibration scaling.
        
        Args:
            epsilon: Small value to avoid division by zero
            max_ratio: Maximum allowed ratio between min and max scale factors
        """
        super().__init__()
        self.epsilon = epsilon
        self.max_ratio = max_ratio
    
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
            # Use standard equilibration for general matrices
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
        n_rows, n_cols = A.shape
        
        # Prepare scaling factors
        row_scale = np.ones(n_rows)
        col_scale = np.ones(n_cols)
        
        # Apply scaling for each block separately
        n_blocks = n_rows // block_size
        for var_idx in range(block_size):
            # Extract the submatrix for this variable
            row_idx = list(range(var_idx, n_rows, block_size))
            col_idx = list(range(var_idx, n_cols, block_size))
            
            # Get the submatrix
            if self.is_sparse(A):
                # For sparse matrices, we need to extract rows/columns
                submatrix = A[row_idx, :][:, col_idx]
            else:
                # For dense matrices, we can slice directly
                submatrix = A[np.ix_(row_idx, col_idx)]
            
            # Compute row and column norms
            sub_row_norms = self.compute_row_norms(submatrix, norm_type="inf")
            sub_col_norms = self._compute_column_norms(submatrix, norm_type="inf")
            
            # Create safe scaling factors
            sub_row_scale = self._create_safe_scale(sub_row_norms)
            sub_col_scale = self._create_safe_scale(sub_col_norms)
            
            # Apply the scaling factors to the appropriate rows/columns
            for i, scale in enumerate(sub_row_scale):
                row_scale[i * block_size + var_idx] = scale
                
            for j, scale in enumerate(sub_col_scale):
                col_scale[j * block_size + var_idx] = scale
        
        # Scale matrix and right-hand side
        scaled_A = self._scale_matrix(A, row_scale, col_scale)
        scaled_b = b * row_scale
        
        # Return scaled matrix, vector, and scaling info
        return scaled_A, scaled_b, {
            "row_scale": row_scale, 
            "col_scale": col_scale,
            "block_size": block_size
        }
    
    def _scale_standard(self, A, b):
        """
        Standard equilibration scaling for general matrices.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Compute row and column norms
        row_norms = self.compute_row_norms(A, norm_type="inf")
        col_norms = self._compute_column_norms(A, norm_type="inf")
        
        # Create safe scaling factors
        row_scale = self._create_safe_scale(row_norms)
        col_scale = self._create_safe_scale(col_norms)
        
        # Scale matrix and right-hand side
        scaled_A = self._scale_matrix(A, row_scale, col_scale)
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
    
    def _create_safe_scale(self, norms):
        """
        Create safe scaling factors with limited dynamic range.
        
        Args:
            norms: Array of norms
            
        Returns:
            scale_factors: Array of scaling factors
        """
        # Avoid division by zero
        safe_norms = norms.copy()
        safe_norms[safe_norms == 0] = 1.0
        
        # Create scaling factors
        scale_factors = 1.0 / safe_norms
        
        # Limit extreme scaling factors to improve numerical stability
        if len(scale_factors) > 0:
            max_scale = np.max(scale_factors[scale_factors < np.inf])
            min_scale = np.min(scale_factors[scale_factors > 0])
            
            # Cap the ratio to max_ratio
            if max_scale / min_scale > self.max_ratio:
                scale_factors = np.minimum(scale_factors, min_scale * self.max_ratio)
        
        return scale_factors
    
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
            # Handle sparse matrix efficiently
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
            return (row_scale.reshape(-1, 1) * A) * col_scale
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return "EquilibrationScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return "Equilibration scaling for numerical stability, with CCD-aware block handling."