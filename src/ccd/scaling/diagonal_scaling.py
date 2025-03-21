"""
Diagonal scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class DiagonalScaling(BaseScaling):
    """Diagonal scaling (symmetrized row/column scaling)."""
    
    def __init__(self, norm_type="2", epsilon=1e-10):
        """
        Initialize diagonal scaling.
        
        Args:
            norm_type: Type of norm to use ("2", "inf", "1")
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.norm_type = norm_type
        self.epsilon = epsilon
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b using diagonal scaling.
        
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
            # Use standard diagonal scaling for general matrices
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
        
        # Prepare scaling factors with proper size
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
            
            # Compute diagonal scaling factors for this submatrix
            # For simplicity, use row and column norms
            sub_row_norms = self.compute_row_norms(submatrix, self.norm_type)
            sub_col_norms = self._compute_column_norms(submatrix, self.norm_type)
            
            # Create scaling factors (sqrt for symmetry)
            sub_row_scale = 1.0 / np.sqrt(sub_row_norms + self.epsilon)
            sub_row_scale[sub_row_norms == 0] = 1.0
            
            sub_col_scale = 1.0 / np.sqrt(sub_col_norms + self.epsilon)
            sub_col_scale[sub_col_norms == 0] = 1.0
            
            # Apply the scaling factors to the appropriate rows/columns
            for i, scale in enumerate(sub_row_scale):
                row_scale[i * block_size + var_idx] = scale
                
            for j, scale in enumerate(sub_col_scale):
                col_scale[j * block_size + var_idx] = scale
        
        # Scale matrix and right-hand side
        scaled_A = self._scale_matrix(A, row_scale, col_scale)
        scaled_b = b * row_scale
        
        return scaled_A, scaled_b, {
            "row_scale": row_scale, 
            "col_scale": col_scale,
            "block_size": block_size
        }
    
    def _scale_standard(self, A, b):
        """
        Standard diagonal scaling for general matrices.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # For non-CCD matrices
        if self.norm_type == "2":
            # Compute diagonal scaling factors using eigenvalue approach
            d = self._compute_diagonal_factors(A)
        else:
            # Compute row and column norms
            row_norms = self.compute_row_norms(A, self.norm_type)
            col_norms = self._compute_column_norms(A, self.norm_type)
            
            # Create symmetric scaling factors
            row_scale = 1.0 / np.sqrt(row_norms + self.epsilon)
            row_scale[row_norms == 0] = 1.0
            
            col_scale = 1.0 / np.sqrt(col_norms + self.epsilon)
            col_scale[col_norms == 0] = 1.0
            
            # Limit extreme scaling factors to improve numerical stability
            max_row_scale = np.max(row_scale[row_scale < np.inf])
            min_row_scale = np.min(row_scale[row_scale > 0])
            
            max_col_scale = np.max(col_scale[col_scale < np.inf])
            min_col_scale = np.min(col_scale[col_scale > 0])
            
            # Cap the ratio to 1e4 (sqrt of 1e8 because we apply the sqrt scale)
            if max_row_scale / min_row_scale > 1e4:
                row_scale = np.minimum(row_scale, min_row_scale * 1e4)
                
            if max_col_scale / min_col_scale > 1e4:
                col_scale = np.minimum(col_scale, min_col_scale * 1e4)
            
            # Scale matrix and right-hand side
            scaled_A = self._scale_matrix(A, row_scale, col_scale)
            scaled_b = b * row_scale
            
            return scaled_A, scaled_b, {"row_scale": row_scale, "col_scale": col_scale}
    
        # Square root of scaling factors for symmetry
        D_sqrt_inv = 1.0 / np.sqrt(d)
        
        # Scale matrix: D^(-1/2) * A * D^(-1/2)
        scaled_A = self._scale_matrix_symmetric(A, D_sqrt_inv)
        
        # Scale right-hand side: D^(-1/2) * b
        scaled_b = b * D_sqrt_inv
        
        # Return scaled matrix, vector, and scaling info
        return scaled_A, scaled_b, {"D_sqrt_inv": D_sqrt_inv}
    
    def unscale(self, x, scale_info):
        """
        Unscale the solution vector.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        # Check for different scaling types
        D_sqrt_inv = scale_info.get("D_sqrt_inv")
        col_scale = scale_info.get("col_scale")
        
        if D_sqrt_inv is not None:
            # Symmetric scaling: D^(-1/2) * x
            return x * D_sqrt_inv
        elif col_scale is not None:
            # Row/column scaling: x / col_scale
            return x / col_scale
            
        # Default (no unscaling needed)
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
        D_sqrt_inv = scale_info.get("D_sqrt_inv")
        row_scale = scale_info.get("row_scale")
        
        if D_sqrt_inv is not None:
            return b * D_sqrt_inv
        elif row_scale is not None:
            return b * row_scale
            
        return b
    
    def _compute_diagonal_factors(self, A, epsilon=1e-10):
        """
        Compute diagonal scaling factors.
        
        Args:
            A: System matrix
            epsilon: Small value to avoid division by zero
            
        Returns:
            d: Diagonal scaling factors
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            if self.norm_type == "2":
                # For 2-norm, use the diagonal elements of A^T * A
                d = self._sparse_diagonal_2_norm(A)
            elif self.norm_type == "inf":
                # For inf-norm, use the maximum absolute value in each row and column
                d = self._sparse_diagonal_inf_norm(A)
            else:  # default to "1"
                # For 1-norm, use the sum of absolute values in each row and column
                d = self._sparse_diagonal_1_norm(A)
        else:
            # Handle dense matrix
            if self.norm_type == "2":
                # For 2-norm, use the diagonal elements of A^T * A
                d = np.diag(A.T @ A)
            elif self.norm_type == "inf":
                # For inf-norm, use the maximum absolute value in each row and column
                row_max = np.max(np.abs(A), axis=1)
                col_max = np.max(np.abs(A), axis=0)
                d = row_max * col_max
            else:  # default to "1"
                # For 1-norm, use the sum of absolute values in each row and column
                row_sum = np.sum(np.abs(A), axis=1)
                col_sum = np.sum(np.abs(A), axis=0)
                d = row_sum * col_sum
        
        # Avoid division by zero
        d = np.maximum(d, epsilon)
        
        return d
    
    def _sparse_diagonal_2_norm(self, A):
        """
        Compute 2-norm diagonal scaling factors for a sparse matrix.
        
        Args:
            A: Sparse matrix
            
        Returns:
            d: Diagonal scaling factors
        """
        from scipy import sparse
        
        # Convert to CSR if not already
        if not hasattr(A, "tocsr") or A.format != "csr":
            A = A.tocsr()
        
        # Compute diagonal elements of A^T * A efficiently
        d = np.zeros(A.shape[1])
        
        # For each column
        for i in range(A.shape[1]):
            # Get column i from A
            col_i = A.getcol(i)
            
            # Compute sum of squares of elements in column i
            d[i] = np.sum(col_i.data ** 2)
            
        return d
    
    def _sparse_diagonal_inf_norm(self, A):
        """
        Compute infinity-norm diagonal scaling factors for a sparse matrix.
        
        Args:
            A: Sparse matrix
            
        Returns:
            d: Diagonal scaling factors
        """
        # Convert to CSR for row operations
        if hasattr(A, "tocsr"):
            A_csr = A.tocsr()
        else:
            A_csr = A
            
        # Convert to CSC for column operations
        if hasattr(A, "tocsc"):
            A_csc = A.tocsc()
        else:
            A_csc = A.tocsc()
        
        n_rows, n_cols = A.shape
        d = np.ones(max(n_rows, n_cols))
        
        # Compute row maximums
        row_max = np.zeros(n_rows)
        for i in range(n_rows):
            start, end = A_csr.indptr[i], A_csr.indptr[i+1]
            if start < end:
                row_data = A_csr.data[start:end]
                row_max[i] = np.max(np.abs(row_data))
        
        # Compute column maximums
        col_max = np.zeros(n_cols)
        for j in range(n_cols):
            start, end = A_csc.indptr[j], A_csc.indptr[j+1]
            if start < end:
                col_data = A_csc.data[start:end]
                col_max[j] = np.max(np.abs(col_data))
        
        # Set diagonal scaling factors
        for i in range(min(n_rows, n_cols)):
            d[i] = row_max[i] * col_max[i]
        
        return d
    
    def _sparse_diagonal_1_norm(self, A):
        """
        Compute 1-norm diagonal scaling factors for a sparse matrix.
        
        Args:
            A: Sparse matrix
            
        Returns:
            d: Diagonal scaling factors
        """
        # Convert to CSR for row operations
        if hasattr(A, "tocsr"):
            A_csr = A.tocsr()
        else:
            A_csr = A
            
        # Convert to CSC for column operations
        if hasattr(A, "tocsc"):
            A_csc = A.tocsc()
        else:
            A_csc = A.tocsc()
        
        n_rows, n_cols = A.shape
        d = np.ones(max(n_rows, n_cols))
        
        # Compute row sums
        row_sum = np.zeros(n_rows)
        for i in range(n_rows):
            start, end = A_csr.indptr[i], A_csr.indptr[i+1]
            if start < end:
                row_data = A_csr.data[start:end]
                row_sum[i] = np.sum(np.abs(row_data))
        
        # Compute column sums
        col_sum = np.zeros(n_cols)
        for j in range(n_cols):
            start, end = A_csc.indptr[j], A_csc.indptr[j+1]
            if start < end:
                col_data = A_csc.data[start:end]
                col_sum[j] = np.sum(np.abs(col_data))
        
        # Set diagonal scaling factors
        for i in range(min(n_rows, n_cols)):
            d[i] = row_sum[i] * col_sum[i]
        
        return d
    
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
            
            # Make sure A is in CSR format for efficient operations
            if hasattr(A, "tocsr"):
                A_csr = A.tocsr()
            else:
                A_csr = A
            
            # Scale rows and columns: D_r * A * D_c
            return D_r @ A_csr @ D_c
        else:
            # Handle dense matrix: D_r * A * D_c
            return (row_scale.reshape(-1, 1) * A) * col_scale
    
    def _scale_matrix_symmetric(self, A, D_sqrt_inv):
        """
        Scale matrix A using symmetric scaling.
        
        Args:
            A: Matrix
            D_sqrt_inv: Square root of inverse diagonal scaling factors
            
        Returns:
            scaled_A: Scaled matrix
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            from scipy import sparse
            D = sparse.diags(D_sqrt_inv)
            
            # Make sure A is in CSR format for efficient operations
            if hasattr(A, "tocsr"):
                A_csr = A.tocsr()
            else:
                A_csr = A
            
            # Scale symmetrically: D * A * D
            return D @ A_csr @ D
        else:
            # Handle dense matrix: D * A * D
            return (D_sqrt_inv.reshape(-1, 1) * A) * D_sqrt_inv
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"DiagonalScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Diagonal scaling using {self.norm_type}-norm (symmetrized row/column scaling), with CCD-aware block handling."