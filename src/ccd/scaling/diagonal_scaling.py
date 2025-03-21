"""
Diagonal scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class DiagonalScaling(BaseScaling):
    """Diagonal scaling (symmetrized row/column scaling)."""
    
    def __init__(self, norm_type="2"):
        """
        Initialize diagonal scaling.
        
        Args:
            norm_type: Type of norm to use ("2", "inf", "1")
        """
        super().__init__()
        self.norm_type = norm_type
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b using diagonal scaling.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Compute diagonal scaling factors
        d = self._compute_diagonal_factors(A)
        D_sqrt_inv = 1.0 / np.sqrt(d)
        
        # Scale matrix: D^(-1/2) * A * D^(-1/2)
        scaled_A = self._scale_matrix(A, D_sqrt_inv)
        
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
        # Recover original solution: D^(-1/2) * x
        D_sqrt_inv = scale_info.get("D_sqrt_inv")
        if D_sqrt_inv is not None:
            return x * D_sqrt_inv
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
        if D_sqrt_inv is not None:
            return b * D_sqrt_inv
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
    
    def _scale_matrix(self, A, D_sqrt_inv):
        """
        Scale matrix A by diagonal scaling factors.
        
        Args:
            A: System matrix
            D_sqrt_inv: Array of sqrt(1/d) scaling factors
            
        Returns:
            scaled_A: Scaled matrix
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            return self._scale_sparse_matrix(A, D_sqrt_inv)
        else:
            # Handle dense matrix
            # D^(-1/2) * A * D^(-1/2)
            D_sqrt_inv_mat = np.diag(D_sqrt_inv)
            return D_sqrt_inv_mat @ A @ D_sqrt_inv_mat
    
    def _scale_sparse_matrix(self, A, D_sqrt_inv):
        """
        Scale sparse matrix A by diagonal scaling factors.
        
        Args:
            A: Sparse matrix
            D_sqrt_inv: Array of sqrt(1/d) scaling factors
            
        Returns:
            scaled_A: Scaled sparse matrix
        """
        from scipy import sparse
        
        # Convert to CSR if not already
        if hasattr(A, "tocsr"):
            A = A.tocsr()
        
        n_rows, n_cols = A.shape
        
        # Create diagonal matrices
        D_sqrt_inv_row = sparse.diags(D_sqrt_inv[:n_rows])
        D_sqrt_inv_col = sparse.diags(D_sqrt_inv[:n_cols])
        
        # Scale matrix: D^(-1/2) * A * D^(-1/2)
        scaled_A = D_sqrt_inv_row @ A @ D_sqrt_inv_col
        
        return scaled_A
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"DiagonalScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Diagonal scaling using {self.norm_type}-norm (symmetrized row/column scaling)."