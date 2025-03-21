"""
Ruiz scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class RuizScaling(BaseScaling):
    """Ruiz scaling using iterative equilibration."""
    
    def __init__(self, max_iter=5, tol=1e-2, norm_type="inf", epsilon=1e-10):
        """
        Initialize Ruiz scaling.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Tolerance for early stopping
            norm_type: Type of norm to use ("inf", "1", "2")
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.norm_type = norm_type
        self.epsilon = epsilon
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b using Ruiz scaling.
        
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
            # Use standard Ruiz scaling for general matrices
            return self._scale_standard(A, b)
    
    def _scale_ccd_system(self, A, b, block_size):
        """
        Scale a CCD system with block structure using Ruiz method.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            block_size: Size of blocks (4 for 1D, 7 for 2D)
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        n_rows, n_cols = A.shape
        n_blocks = n_rows // block_size
        
        # Copy A to avoid modifying the original
        if self.is_sparse(A):
            A_k = A.copy()
        else:
            A_k = A.copy()
        
        # Initialize accumulated scaling factors
        row_scale_accum = np.ones(n_rows)
        col_scale_accum = np.ones(n_cols)
        
        # Iteratively scale the matrix
        for k in range(self.max_iter):
            # Initialize scaling factors for this iteration
            row_scale = np.ones(n_rows)
            col_scale = np.ones(n_cols)
            
            # Process each block separately
            for var_idx in range(block_size):
                # Extract the submatrix for this variable type
                row_idx = list(range(var_idx, n_rows, block_size))
                col_idx = list(range(var_idx, n_cols, block_size))
                
                # Get the submatrix
                if self.is_sparse(A):
                    submatrix = A_k[row_idx, :][:, col_idx]
                else:
                    submatrix = A_k[np.ix_(row_idx, col_idx)]
                
                # Compute row norms
                sub_row_norms = self.compute_row_norms(submatrix, self.norm_type)
                sub_row_scale = 1.0 / np.sqrt(sub_row_norms + self.epsilon)
                sub_row_scale[sub_row_norms == 0] = 1.0
                
                # Compute column norms
                sub_col_norms = self._compute_column_norms(submatrix, self.norm_type)
                sub_col_scale = 1.0 / np.sqrt(sub_col_norms + self.epsilon)
                sub_col_scale[sub_col_norms == 0] = 1.0
                
                # Apply scales to the right indices
                for i, idx in enumerate(row_idx):
                    row_scale[idx] = sub_row_scale[i]
                
                for j, idx in enumerate(col_idx):
                    col_scale[idx] = sub_col_scale[j]
            
            # Apply scaling to the matrix
            A_k = self._scale_matrix(A_k, row_scale, col_scale)
            
            # Accumulate scaling factors
            row_scale_accum *= row_scale
            col_scale_accum *= col_scale
            
            # Check convergence on maximum element ratio
            max_elem = 0.0
            min_elem = float('inf')
            for i in range(n_rows):
                for j in range(n_cols):
                    if A_k[i, j] != 0:
                        val = abs(A_k[i, j])
                        max_elem = max(max_elem, val)
                        min_elem = min(min_elem, val)
            
            if min_elem > 0 and max_elem / min_elem < 1.0 + self.tol:
                break
        
        # Scale right-hand side
        scaled_b = b * row_scale_accum
        
        # Return result with all scaling info
        return A_k, scaled_b, {
            "row_scale": row_scale_accum,
            "col_scale": col_scale_accum,
            "iterations": k+1,
            "block_size": block_size
        }
    
    def _scale_standard(self, A, b):
        """
        Standard Ruiz scaling for general matrices.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Initialize scaling factors
        n_rows, n_cols = A.shape
        d_r = np.ones(n_rows)  # Row scaling
        d_c = np.ones(n_cols)  # Column scaling
        
        # Copy A to avoid modifying the original
        if self.is_sparse(A):
            A_k = A.copy()
        else:
            A_k = A.copy()
        
        # Iterative scaling
        for k in range(self.max_iter):
            # Compute row norms
            row_norms = self.compute_row_norms(A_k, self.norm_type)
            row_scale = 1.0 / np.sqrt(row_norms + self.epsilon)
            row_scale[row_norms == 0] = 1.0
            
            # Scale rows
            A_k = self._scale_rows(A_k, row_scale)
            d_r *= row_scale
            
            # Compute column norms
            col_norms = self._compute_column_norms(A_k, self.norm_type)
            col_scale = 1.0 / np.sqrt(col_norms + self.epsilon)
            col_scale[col_norms == 0] = 1.0
            
            # Scale columns
            A_k = self._scale_columns(A_k, col_scale)
            d_c *= col_scale
            
            # Check convergence
            max_elem = 0.0
            min_elem = float('inf')
            
            # For sparse matrices, only check non-zero elements
            if self.is_sparse(A_k):
                if hasattr(A_k, "data"):
                    data = A_k.data
                    if len(data) > 0:
                        max_elem = np.max(np.abs(data))
                        min_elem = np.min(np.abs(data[data != 0]))
            else:
                # For dense matrices
                A_flat = A_k.flatten()
                nonzeros = A_flat[A_flat != 0]
                if len(nonzeros) > 0:
                    max_elem = np.max(np.abs(nonzeros))
                    min_elem = np.min(np.abs(nonzeros))
            
            if min_elem > 0 and max_elem / min_elem < 1.0 + self.tol:
                break
        
        # Scale right-hand side
        scaled_b = b * d_r
        
        # Return result
        return A_k, scaled_b, {"row_scale": d_r, "col_scale": d_c, "iterations": k+1}
    
    def unscale(self, x, scale_info):
        """
        Unscale the solution vector.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        # Recover original solution: x / d_c
        col_scale = scale_info.get("col_scale")
        if col_scale is not None:
            # Only need to unscale by the column factors
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
        Compute column norms for Ruiz scaling.
        
        Args:
            A: Matrix
            norm_type: Norm type ("inf", "1", "2")
            
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
    
    def _scale_rows(self, A, row_scale):
        """
        Scale rows of matrix A.
        
        Args:
            A: Matrix
            row_scale: Row scaling factors
            
        Returns:
            scaled_A: Matrix with scaled rows
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            if hasattr(A, "tocsr"):
                A_csr = A.tocsr()
            else:
                A_csr = A
                
            # Create a copy if necessary
            if A is A_csr:
                A_csr = A_csr.copy()
                
            # For each row, scale the data
            for i in range(A_csr.shape[0]):
                start, end = A_csr.indptr[i], A_csr.indptr[i+1]
                if start < end:
                    A_csr.data[start:end] *= row_scale[i]
                    
            return A_csr
        else:
            # Handle dense matrix (in-place scaling if possible)
            return row_scale.reshape(-1, 1) * A
    
    def _scale_columns(self, A, col_scale):
        """
        Scale columns of matrix A.
        
        Args:
            A: Matrix
            col_scale: Column scaling factors
            
        Returns:
            scaled_A: Matrix with scaled columns
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Handle sparse matrix efficiently
            if hasattr(A, "tocsc"):
                A_csc = A.tocsc()
            else:
                A_csc = A
                
            # Create a copy if necessary
            if A is A_csc:
                A_csc = A_csc.copy()
                
            # For each column, scale the data
            for j in range(A_csc.shape[1]):
                start, end = A_csc.indptr[j], A_csc.indptr[j+1]
                if start < end:
                    A_csc.data[start:end] *= col_scale[j]
                    
            return A_csc
        else:
            # Handle dense matrix (in-place scaling if possible)
            return A * col_scale
    
    def _scale_matrix(self, A, row_scale, col_scale):
        """
        Scale matrix A with both row and column factors.
        
        Args:
            A: Matrix
            row_scale: Row scaling factors
            col_scale: Column scaling factors
            
        Returns:
            scaled_A: Scaled matrix
        """
        # First scale rows, then columns
        A_scaled = self._scale_rows(A, row_scale)
        return self._scale_columns(A_scaled, col_scale)
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"RuizScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Ruiz scaling using iterative equilibration with {self.norm_type}-norm, with CCD-aware block handling."