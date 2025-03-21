"""
Ruiz scaling implementation.
"""

import numpy as np
from .base import BaseScaling


class RuizScaling(BaseScaling):
    """Ruiz scaling using iterative equilibration."""
    
    def __init__(self, max_iter=5, tol=1e-2, norm_type="inf"):
        """
        Initialize Ruiz scaling.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Tolerance for early stopping
            norm_type: Type of norm to use ("inf", "1", "2")
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.norm_type = norm_type
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b using Ruiz scaling.
        
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
        
        # Track intermediates for debugging
        row_scales = []
        col_scales = []
        
        # Iterative scaling
        for k in range(self.max_iter):
            # Compute row norms
            row_norms = self._compute_row_norms(A_k)
            row_scale = 1.0 / np.sqrt(row_norms + 1e-10)
            row_scale[row_norms == 0] = 1.0
            
            # Scale rows
            A_k = self._scale_rows(A_k, row_scale)
            d_r *= row_scale
            row_scales.append(row_scale)
            
            # Compute column norms
            col_norms = self._compute_column_norms(A_k)
            col_scale = 1.0 / np.sqrt(col_norms + 1e-10)
            col_scale[col_norms == 0] = 1.0
            
            # Scale columns
            A_k = self._scale_columns(A_k, col_scale)
            d_c *= col_scale
            col_scales.append(col_scale)
            
            # Check convergence
            norms_ratio = np.max(row_norms) / np.min(row_norms[row_norms > 0])
            if norms_ratio < 1.0 + self.tol:
                break
        
        # Scale right-hand side
        scaled_b = b * d_r
        
        # Store all scaling information
        scaling_info = {
            "row_scale": d_r,             # Cumulative row scaling
            "col_scale": d_c,             # Cumulative column scaling
            "row_scales": row_scales,     # Individual iteration row scales
            "col_scales": col_scales,     # Individual iteration column scales
            "iterations": k+1             # Number of iterations performed
        }
        
        return A_k, scaled_b, scaling_info
    
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
    
    def _compute_row_norms(self, A):
        """
        Compute row norms for Ruiz scaling.
        
        Args:
            A: Matrix
            
        Returns:
            row_norms: Array of row norms
        """
        return self.compute_row_norms(A, self.norm_type)
    
    def _compute_column_norms(self, A):
        """
        Compute column norms for Ruiz scaling.
        
        Args:
            A: Matrix
            
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
                    if self.norm_type == "inf":
                        col_norms[j] = np.max(np.abs(col_data))
                    elif self.norm_type == "1":
                        col_norms[j] = np.sum(np.abs(col_data))
                    else:  # default to "2"
                        col_norms[j] = np.sqrt(np.sum(col_data * col_data))
            
            return col_norms
        else:
            # Handle dense matrix
            if self.norm_type == "inf":
                return np.max(np.abs(A), axis=0)
            elif self.norm_type == "1":
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
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return f"RuizScaling_{self.norm_type}"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Ruiz scaling using iterative equilibration with {self.norm_type}-norm."