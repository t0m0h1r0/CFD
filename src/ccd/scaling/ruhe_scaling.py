"""
Ruhe diagonal scaling implementation

This scaling method implements Axel Ruhe's iterative technique for improving
matrix condition number through diagonal scaling. The method iteratively
computes scaling factors to make row and column norms approximately equal.
"""

from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling


class RuheScaling(BaseScaling):
    """
    Ruhe diagonal scaling technique.
    
    This scaling strategy iteratively balances the matrix to achieve
    approximately equal row and column norms, which often improves
    the condition number significantly for many problem types.
    
    References:
        Ruhe, A. (1980). "Perturbation bounds for means of eigenvalues and invariant subspaces."
    """
    
    def __init__(self, max_iterations=10, tolerance=1e-6, norm_type=2):
        """
        Initialize the Ruhe scaling algorithm.
        
        Args:
            max_iterations: Maximum number of iterations for the algorithm
            tolerance: Convergence tolerance for row/column norm ratios
            norm_type: Type of norm to use (2 for Euclidean, float('inf') for max)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.norm_type = norm_type
        
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        Apply Ruhe diagonal scaling to matrix A and right-hand side b.
        
        Args:
            A: System matrix to scale
            b: Right-hand side vector
            
        Returns:
            Tuple of (scaled_A, scaled_b, scaling_info)
        """
        m, n = A.shape
        
        # Initialize scaling vectors with ones
        d_row = cp.ones(m, dtype=cp.float64)
        d_col = cp.ones(n, dtype=cp.float64)
        
        # Create a copy of A to work with
        scaled_A = A.copy()
        
        # Determine if A is stored in CSR or CSC format for efficient operations
        is_csr = hasattr(scaled_A, 'format') and scaled_A.format == 'csr'
        is_csc = hasattr(scaled_A, 'format') and scaled_A.format == 'csc'
        
        # Iterative scaling algorithm
        for iteration in range(self.max_iterations):
            # Compute row and column norms
            row_norms = self._compute_row_norms(scaled_A, is_csr)
            col_norms = self._compute_column_norms(scaled_A, is_csc)
            
            # Check convergence
            row_norm_avg = cp.mean(row_norms)
            col_norm_avg = cp.mean(col_norms)
            
            # Break if norms are nearly equal
            if abs(row_norm_avg - col_norm_avg) < self.tolerance:
                break
                
            # Compute scaling factors
            alpha_row = cp.sqrt(col_norm_avg / cp.maximum(row_norm_avg, 1e-15))
            alpha_col = cp.sqrt(row_norm_avg / cp.maximum(col_norm_avg, 1e-15))
            
            # Update scaling vectors
            d_row_update = cp.power(row_norms, -0.5) * alpha_row
            d_col_update = cp.power(col_norms, -0.5) * alpha_col
            
            # Apply numerical stability safeguards
            d_row_update = cp.where(cp.isfinite(d_row_update), d_row_update, 1.0)
            d_col_update = cp.where(cp.isfinite(d_col_update), d_col_update, 1.0)
            
            # Update cumulative scaling factors
            d_row *= d_row_update
            d_col *= d_col_update
            
            # Construct diagonal scaling matrices
            D_row = sp.diags(d_row_update)
            D_col = sp.diags(d_col_update)
            
            # Apply scaling to the matrix
            scaled_A = D_row @ scaled_A @ D_col
        
        # Scale the right-hand side vector
        scaled_b = b * d_row
        
        # Return the scaled matrix, scaled RHS, and scaling information
        return scaled_A, scaled_b, {'row_scale': d_row, 'col_scale': d_col}
        
    def _compute_row_norms(self, A, is_csr=False):
        """Efficiently compute row norms of matrix A"""
        m = A.shape[0]
        row_norms = cp.zeros(m)
        
        if is_csr:
            # For CSR format, use indptr for efficient row access
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_data = A.data[start:end]
                    if self.norm_type == float('inf'):
                        row_norms[i] = cp.max(cp.abs(row_data))
                    else:
                        row_norms[i] = cp.linalg.norm(row_data, ord=self.norm_type)
        else:
            # General case
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = cp.linalg.norm(row, ord=self.norm_type)
        
        return row_norms
    
    def _compute_column_norms(self, A, is_csc=False):
        """Efficiently compute column norms of matrix A"""
        n = A.shape[1]
        col_norms = cp.zeros(n)
        
        if is_csc:
            # For CSC format, use indptr for efficient column access
            for j in range(n):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    col_data = A.data[start:end]
                    if self.norm_type == float('inf'):
                        col_norms[j] = cp.max(cp.abs(col_data))
                    else:
                        col_norms[j] = cp.linalg.norm(col_data, ord=self.norm_type)
        else:
            # General case
            for j in range(n):
                col = A[:, j]
                if hasattr(col, 'toarray'):
                    col = col.toarray().flatten()
                col_norms[j] = cp.linalg.norm(col, ord=self.norm_type)
        
        return col_norms
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        Unscale the solution vector.
        
        Args:
            x: Solution vector to unscale
            scale_info: Scaling information from the scale method
            
        Returns:
            Unscaled solution vector
        """
        col_scale = scale_info.get('col_scale')
        if col_scale is None:
            return x
        return x / col_scale
    
    def scale_b_only(self, b: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        Scale only the right-hand side vector.
        
        Args:
            b: Right-hand side vector to scale
            scale_info: Scaling information from the scale method
            
        Returns:
            Scaled right-hand side vector
        """
        row_scale = scale_info.get('row_scale')
        if row_scale is None:
            return b
        return b * row_scale
    
    @property
    def name(self) -> str:
        return "RuheScaling"
    
    @property
    def description(self) -> str:
        return "Ruhe's iterative diagonal scaling to improve matrix condition number"
