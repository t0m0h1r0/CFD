"""
Van der Sluis diagonal scaling implementation

This scaling method implements Van der Sluis's optimal technique for improving
matrix condition number through row scaling. It's particularly effective for
ill-conditioned matrices and is theoretically proven to produce good results
for many problem classes.
"""

from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling


class VanDerSluisScaling(BaseScaling):
    """
    Van der Sluis diagonal scaling.
    
    This scaling strategy uses the geometric mean of row norms to compute
    optimal scaling factors, which has been proven to minimize the condition
    number (in a certain sense) for many matrix classes.
    
    References:
        Van der Sluis, A. (1969). "Condition numbers and equilibration of matrices."
        Numerische Mathematik, 14(1), 14-23.
    """
    
    def __init__(self, norm_type=2):
        """
        Initialize Van der Sluis scaling.
        
        Args:
            norm_type: Type of norm to use (2 for Euclidean, float('inf') for max)
        """
        self.norm_type = norm_type
    
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        Apply Van der Sluis optimal row scaling to matrix A and right-hand side b.
        
        Args:
            A: System matrix to scale
            b: Right-hand side vector
            
        Returns:
            Tuple of (scaled_A, scaled_b, scaling_info)
        """
        m, n = A.shape
        
        # Compute row norms efficiently
        is_csr = hasattr(A, 'format') and A.format == 'csr'
        row_norms = self._compute_row_norms(A, is_csr)
        
        # Replace zeros and small values with ones for stability
        row_norms = cp.where(row_norms < 1e-15, 1.0, row_norms)
        
        # Compute optimal scaling factors
        # In van der Sluis method, each row is scaled by 1/row_norm
        row_scale = 1.0 / row_norms
        
        # For very ill-conditioned matrices, apply dampening
        # This helps avoid catastrophic cancellation in extreme cases
        if cp.max(row_scale) / cp.min(row_scale) > 1e10:
            row_scale = cp.sqrt(row_scale)  # Dampen the scaling
        
        # Apply scaling to the matrix based on its format
        if hasattr(A, 'format'):
            # Handle sparse matrices
            if A.format == 'csr':
                # For CSR format, we can directly scale the data by row
                scaled_data = A.data.copy()
                for i in range(m):
                    start, end = A.indptr[i], A.indptr[i+1]
                    scaled_data[start:end] = A.data[start:end] * row_scale[i]
                scaled_A = sp.csr_matrix((scaled_data, A.indices, A.indptr), shape=A.shape)
            elif A.format == 'csc':
                # For CSC format, we need to convert indices to rows
                A_csr = A.tocsr()
                scaled_data = A_csr.data.copy()
                for i in range(m):
                    start, end = A_csr.indptr[i], A_csr.indptr[i+1]
                    scaled_data[start:end] = A_csr.data[start:end] * row_scale[i]
                scaled_A = sp.csr_matrix((scaled_data, A_csr.indices, A_csr.indptr), shape=A.shape)
                # Convert back to original format if needed
                if A.format == 'csc':
                    scaled_A = scaled_A.tocsc()
            else:
                # For other formats, try direct matrix multiplication
                try:
                    D_row = sp.diags(row_scale)
                    scaled_A = D_row @ A
                except TypeError:
                    # Fallback: convert to array, scale, then back to original format
                    A_array = A.toarray()
                    scaled_array = A_array * row_scale[:, cp.newaxis]
                    if A.format == 'dia':
                        scaled_A = sp.dia_matrix(scaled_array)
                    else:
                        scaled_A = sp.csr_matrix(scaled_array)
                        if A.format != 'csr':
                            # Try to convert back to original format
                            conversion_method = getattr(scaled_A, f'to{A.format}', None)
                            if conversion_method:
                                scaled_A = conversion_method()
        else:
            # For dense matrices, use broadcasting
            scaled_A = A * row_scale[:, cp.newaxis]
        
        # Apply scaling to the right-hand side
        scaled_b = b * row_scale
        
        return scaled_A, scaled_b, {'row_scale': row_scale}
    
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
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        Unscale the solution vector.
        
        Args:
            x: Solution vector to unscale
            scale_info: Scaling information from the scale method
            
        Returns:
            Unscaled solution vector
        """
        # Van der Sluis scaling only affects rows, not the solution vector
        return x
    
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
        return "VanDerSluisScaling"
    
    @property
    def description(self) -> str:
        return "Van der Sluis optimal row scaling to minimize condition number"