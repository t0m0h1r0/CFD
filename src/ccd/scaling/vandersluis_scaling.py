"""
Van der Sluis diagonal scaling implementation

This scaling method implements Van der Sluis's optimal technique for improving
matrix condition number through row scaling. It's particularly effective for
ill-conditioned matrices and is theoretically proven to produce good results
for many problem classes.
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.sparse as sp
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
        super().__init__()
        self.norm_type = norm_type
    
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
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
        row_norms = np.where(row_norms < 1e-15, 1.0, row_norms)
        
        # Compute optimal scaling factors
        # In van der Sluis method, each row is scaled by 1/row_norm
        row_scale = 1.0 / row_norms
        
        # For very ill-conditioned matrices, apply dampening
        max_scale = np.max(row_scale)
        min_scale = np.min(row_scale)
        if max_scale / min_scale > 1e10:
            row_scale = np.sqrt(row_scale)  # Dampen the scaling
        
        # Apply scaling to the matrix based on its format
        if hasattr(A, 'format'):
            # 疎行列の場合
            if A.format == 'csr':
                # CSR形式の場合、行ごとに直接データをスケーリング
                scaled_data = A.data.copy()
                for i in range(m):
                    start, end = A.indptr[i], A.indptr[i+1]
                    if end > start:
                        scaled_data[start:end] = A.data[start:end] * row_scale[i]
                
                # NumPyでCSR行列を作成
                scaled_A = sp.csr_matrix((scaled_data, A.indices.copy(), A.indptr.copy()), shape=A.shape)
            else:
                # その他の形式は対角行列との積で処理
                D_row = sp.diags(row_scale)
                scaled_A = D_row @ A
        else:
            # 密行列の場合はブロードキャスティングで行単位のスケーリング
            scaled_A = A * row_scale.reshape(-1, 1)
        
        # 右辺ベクトルをスケーリング
        scaled_b = b * row_scale
        
        return scaled_A, scaled_b, {'row_scale': row_scale}
    
    def _compute_row_norms(self, A, is_csr=False):
        """Efficiently compute row norms of matrix A"""
        m = A.shape[0]
        row_norms = np.zeros(m)
        
        if is_csr:
            # CSR形式では効率的に行アクセス
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_data = A.data[start:end]
                    if self.norm_type == float('inf'):
                        row_norms[i] = np.max(np.abs(row_data))
                    else:
                        row_norms[i] = np.linalg.norm(row_data, ord=self.norm_type)
        else:
            # 一般的なケース
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = np.linalg.norm(row, ord=self.norm_type)
        
        return row_norms
    
    def unscale(self, x, scale_info: Dict[str, Any]):
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
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
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