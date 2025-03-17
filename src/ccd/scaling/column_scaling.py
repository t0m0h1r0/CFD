"""
列スケーリング手法の実装
"""

from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class ColumnScaling(BaseScaling):
    """列スケーリング手法: A → AD⁻¹, b → b (Dは列ノルムの対角行列)"""
    
    def __init__(self, norm_type=2):
        """
        ノルム型を指定して初期化
        
        Args:
            norm_type: 列スケーリングに使用するノルム型（デフォルト: 2-ノルム）
        """
        self.norm_type = norm_type
    
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        Aの各列をそのノルムでスケーリング
        """
        # 列ノルムを計算（効率化された実装）
        col_norms = self._compute_column_norms(A)
        
        # 数値的安定性のための処理
        col_scale = 1.0 / cp.where(col_norms < 1e-15, 1.0, col_norms)
        
        # スケーリング適用
        D_inv = sp.diags(col_scale)
        scaled_A = A @ D_inv
        
        # bは変更しない
        return scaled_A, b, {'col_scale': col_scale}
    
    def _compute_column_norms(self, A: Union[sp.spmatrix, cp.ndarray]) -> cp.ndarray:
        """行列の各列のノルムを計算（効率化実装）"""
        col_norms = cp.zeros(A.shape[1])
        
        # CSCフォーマットでより効率的に計算
        if hasattr(A, 'format') and A.format == 'csc':
            for j in range(A.shape[1]):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    if self.norm_type == float('inf'):
                        col_norms[j] = cp.max(cp.abs(A.data[start:end]))
                    else:
                        col_norms[j] = cp.linalg.norm(A.data[start:end], ord=self.norm_type)
        else:
            # その他の形式
            for j in range(A.shape[1]):
                col = A[:, j].toarray().flatten() if hasattr(A[:, j], 'toarray') else A[:, j]
                col_norms[j] = cp.linalg.norm(col, ord=self.norm_type)
        
        return col_norms
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """解ベクトルをアンスケーリング"""
        col_scale = scale_info.get('col_scale')
        if col_scale is None:
            return x
        return x / col_scale
    
    def scale_b_only(self, b: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """列スケーリングでは右辺ベクトルは変更されない"""
        return b
    
    @property
    def name(self) -> str:
        return "ColumnScaling"
    
    @property
    def description(self) -> str:
        return "各列をそのノルムでスケーリング"