"""
行スケーリング手法の実装
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.sparse as sp
from .base import BaseScaling


class RowScaling(BaseScaling):
    """行スケーリング手法: A → D⁻¹A, b → D⁻¹b (Dは行ノルムの対角行列)"""
    
    def __init__(self, norm_type=float('inf')):
        """
        ノルム型を指定して初期化
        
        Args:
            norm_type: 行スケーリングに使用するノルム型（デフォルト: 無限大ノルム）
        """
        super().__init__()
        self.norm_type = norm_type
    
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Aの各行をそのノルムでスケーリングし、bも対応して調整
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 行ノルムを計算
        row_norms = self._compute_row_norms(A)
        
        # 数値的安定性のための処理
        row_scale = 1.0 / np.where(row_norms < 1e-15, 1.0, row_norms)
        
        # スケーリング適用
        D_inv = sp.diags(row_scale)
        scaled_A = D_inv @ A
        scaled_b = D_inv @ b
        
        return scaled_A, scaled_b, {'row_scale': row_scale}
    
    def _compute_row_norms(self, A):
        """行列の各行のノルムを計算"""
        m = A.shape[0]
        row_norms = np.zeros(m)
        
        # CSRフォーマットでより効率的に計算
        if hasattr(A, 'format') and A.format == 'csr':
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    if self.norm_type == float('inf'):
                        row_norms[i] = np.max(np.abs(A.data[start:end]))
                    else:
                        row_norms[i] = np.linalg.norm(A.data[start:end], ord=self.norm_type)
        else:
            # その他の形式
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = np.linalg.norm(row, ord=self.norm_type)
        
        return row_norms
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """行スケーリングは解ベクトルに影響しない"""
        return x
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """右辺ベクトルbのみを効率的にスケーリング"""
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
        return b
    
    @property
    def name(self) -> str:
        return "RowScaling"
    
    @property
    def description(self) -> str:
        return "各行をそのノルムでスケーリング"