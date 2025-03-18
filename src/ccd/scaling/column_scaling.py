"""
列スケーリング手法の実装
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.sparse as sp
from .base import BaseScaling


class ColumnScaling(BaseScaling):
    """列スケーリング手法: A → AD⁻¹, b → b (Dは列ノルムの対角行列)"""
    
    def __init__(self, norm_type=2):
        """
        ノルム型を指定して初期化
        
        Args:
            norm_type: 列スケーリングに使用するノルム型（デフォルト: 2-ノルム）
        """
        super().__init__()
        self.norm_type = norm_type
    
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Aの各列をそのノルムでスケーリング
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 列ノルムを計算
        col_norms = self._compute_column_norms(A)
        
        # 数値的安定性のための処理
        col_scale = 1.0 / np.where(col_norms < 1e-15, 1.0, col_norms)
        
        # スケーリング適用
        D_inv = sp.diags(col_scale)
        scaled_A = A @ D_inv
        
        # bは変更しない
        return scaled_A, b, {'col_scale': col_scale}
    
    def _compute_column_norms(self, A) -> np.ndarray:
        """行列の各列のノルムを計算"""
        col_norms = np.zeros(A.shape[1])
        
        # CSCフォーマットでより効率的に計算
        if hasattr(A, 'format') and A.format == 'csc':
            for j in range(A.shape[1]):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    if self.norm_type == float('inf'):
                        col_norms[j] = np.max(np.abs(A.data[start:end]))
                    else:
                        col_norms[j] = np.linalg.norm(A.data[start:end], ord=self.norm_type)
        else:
            # その他の形式
            for j in range(A.shape[1]):
                col = A[:, j]
                if hasattr(col, 'toarray'):
                    col = col.toarray().flatten()
                col_norms[j] = np.linalg.norm(col, ord=self.norm_type)
        
        return col_norms
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """解ベクトルをアンスケーリング"""
        col_scale = scale_info.get('col_scale')
        if col_scale is None:
            return x
        return x / col_scale
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """列スケーリングでは右辺ベクトルは変更されない"""
        return b
    
    @property
    def name(self) -> str:
        return "ColumnScaling"
    
    @property
    def description(self) -> str:
        return "各列をそのノルムでスケーリング"