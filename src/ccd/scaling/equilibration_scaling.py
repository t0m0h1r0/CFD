"""
行と列のスケーリングを組み合わせた平衡化スケーリング実装
"""

from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class EquilibrationScaling(BaseScaling):
    """行と列のスケーリングを組み合わせた平衡化スケーリング: A → D_r A D_c, b → D_r b"""
    
    def __init__(self, max_iterations=5, tolerance=1e-8):
        """
        反復パラメータを指定して初期化
        
        Args:
            max_iterations: 平衡化の最大反復回数
            tolerance: 収束許容誤差
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        反復的平衡化スケーリングを適用
        """
        m, n = A.shape
        scaled_A = A.copy()
        
        # スケーリングベクトルを初期化
        row_scale = cp.ones(m)
        col_scale = cp.ones(n)
        
        # 行列のフォーマットに応じた処理戦略を選択
        is_csr = hasattr(scaled_A, 'format') and scaled_A.format == 'csr'
        is_csc = hasattr(scaled_A, 'format') and scaled_A.format == 'csc'
        
        # 反復処理
        for _ in range(self.max_iterations):
            # 行/列ノルムを計算
            row_norms = self._compute_row_norms(scaled_A, is_csr)
            col_norms = self._compute_column_norms(scaled_A, is_csc)
            
            # 収束確認
            if (cp.abs(row_norms - 1.0) < self.tolerance).all() and (cp.abs(col_norms - 1.0) < self.tolerance).all():
                break
            
            # スケーリング係数の更新（安定化のため平方根を使用）
            row_scale_update = 1.0 / cp.sqrt(cp.where(row_norms < 1e-15, 1.0, row_norms))
            col_scale_update = 1.0 / cp.sqrt(cp.where(col_norms < 1e-15, 1.0, col_norms))
            
            # 合計スケーリング係数を更新
            row_scale *= row_scale_update
            col_scale *= col_scale_update
            
            # 行列をスケーリング
            DR = sp.diags(row_scale_update)
            DC = sp.diags(col_scale_update)
            scaled_A = DR @ scaled_A @ DC
        
        # bをスケーリング
        scaled_b = b * row_scale
        
        return scaled_A, scaled_b, {'row_scale': row_scale, 'col_scale': col_scale}
    
    def _compute_row_norms(self, A, is_csr=False):
        """効率的な行ノルム計算"""
        m = A.shape[0]
        row_norms = cp.zeros(m)
        
        if is_csr:
            # CSRフォーマットではindptrを使って行単位で処理
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_norms[i] = cp.max(cp.abs(A.data[start:end]))
        else:
            # その他の形式
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = cp.max(cp.abs(row)) if row.size > 0 else 0.0
        
        return row_norms
    
    def _compute_column_norms(self, A, is_csc=False):
        """効率的な列ノルム計算"""
        n = A.shape[1]
        col_norms = cp.zeros(n)
        
        if is_csc:
            # CSCフォーマットではindptrを使って列単位で処理
            for j in range(n):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    col_norms[j] = cp.max(cp.abs(A.data[start:end]))
        else:
            # その他の形式
            for j in range(n):
                col = A[:, j]
                if hasattr(col, 'toarray'):
                    col = col.toarray().flatten()
                col_norms[j] = cp.max(cp.abs(col)) if col.size > 0 else 0.0
        
        return col_norms
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """解ベクトルをアンスケーリング"""
        col_scale = scale_info.get('col_scale')
        if col_scale is None:
            return x
        return x / col_scale
    
    def scale_b_only(self, b: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """右辺ベクトルbのみをスケーリング"""
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
        return b
    
    @property
    def name(self) -> str:
        return "EquilibrationScaling"
    
    @property
    def description(self) -> str:
        return "行と列のノルムをバランスさせる反復的平衡化スケーリング"