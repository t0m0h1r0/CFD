"""
行スケーリング手法の実装（バックエンド非依存版）
"""

from typing import Dict, Any, Tuple
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
        # 配列モジュールの取得（NumPyまたはCuPy）
        xp = self._get_array_module(A)
        
        # 行ノルムを計算
        row_norms = self._compute_row_norms(A)
        
        # 数値的安定性のための処理（0除算を避ける）
        row_scale = 1.0 / self._maximum(row_norms, 1e-15)
        
        # スケーリング行列を作成し、A に適用
        D_inv = self._diags(row_scale)
        scaled_A = D_inv @ A
        
        # bのスケーリング
        scaled_b = b * row_scale
        
        return scaled_A, scaled_b, {'row_scale': row_scale}
    
    def _compute_row_norms(self, A):
        """
        行列の各行のノルムを計算
        
        Args:
            A: 行列（NumPyまたはCuPy）
            
        Returns:
            row_norms: 各行のノルム
        """
        xp = self._get_array_module(A)
        is_sparse = self._is_sparse(A)
        m = A.shape[0]
        
        # 行ノームを保存する配列
        row_norms = self._zeros(m, dtype=A.dtype if hasattr(A, 'dtype') else None, array_ref=A)
        
        if is_sparse and hasattr(A, 'format') and A.format == 'csr':
            # CSR形式の効率的な処理
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_data = A.data[start:end]
                    if self.norm_type == float('inf'):
                        row_norms[i] = xp.max(xp.abs(row_data))
                    else:
                        # L2ノームなど
                        if hasattr(xp, 'linalg') and hasattr(xp.linalg, 'norm'):
                            row_norms[i] = xp.linalg.norm(row_data, ord=self.norm_type)
                        else:
                            # フォールバック
                            if self.norm_type == 2:
                                row_norms[i] = xp.sqrt(xp.sum(row_data * row_data))
                            else:
                                row_norms[i] = xp.sum(xp.abs(row_data) ** self.norm_type) ** (1.0 / self.norm_type)
        else:
            # 非CSR形式または密行列
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                
                if self.norm_type == float('inf'):
                    row_norms[i] = xp.max(xp.abs(row))
                else:
                    # L2ノームなど
                    if hasattr(xp, 'linalg') and hasattr(xp.linalg, 'norm'):
                        row_norms[i] = xp.linalg.norm(row, ord=self.norm_type)
                    else:
                        # フォールバック
                        if self.norm_type == 2:
                            row_norms[i] = xp.sqrt(xp.sum(row * row))
                        else:
                            row_norms[i] = xp.sum(xp.abs(row) ** self.norm_type) ** (1.0 / self.norm_type)
        
        return row_norms
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """行スケーリングは解ベクトルに影響しない"""
        return x
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """右辺ベクトルbのみをスケーリング"""
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