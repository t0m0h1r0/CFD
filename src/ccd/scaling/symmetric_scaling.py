"""
対角要素を使用した対称スケーリング実装
"""

from typing import Dict, Any, Tuple
from .base import BaseScaling


class SymmetricScaling(BaseScaling):
    """対称スケーリング手法: A → D⁻¹/² A D⁻¹/², b → D⁻¹/² b (Dは対角要素の絶対値)"""
    
    def __init__(self, backend='numpy'):
        """
        初期化
        
        Args:
            backend: 計算バックエンド ('numpy', 'cupy', 'jax')
        """
        super().__init__(backend)
    
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        対角要素を使ってAを対称的にスケーリング
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 対角要素を取得
        diag = A.diagonal()
        
        # 数値的安定性のための処理
        D_sqrt_inv = self.array_utils.sqrt(1.0 / self.array_utils.where(
            self.array_utils.abs(diag) < 1e-15, 1.0, self.array_utils.abs(diag)))
        
        # スケーリング行列を構築
        D_sqrt_inv_mat = self.array_utils.diags(D_sqrt_inv)
        
        # スケーリング適用
        scaled_A = D_sqrt_inv_mat @ A @ D_sqrt_inv_mat
        scaled_b = D_sqrt_inv_mat @ b
        
        return scaled_A, scaled_b, {'D_sqrt_inv': D_sqrt_inv}
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """解ベクトルをアンスケーリング"""
        D_sqrt_inv = scale_info.get('D_sqrt_inv')
        if D_sqrt_inv is None:
            return x
        return x / D_sqrt_inv
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """右辺ベクトルbのみをスケーリング"""
        D_sqrt_inv = scale_info.get('D_sqrt_inv')
        if D_sqrt_inv is None:
            return b
        return b * D_sqrt_inv
    
    @property
    def name(self) -> str:
        return "SymmetricScaling"
    
    @property
    def description(self) -> str:
        return "対角要素を使用した対称スケーリング (D^-1/2 * A * D^-1/2)"
