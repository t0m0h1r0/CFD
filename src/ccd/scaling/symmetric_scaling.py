"""
対角要素を使用した対称スケーリング実装
"""

from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class SymmetricScaling(BaseScaling):
    """対称スケーリング手法: A → D⁻¹/² A D⁻¹/², b → D⁻¹/² b (Dは対角要素の絶対値)"""
    
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        対角要素を使ってAを対称的にスケーリング
        """
        # 対角要素を取得
        diag = A.diagonal()
        
        # 数値的安定性のための処理
        D_sqrt_inv = cp.sqrt(1.0 / cp.where(cp.abs(diag) < 1e-15, 1.0, cp.abs(diag)))
        
        # スケーリング行列を構築
        D_sqrt_inv_mat = sp.diags(D_sqrt_inv)
        
        # スケーリング適用
        scaled_A = D_sqrt_inv_mat @ A @ D_sqrt_inv_mat
        scaled_b = D_sqrt_inv_mat @ b
        
        return scaled_A, scaled_b, {'D_sqrt_inv': D_sqrt_inv}
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """解ベクトルをアンスケーリング"""
        D_sqrt_inv = scale_info.get('D_sqrt_inv')
        if D_sqrt_inv is None:
            return x
        return x / D_sqrt_inv
    
    def scale_b_only(self, b: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
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