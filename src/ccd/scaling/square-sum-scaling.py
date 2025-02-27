"""
二乗和スケーリング戦略（Sum of Squares Scaling）

CCD法の二乗和スケーリング戦略を提供します。
各行と列の要素の二乗和が等しくなるようスケーリングします。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class SquareSumScaling(ScalingStrategy):
    """二乗和スケーリング（Sum of Squares Scaling）"""
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        二乗和スケーリングを適用
        
        各行と列の要素の二乗和が等しくなるようスケーリングします。
        特異値分解の前処理として効果的です。
        
        Returns:
            スケーリングされた行列L、逆変換関数
        """
        # 1. 行の二乗和を計算
        row_sqr_sums = jnp.sqrt(jnp.sum(self.L * self.L, axis=1))
        
        # 0除算を防ぐため、非常に小さい値をクリップ
        row_sqr_sums = jnp.maximum(row_sqr_sums, 1e-10)
        
        D_row = jnp.diag(1.0 / row_sqr_sums)
        L_row_scaled = D_row @ self.L
        
        # 2. 列の二乗和を計算
        col_sqr_sums = jnp.sqrt(jnp.sum(L_row_scaled * L_row_scaled, axis=0))
        
        # 0除算を防ぐため、非常に小さい値をクリップ
        col_sqr_sums = jnp.maximum(col_sqr_sums, 1e-10)
        
        D_col = jnp.diag(1.0 / col_sqr_sums)
        
        # 3. スケーリングを適用
        L_scaled = L_row_scaled @ D_col
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("square_sum", SquareSumScaling)