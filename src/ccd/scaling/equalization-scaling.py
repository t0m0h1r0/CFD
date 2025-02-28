"""
均等化スケーリング戦略

CCD法の均等化スケーリング戦略を提供します。
各行と列の最大絶対値を1にスケーリングします。
右辺ベクトルのスケーリングにも対応しました。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class EqualizationScaling(ScalingStrategy):
    """均等化スケーリング"""
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        均等化スケーリングを適用
        
        各行と列の最大絶対値を1にスケーリングします。
        行列の要素間のスケールを均一化することで数値的な安定性を向上させます。
        
        Returns:
            スケーリングされた行列L、逆変換関数
        """
        # 1. 行の均等化
        row_max = jnp.max(jnp.abs(self.L), axis=1)
        # 0除算を防ぐため、非常に小さい値をクリップ
        row_max = jnp.maximum(row_max, 1e-10)
        D_row = jnp.diag(1.0 / row_max)
        L_row_eq = D_row @ self.L
        
        # 2. 列の均等化
        col_max = jnp.max(jnp.abs(L_row_eq), axis=0)
        # 0除算を防ぐため、非常に小さい値をクリップ
        col_max = jnp.maximum(col_max, 1e-10)
        D_col = jnp.diag(1.0 / col_max)
        
        # 3. スケーリングを適用
        L_scaled = L_row_eq @ D_col
        
        # スケーリング行列を保存
        self.scaling_matrix_row = D_row
        self.scaling_matrix_col = D_col
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("equalization", EqualizationScaling)