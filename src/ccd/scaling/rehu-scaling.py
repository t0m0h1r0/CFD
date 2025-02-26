"""
Rehu法によるスケーリング戦略

CCD法のRehu法（行と列の最大絶対値）によるスケーリング戦略を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class RehuScaling(ScalingStrategy):
    """Rehu法によるスケーリング"""
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        Rehu法（行と列の最大絶対値）によるスケーリングを適用
        
        各行と列の最大絶対値の平方根でスケーリングする効果的な方法
        
        Returns:
            スケーリングされた行列L、スケーリングされた行列K、逆変換関数
        """
        # 行列の各行と列の最大絶対値を計算
        max_values_row = jnp.max(jnp.abs(self.L), axis=1)
        max_values_col = jnp.max(jnp.abs(self.L), axis=0)
        
        # スケーリング行列を作成
        D_row = jnp.diag(1.0 / jnp.sqrt(max_values_row))
        D_col = jnp.diag(1.0 / jnp.sqrt(max_values_col))
        
        # スケーリングを適用
        L_scaled = D_row @ self.L @ D_col
        K_scaled = D_row @ self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("rehu", RehuScaling)
