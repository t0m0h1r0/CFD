"""
Van der Sluis スケーリング戦略

CCD法のVan der Sluis スケーリング戦略を提供します。
各列を2-ノルムで正規化します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class VanDerSluisScaling(ScalingStrategy):
    """Van der Sluis スケーリング"""
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        Van der Sluis スケーリングを適用
        
        各列を2-ノルムで正規化するスケーリング手法です。
        行列の列間でスケールが大きく異なる場合に特に効果的です。
        
        Returns:
            スケーリングされた行列L、スケーリングされた行列K、逆変換関数
        """
        # 各列の2-ノルムを計算
        col_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=0))
        
        # 0除算を防ぐため、非常に小さい値をクリップ
        col_norms = jnp.maximum(col_norms, 1e-10)
        
        # スケーリング行列を作成（列のみスケーリング）
        D_col = jnp.diag(1.0 / col_norms)
        
        # スケーリングを適用
        L_scaled = self.L @ D_col
        K_scaled = self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("van_der_sluis", VanDerSluisScaling)
