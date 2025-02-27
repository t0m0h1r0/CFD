"""
基本的なスケーリング戦略 - 正規化スケーリング

CCD法の基本的なスケーリング戦略（Normalization）を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class NormalizationScaling(ScalingStrategy):
    """行と列のノルムによるスケーリング"""
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        行と列のL2ノルムでスケーリングを適用する
        
        各要素を行と列のL2ノルムの平方根で割ることでスケーリング
        
        Returns:
            スケーリングされた行列L、逆変換関数
        """
        # 行と列のL2ノルムを計算
        row_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=1))
        col_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=0))
        
        # 0除算を防ぐため、非常に小さい値をクリップ
        row_norms = jnp.maximum(row_norms, 1e-10)
        col_norms = jnp.maximum(col_norms, 1e-10)
        
        # スケーリング行列を作成
        D_row = jnp.diag(1.0 / jnp.sqrt(row_norms))
        D_col = jnp.diag(1.0 / jnp.sqrt(col_norms))
        
        # スケーリングを適用
        L_scaled = D_row @ self.L @ D_col
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("normalization", NormalizationScaling)