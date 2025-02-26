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
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        行と列のL2ノルムでスケーリングを適用する
        
        各要素を行と列のL2ノルムの平方根で割ることでスケーリング
        
        Returns:
            スケーリングされた行列L、スケーリングされた行列K、逆変換関数
        """
        # 行と列のL2ノルムを計算
        row_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=1))
        col_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=0))
        
        # スケーリング行列を作成
        D = jnp.diag(1.0 / jnp.sqrt(row_norms * col_norms))
        
        # スケーリングを適用
        L_scaled = D @ self.L @ D
        K_scaled = D @ self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("normalization", NormalizationScaling)
