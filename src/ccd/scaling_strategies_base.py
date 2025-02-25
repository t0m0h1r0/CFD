"""
スケーリング戦略の基底クラスモジュール

CCD法のスケーリング戦略の基底クラスを提供します。
"""

import jax.numpy as jnp
from typing import Callable, Tuple


class ScalingStrategy:
    """スケーリング戦略の基底クラス"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray):
        """
        Args:
            L: スケーリングする行列
            K: スケーリングする右辺行列
        """
        self.L = L
        self.K = K
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        スケーリングを適用し、スケーリングされた行列と逆変換関数を返す
        
        Returns:
            スケーリングされた行列L、スケーリングされた行列K、逆変換関数
        """
        # デフォルトはスケーリングなし
        return self.L, self.K, lambda x: x


class NoneScaling(ScalingStrategy):
    """スケーリングなし"""
    pass