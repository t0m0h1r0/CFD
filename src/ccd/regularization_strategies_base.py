"""
正則化戦略の基底クラスモジュール

CCD法の正則化戦略の基底クラスを提供します。
"""

import jax.numpy as jnp
from typing import Callable, Tuple


class RegularizationStrategy:
    """正則化戦略の基底クラス"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
        """
        self.L = L
        self.K = K
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        正則化を適用し、正則化された行列とソルバー関数を返す
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        # デフォルトは正則化なし - 標準的な行列の解法
        def solver_func(rhs):
            return jnp.linalg.solve(self.L, rhs)
        
        return self.L, self.K, solver_func


class NoneRegularization(RegularizationStrategy):
    """正則化なし"""
    pass