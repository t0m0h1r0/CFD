"""
基本的な正則化戦略モジュール

CCD法の基本的な正則化戦略（Tikhonov）を提供します。
"""

import jax.numpy as jnp

from regularization_strategies_base import RegularizationStrategy


class TikhonovRegularization(RegularizationStrategy):
    """Tikhonov正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, alpha: float = 1e-6):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            alpha: 正則化パラメータ（大きいほど正則化の効果が強くなる）
        """
        super().__init__(L, K)
        self.alpha = alpha
    
    def apply_regularization(self):
        """
        Tikhonov正則化を適用
        
        行列に単位行列の定数倍を加算することで、数値的な安定性を向上させる
        正則化手法です。α値が大きいほど正則化の効果が強くなります。
        """
        n = self.L.shape[0]
        # 単位行列を生成
        I = jnp.eye(n)
        # 行列を正則化
        L_reg = self.L + self.alpha * I
        
        # ソルバー関数
        def solver_func(rhs):
            return jnp.linalg.solve(L_reg, rhs)
        
        return L_reg, self.K, solver_func