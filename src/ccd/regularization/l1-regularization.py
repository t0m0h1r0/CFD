"""
L1正則化戦略（LASSO）

CCD法のL1正則化（LASSO）を提供します。
解のL1ノルムに対するペナルティを課します。
右辺ベクトルの変換と解の逆変換をサポートするように修正しました。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class L1Regularization(RegularizationStrategy):
    """L1正則化（LASSO）"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.alpha = kwargs.get('alpha', 1e-4)
        self.iterations = kwargs.get('iterations', 100)
        self.tol = kwargs.get('tol', 1e-6)
        self.L_T = self.L.T
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            'alpha': {
                'type': float,
                'default': 1e-4,
                'help': '正則化パラメータ'
            },
            'iterations': {
                'type': int,
                'default': 100,
                'help': '反復回数'
            },
            'tol': {
                'type': float,
                'default': 1e-6,
                'help': '収束判定閾値'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        L1正則化（LASSO）を適用
        
        解のL1ノルムに対するペナルティを課す正則化手法で、
        スパース性（多くの要素がゼロ）を持つ解を生成します。
        
        Returns:
            正則化された行列L、逆変換関数
        """
        # 行列のスケールを確認
        matrix_norm = jnp.linalg.norm(self.L, ord=2)
        
        # 行列のスケールが大きい場合はスケーリング
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.L * self.reg_factor
            alpha_scaled = self.alpha * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.L
            alpha_scaled = self.alpha
        
        # L1正則化では対角成分に正則化パラメータを加算
        n = L_scaled.shape[1]
        # 単位行列に正則化パラメータをスケールして加算
        L_reg = L_scaled + alpha_scaled * jnp.eye(n)
        
        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor
        
        return L_reg, inverse_transform
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """右辺ベクトルに正則化の変換を適用"""
        # 行列と同じスケーリングを右辺ベクトルにも適用
        return rhs * self.reg_factor


# 正則化戦略をレジストリに登録
regularization_registry.register("l1", L1Regularization)