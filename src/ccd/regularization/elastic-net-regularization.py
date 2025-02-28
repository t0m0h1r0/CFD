"""
Elastic Net 正則化戦略

CCD法のElastic Net 正則化を提供します。
L1正則化とL2正則化を組み合わせた手法です。
右辺ベクトルの変換と解の逆変換をサポートするように修正しました。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class ElasticNetRegularization(RegularizationStrategy):
    """Elastic Net 正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.alpha = kwargs.get('alpha', 1e-4)
        self.l1_ratio = kwargs.get('l1_ratio', 0.5)
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
                'help': '正則化パラメータの強さ'
            },
            'l1_ratio': {
                'type': float,
                'default': 0.5,
                'help': 'L1正則化の割合（0=L2のみ、1=L1のみ）'
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
        Elastic Net 正則化を適用
        
        L1正則化とL2正則化を組み合わせた手法で、スパース性を保ちながら
        相関の強い特徴間で選択の安定性を向上させます。
        
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
        
        # L1とL2の重みを計算
        alpha_l1 = alpha_scaled * self.l1_ratio
        alpha_l2 = alpha_scaled * (1 - self.l1_ratio)
        
        # L2正則化を考慮した行列の事前計算
        n = L_scaled.shape[1]
        L_reg = L_scaled + alpha_l2 * jnp.eye(n)
        
        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor
        
        return L_reg, inverse_transform
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """右辺ベクトルに正則化の変換を適用"""
        # 行列と同じスケーリングを右辺ベクトルにも適用
        return rhs * self.reg_factor


# 正則化戦略をレジストリに登録
regularization_registry.register("elastic_net", ElasticNetRegularization)