"""
基本的な正則化戦略 - ティホノフ正則化

CCD法の基本的な正則化戦略（Tikhonov正則化）を提供します。
右辺ベクトルの変換と解の逆変換をサポートするように修正しました。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class TikhonovRegularization(RegularizationStrategy):
    """Tikhonov正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.alpha = kwargs.get('alpha', 1e-6)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            'alpha': {
                'type': float,
                'default': 1e-6,
                'help': '正則化パラメータ（大きいほど正則化の効果が強くなる）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        Tikhonov正則化を適用
        
        行列に単位行列の定数倍を加算することで、数値的な安定性を向上させる
        正則化手法です。α値が大きいほど正則化の効果が強くなります。
        
        Returns:
            正則化された行列L、逆変換関数
        """
        # 行列のスケールを確認
        matrix_norm = jnp.linalg.norm(self.L, ord=2)
        
        # 行列のスケールに対する正則化パラメータの比率
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.L * self.reg_factor
            alpha_scaled = self.alpha * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.L
            alpha_scaled = self.alpha
        
        n = L_scaled.shape[0]
        # 単位行列を生成
        I = jnp.eye(n)
        # 行列を正則化
        L_reg = L_scaled + alpha_scaled * I
        
        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor
        
        return L_reg, inverse_transform
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """右辺ベクトルに正則化の変換を適用"""
        # 行列と同じスケーリングを右辺ベクトルにも適用
        return rhs * self.reg_factor


# 正則化戦略をレジストリに登録
regularization_registry.register("tikhonov", TikhonovRegularization)