"""
基本的な正則化戦略 - ティホノフ正則化

CCD法の基本的な正則化戦略（Tikhonov正則化）を提供します。
JAX互換の実装です。
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
        n = self.L.shape[0]
        # 単位行列を生成
        I = jnp.eye(n)
        # 行列を正則化
        L_reg = self.L + self.alpha * I
        
        # 逆変換関数（この場合は恒等関数）
        def inverse_scaling(x_scaled):
            return x_scaled
        
        return L_reg, inverse_scaling


# 正則化戦略をレジストリに登録
regularization_registry.register("tikhonov", TikhonovRegularization)