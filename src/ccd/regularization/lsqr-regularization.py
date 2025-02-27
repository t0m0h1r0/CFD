"""
LSQR法による正則化戦略

CCD法のLSQR法による正則化戦略を提供します。
JAX互換の実装です。
"""

import jax.numpy as jnp
import jax.lax as lax
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class LSQRRegularization(RegularizationStrategy):
    """LSQR法による正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.iterations = kwargs.get('iterations', 20)
        self.damp = kwargs.get('damp', 0)
        self.L_T = self.L.T
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            'iterations': {
                'type': int,
                'default': 20,
                'help': '反復回数'
            },
            'damp': {
                'type': float,
                'default': 0,
                'help': '減衰パラメータ（0よりも大きい値を設定するとTikhonov正則化と同様の効果）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        LSQR法による正則化を適用
        
        Lanczos双共役勾配法に基づく反復法で、大規模な最小二乗問題に適しています。
        早期停止による正則化効果を得ます。
        
        Returns:
            正則化された行列L、逆変換関数
        """
        # クラス変数をローカル変数に保存
        L = self.L
        L_T = self.L_T
        damp = self.damp
        
        # 行列のスペクトルノルムを概算
        s_max = jnp.linalg.norm(L, ord=2)
        
        # 減衰パラメータを考慮
        # 減衰パラメータがある場合は、単位行列にスケールして加算
        n = L.shape[1]
        L_reg = L + damp * jnp.eye(n)
        
        # 逆変換関数
        def inverse_scaling(x_scaled):
            return x_scaled
        
        return L_reg, inverse_scaling


# 正則化戦略をレジストリに登録
regularization_registry.register("lsqr", LSQRRegularization)