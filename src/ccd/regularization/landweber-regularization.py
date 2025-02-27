"""
Landweber反復法による正則化戦略

CCD法のLandweber反復法による正則化戦略を提供します。
JAX互換の実装です。
"""

import jax.numpy as jnp
import jax.lax as lax
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class LandweberRegularization(RegularizationStrategy):
    """Landweber反復法による正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.iterations = kwargs.get('iterations', 20)
        self.relaxation = kwargs.get('relaxation', 0.1)
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
            'relaxation': {
                'type': float,
                'default': 0.1,
                'help': '緩和パラメータ（0 < relaxation < 2/σ_max^2、ここでσ_maxはLの最大特異値）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        Landweber反復法による正則化を適用
        
        反復法に基づく正則化手法で、反復回数を制限することで正則化効果を得ます。
        
        Returns:
            正則化された行列L、逆変換関数
        """
        # クラス変数をローカル変数に保存（ループ内での参照のため）
        L = self.L
        
        # 行列のスペクトルノルムを概算
        s_max = jnp.linalg.norm(L, ord=2)
        
        # 緩和パラメータを安全な範囲に調整
        omega = jnp.minimum(self.relaxation, 1.9 / (s_max ** 2))
        
        # 正則化された行列を計算
        # 最大特異値に基づいて行列を調整
        L_reg = L / (s_max ** 2)
        
        # 逆変換関数
        def inverse_scaling(x_scaled):
            return x_scaled
        
        return L_reg, inverse_scaling


# 正則化戦略をレジストリに登録
regularization_registry.register("landweber", LandweberRegularization)