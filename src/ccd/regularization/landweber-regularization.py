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
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        Landweber反復法による正則化を適用
        
        反復法に基づく正則化手法で、反復回数を制限することで正則化効果を得ます。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        # クラス変数をローカル変数に保存（ループ内での参照のため）
        L = self.L
        L_T = self.L_T
        
        # 行列のスペクトルノルムを概算
        s_max = jnp.linalg.norm(L, ord=2)
        
        # 緩和パラメータを安全な範囲に調整
        omega = jnp.minimum(self.relaxation, 1.9 / (s_max ** 2))
        
        # JAX互換のソルバー関数
        def solver_func(rhs):
            # 初期解を0に設定
            x_init = jnp.zeros_like(rhs)
            
            # Landweber反復のボディ関数
            def landweber_body(i, x):
                # 残差: r = rhs - L @ x
                residual = rhs - L @ x
                # 反復更新: x = x + omega * L^T @ residual
                x_new = x + omega * (L_T @ residual)
                return x_new
            
            # Landweber反復を実行
            final_x = lax.fori_loop(0, self.iterations, landweber_body, x_init)
            
            return final_x
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("landweber", LandweberRegularization)