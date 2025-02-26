"""
LSQR法による正則化戦略

CCD法のLSQR法による正則化戦略を提供します。
"""

import jax.numpy as jnp
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
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        LSQR法による正則化を適用
        
        Lanczos双共役勾配法に基づく反復法で、大規模な最小二乗問題に適しています。
        早期停止による正則化効果を得ます。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        # ソルバー関数
        def solver_func(rhs):
            # 初期解を0に設定
            x = jnp.zeros_like(rhs)
            
            # 初期残差とその転置
            r = rhs - self.L @ x
            u = r
            beta = jnp.sqrt(jnp.sum(u * u))
            
            if beta > 0:
                u = u / beta
            
            v = self.L_T @ u
            alpha = jnp.sqrt(jnp.sum(v * v))
            
            if alpha > 0:
                v = v / alpha
            
            # Lanczos双共役勾配法の初期ベクトル
            w = v
            phi_bar = beta
            rho_bar = alpha
            
            # LSQR反復
            for i in range(self.iterations):
                # 双共役勾配法のステップ
                u_next = self.L @ v - alpha * u
                beta = jnp.sqrt(jnp.sum(u_next * u_next))
                
                if beta > 0:
                    u = u_next / beta
                else:
                    u = u_next
                
                v_next = self.L_T @ u - beta * v
                # 減衰パラメータを追加（Tikhonov正則化と同様の効果）
                if self.damp > 0:
                    v_next = v_next - self.damp * v
                
                alpha = jnp.sqrt(jnp.sum(v_next * v_next))
                
                if alpha > 0:
                    v = v_next / alpha
                else:
                    v = v_next
                
                # ギブンス回転の適用
                rho = jnp.sqrt(rho_bar**2 + beta**2)
                c = rho_bar / rho
                s = beta / rho
                theta = s * alpha
                rho_bar = -c * alpha
                phi = c * phi_bar
                phi_bar = s * phi_bar
                
                # 解の更新
                x = x + (phi / rho) * w
                w = v - (theta / rho) * w
            
            return x
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("lsqr", LSQRRegularization)
