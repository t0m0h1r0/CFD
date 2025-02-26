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
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        LSQR法による正則化を適用
        
        Lanczos双共役勾配法に基づく反復法で、大規模な最小二乗問題に適しています。
        早期停止による正則化効果を得ます。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        L = self.L
        L_T = self.L_T
        damp = self.damp

        # ソルバー関数をJAX互換にする
        def solver_func(rhs):
            # 初期値
            x = jnp.zeros_like(rhs)
            u = rhs - L @ x
            beta = jnp.sqrt(jnp.sum(u * u))
            
            # JAX互換の条件付き処理
            u = jnp.where(beta > 0, u / beta, u)
            
            v = L_T @ u
            alpha = jnp.sqrt(jnp.sum(v * v))
            
            # JAX互換の条件付き処理
            v = jnp.where(alpha > 0, v / alpha, v)
            
            # Lanczos双共役勾配法の初期ベクトル
            w = v
            phi_bar = beta
            rho_bar = alpha
            
            # LSQR反復をJAX互換のループで実装
            # すべての状態変数を引数としてもつループ関数
            def lsqr_body(i, loop_state):
                x, u, v, w, phi_bar, rho_bar, alpha_prev = loop_state
                
                # 双共役勾配法のステップ
                u_next = L @ v - alpha_prev * u
                beta = jnp.sqrt(jnp.sum(u_next * u_next))
                
                # JAX互換の条件付き処理
                u_new = jnp.where(beta > 0, u_next / beta, u_next)
                
                v_next = L_T @ u_new - beta * v
                # 減衰パラメータを追加（Tikhonov正則化と同様の効果）
                v_next = v_next - damp * v
                
                alpha_new = jnp.sqrt(jnp.sum(v_next * v_next))
                
                # JAX互換の条件付き処理
                v_new = jnp.where(alpha_new > 0, v_next / alpha_new, v_next)
                
                # ギブンス回転の適用
                rho = jnp.sqrt(rho_bar**2 + beta**2)
                c = rho_bar / rho
                s = beta / rho
                theta = s * alpha_new
                rho_bar_new = -c * alpha_new
                phi = c * phi_bar
                phi_bar_new = s * phi_bar
                
                # 解の更新
                x_new = x + (phi / rho) * w
                w_new = v_new - (theta / rho) * w
                
                return (x_new, u_new, v_new, w_new, phi_bar_new, rho_bar_new, alpha_new)
            
            # 初期状態にalphaも含める
            init_state = (x, u, v, w, phi_bar, rho_bar, alpha)
            final_state = lax.fori_loop(0, self.iterations, lsqr_body, init_state)
            
            # 最終的な解を取得
            final_x = final_state[0]
            return final_x
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("lsqr", LSQRRegularization)