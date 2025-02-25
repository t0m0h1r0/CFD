"""
反復的な正則化戦略モジュール

CCD法の反復的な正則化戦略（Landweber, Precomputed Landweber, LSQR）を提供します。
JAXのJIT互換に修正しています。
"""

import jax.numpy as jnp
from jax import lax

from regularization_strategies_base import RegularizationStrategy


class LandweberRegularization(RegularizationStrategy):
    """Landweber反復法による正則化（JAX JIT対応版）"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, iterations: int = 20, relaxation: float = 0.1):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            iterations: 反復回数
            relaxation: 緩和パラメータ（0 < relaxation < 2/σ_max^2、ここでσ_maxはLの最大特異値）
        """
        super().__init__(L, K)
        self.iterations = iterations
        self.relaxation = relaxation
        self.L_T = L.T
    
    def apply_regularization(self):
        """
        Landweber反復法による正則化を適用
        
        反復法に基づく正則化手法で、反復回数を制限することで正則化効果を得ます。
        """
        # 行列のスペクトルノルムを概算
        s_max = jnp.linalg.norm(self.L, ord=2)
        
        # 緩和パラメータを安全な範囲に調整
        omega = jnp.minimum(self.relaxation, 1.9 / (s_max ** 2))
        
        # ソルバー関数
        def solver_func(rhs):
            # 初期解を0に設定
            x = jnp.zeros_like(rhs)
            
            # JAX対応のLandweber反復
            def body_fun(x_prev, _):
                # 残差: r = rhs - L @ x
                residual = rhs - self.L @ x_prev
                # 反復更新: x = x + omega * L^T @ residual
                x_new = x_prev + omega * (self.L_T @ residual)
                return x_new, None
            
            # lax.scanで反復処理を実行
            x_final, _ = lax.scan(body_fun, x, jnp.arange(self.iterations))
            
            return x_final
        
        return self.L, self.K, solver_func


class PrecomputedLandweberRegularization(RegularizationStrategy):
    """事前計算型のLandweber反復法による正則化（JAX JIT対応版）"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, iterations: int = 20, relaxation: float = 0.1):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            iterations: 反復回数
            relaxation: 緩和パラメータ（0 < relaxation < 2/σ_max^2、ここでσ_maxはLの最大特異値）
        """
        super().__init__(L, K)
        self.iterations = iterations
        self.relaxation = relaxation
        self.L_T = L.T
    
    def apply_regularization(self):
        """
        事前計算型のLandweber反復法による正則化を適用
        
        Landweber反復を行列演算として事前に計算し、効率化した正則化手法です。
        """
        # 行列のスペクトルノルムを概算
        s_max = jnp.linalg.norm(self.L, ord=2)
        
        # 緩和パラメータを安全な範囲に調整
        omega = jnp.minimum(self.relaxation, 1.9 / (s_max ** 2))
        
        n = self.L.shape[0]
        I = jnp.eye(n)
        
        # Landweber反復のマトリックス形式
        LTL = self.L_T @ self.L
        M = I - omega * LTL
        
        # M^n を計算（JAX対応）
        def mat_power(M, n):
            def body_fun(M_pow, _):
                return M_pow @ M, None
            
            M_power, _ = lax.scan(body_fun, I, jnp.arange(n))
            return M_power
        
        # 最終的な変換行列を計算
        M_power = mat_power(M, self.iterations)
        I_minus_M_power = I - M_power
        
        # LTLの擬似逆行列を計算
        U, s, Vh = jnp.linalg.svd(LTL, full_matrices=False)
        threshold = jnp.max(s) * 1e-10
        s_inv = jnp.where(s > threshold, 1.0 / s, 0.0)
        LTL_pinv = Vh.T @ jnp.diag(s_inv) @ U.T
        
        # 最終的な変換行列
        transform_matrix = I_minus_M_power @ LTL_pinv @ self.L_T
        
        # ソルバー関数
        def solver_func(rhs):
            return transform_matrix @ rhs
        
        return self.L, self.K, solver_func


"""
LSQRRegularizationクラスの最終修正
"""

class LSQRRegularization(RegularizationStrategy):
    """LSQR法による正則化（JAX JIT対応版）"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, iterations: int = 20, damp: float = 0):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            iterations: 反復回数
            damp: 減衰パラメータ（0よりも大きい値を設定するとTikhonov正則化と同様の効果）
        """
        super().__init__(L, K)
        self.iterations = iterations
        self.damp = damp
        self.L_T = L.T
    
    def apply_regularization(self):
        """
        LSQR法による正則化を適用
        
        Lanczos双共役勾配法に基づく反復法で、大規模な最小二乗問題に適しています。
        早期停止による正則化効果を得ます。
        """
        # ソルバー関数
        def solver_func(rhs):
            # 初期解を0に設定
            x = jnp.zeros(self.L.shape[1])
            
            # 初期残差とその転置
            r = rhs - self.L @ x
            u = r
            beta = jnp.sqrt(jnp.sum(u * u))
            
            # 小さい値で0除算を防止
            u = jnp.where(beta > 1e-15, u / beta, u)
            
            v = self.L_T @ u
            alpha = jnp.sqrt(jnp.sum(v * v))
            
            # 小さい値で0除算を防止
            v = jnp.where(alpha > 1e-15, v / alpha, v)
            
            # Lanczos双共役勾配法の初期ベクトル
            w = v
            phi_bar = beta
            rho_bar = alpha
            
            # LSQR反復（JAX対応）
            def body_fun(carry, _):
                x, u, v, w, phi_bar, rho_bar, alpha_val = carry
                
                # 双共役勾配法のステップ
                u_next = self.L @ v - alpha_val * u
                beta = jnp.sqrt(jnp.sum(u_next * u_next))
                
                # 0除算対策
                u_new = jnp.where(beta > 1e-15, u_next / beta, u_next)
                
                v_next = self.L_T @ u_new - beta * v
                alpha_new = jnp.sqrt(jnp.sum(v_next * v_next))
                
                # 0除算対策
                v_new = jnp.where(alpha_new > 1e-15, v_next / alpha_new, v_next)
                
                # ギブンス回転の適用
                rho = jnp.sqrt(rho_bar**2 + beta**2)
                c = rho_bar / jnp.where(rho > 1e-15, rho, 1.0)
                s = beta / jnp.where(rho > 1e-15, rho, 1.0)
                theta = s * alpha_new
                rho_bar_new = -c * alpha_new
                phi = c * phi_bar
                phi_bar_new = s * phi_bar
                
                # 解の更新
                x_update = (phi / jnp.where(rho > 1e-15, rho, 1.0)) * w
                x_new = x + x_update
                w_new = v_new - (theta / jnp.where(rho > 1e-15, rho, 1.0)) * w
                
                return (x_new, u_new, v_new, w_new, phi_bar_new, rho_bar_new, alpha_new), None
            
            # lax.scanで反復処理を実行
            initial_carry = (x, u, v, w, phi_bar, rho_bar, alpha)
            (x_final, _, _, _, _, _, _), _ = lax.scan(body_fun, initial_carry, jnp.arange(self.iterations))
            
            return x_final
        
        return self.L, self.K, solver_func