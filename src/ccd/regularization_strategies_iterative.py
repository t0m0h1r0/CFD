"""
反復的な正則化戦略モジュール

CCD法の反復的な正則化戦略（Landweber, Precomputed Landweber, LSQR）を提供します。
"""

import jax.numpy as jnp

from regularization_strategies_base import RegularizationStrategy


class LandweberRegularization(RegularizationStrategy):
    """Landweber反復法による正則化"""
    
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
        omega = min(self.relaxation, 1.9 / (s_max ** 2))
        
        # ソルバー関数
        def solver_func(rhs):
            # 初期解を0に設定
            x = jnp.zeros_like(rhs)
            
            # Landweber反復
            for _ in range(self.iterations):
                # 残差: r = rhs - L @ x
                residual = rhs - self.L @ x
                # 反復更新: x = x + omega * L^T @ residual
                x = x + omega * (self.L_T @ residual)
            
            return x
        
        return self.L, self.K, solver_func


class PrecomputedLandweberRegularization(RegularizationStrategy):
    """事前計算型のLandweber反復法による正則化"""
    
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
        omega = min(self.relaxation, 1.9 / (s_max ** 2))
        
        n = self.L.shape[0]
        I = jnp.eye(n)
        
        # Landweber反復のマトリックス形式
        LTL = self.L_T @ self.L
        M = I - omega * LTL
        
        # M^n を計算
        M_power = I
        for _ in range(self.iterations):
            M_power = M_power @ M
        
        # 最終的な変換行列を計算
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


class LSQRRegularization(RegularizationStrategy):
    """LSQR法による正則化"""
    
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