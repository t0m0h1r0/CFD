"""
高度な正則化戦略モジュール（バグ修正版）

CCD法の高度な正則化戦略（Total Variation, L1, Elastic Net）を提供します。
JAXのJIT互換に修正しています。
"""

import jax.numpy as jnp
from jax import lax

from regularization_strategies_base import RegularizationStrategy


class TotalVariationRegularization(RegularizationStrategy):
    """Total Variation 正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, alpha: float = 1e-4, iterations: int = 50, tol: float = 1e-6):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            alpha: 正則化パラメータ
            iterations: 反復回数
            tol: 収束判定閾値
        """
        super().__init__(L, K)
        self.alpha = alpha
        self.iterations = iterations
        self.tol = tol
        self.L_T = L.T
    
    def apply_regularization(self):
        """
        Total Variation 正則化を適用
        
        解の微分に対するペナルティを課す正則化手法で、
        不連続点を保存しながらノイズを除去する特徴があります。
        """
        # 差分行列（1階微分演算子）の構築
        n = self.L.shape[1] // 3  # 各グリッド点の自由度は3
        
        # 各導関数成分に対する差分行列
        D_blocks = []
        for i in range(3):  # f', f'', f''' の各成分
            D = jnp.zeros((n-1, n))
            for j in range(n-1):
                D = D.at[j, j].set(-1)
                D = D.at[j, j+1].set(1)
            D_blocks.append(D)
        
        # ブロック対角行列として構築
        D = jnp.zeros((3*(n-1), 3*n))
        for i in range(3):
            D = D.at[i*(n-1):(i+1)*(n-1), i*n:(i+1)*n].set(D_blocks[i])
        
        # ADMM用の前処理
        DTD = D.T @ D
        LTL = self.L_T @ self.L
        A = LTL + self.alpha * DTD
        
        # ソルバー関数（ADMM: Alternating Direction Method of Multipliers）
        def solver_func(rhs):
            # 初期解を標準的な最小二乗解に設定
            x = jnp.linalg.lstsq(self.L, rhs, rcond=None)[0]
            z = D @ x
            u = jnp.zeros_like(z)  # 双対変数
            
            # JAX対応の反復処理：lax.scanを使用
            def body_fun(carry, _):
                x_prev, z_prev, u_prev = carry
                
                # x-update (最小二乗問題)
                b = self.L_T @ rhs + self.alpha * D.T @ (z_prev - u_prev)
                x_new = jnp.linalg.solve(A, b)
                
                # z-update (縮小演算子)
                Dx = D @ x_new
                z_new = self._shrinkage(Dx + u_prev, 1.0/self.alpha)
                
                # u-update (双対変数の更新)
                u_new = u_prev + Dx - z_new
                
                return (x_new, z_new, u_new), None
            
            # lax.scanで反復処理を実行
            (x_final, _, _), _ = lax.scan(body_fun, (x, z, u), jnp.arange(self.iterations))
            return x_final
        
        return self.L, self.K, solver_func
    
    def _shrinkage(self, x, threshold):
        """縮小演算子（soft thresholding）"""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)


class L1Regularization(RegularizationStrategy):
    """L1正則化（LASSO）"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, alpha: float = 1e-4, iterations: int = 100, tol: float = 1e-6):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            alpha: 正則化パラメータ
            iterations: 反復回数
            tol: 収束判定閾値
        """
        super().__init__(L, K)
        self.alpha = alpha
        self.iterations = iterations
        self.tol = tol
        self.L_T = L.T
    
    def apply_regularization(self):
        """
        L1正則化（LASSO）を適用
        
        解のL1ノルムに対するペナルティを課す正則化手法で、
        スパース性（多くの要素がゼロ）を持つ解を生成します。
        """
        # ソルバー関数（近位勾配法）
        def solver_func(rhs):
            # 初期解を標準的な最小二乗解に設定
            x = jnp.linalg.lstsq(self.L, rhs, rcond=None)[0]
            
            # 行列AとATAの事前計算
            ATA = self.L_T @ self.L
            ATb = self.L_T @ rhs
            
            # リプシッツ定数の推定（勾配降下法のステップサイズに関連）
            lambda_max = jnp.linalg.norm(ATA, ord=2)
            step_size = 1.0 / lambda_max
            
            # JAX対応の反復処理：lax.scanを使用
            def body_fun(x_prev, _):
                # 勾配ステップ
                grad = ATA @ x_prev - ATb
                x_grad = x_prev - step_size * grad
                
                # 近位演算子ステップ（軟閾値処理）
                x_new = self._soft_threshold(x_grad, self.alpha * step_size)
                return x_new, None
            
            # lax.scanで反復処理を実行
            x_final, _ = lax.scan(body_fun, x, jnp.arange(self.iterations))
            return x_final
        
        return self.L, self.K, solver_func
    
    def _soft_threshold(self, x, threshold):
        """軟閾値処理（soft thresholding）"""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)


class ElasticNetRegularization(RegularizationStrategy):
    """Elastic Net 正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, alpha: float = 1e-4, l1_ratio: float = 0.5, 
                 iterations: int = 100, tol: float = 1e-6):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            alpha: 正則化パラメータの強さ
            l1_ratio: L1正則化の割合（0=L2のみ、1=L1のみ）
            iterations: 反復回数
            tol: 収束判定閾値
        """
        super().__init__(L, K)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.iterations = iterations
        self.tol = tol
        self.L_T = L.T
    
    def apply_regularization(self):
        """
        Elastic Net 正則化を適用
        
        L1正則化とL2正則化を組み合わせた手法で、スパース性を保ちながら
        相関の強い特徴間で選択の安定性を向上させます。
        """
        # L1とL2の重みを計算
        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1 - self.l1_ratio)
        
        # ソルバー関数（近位勾配法）
        def solver_func(rhs):
            # 初期解を標準的な最小二乗解に設定
            x = jnp.linalg.lstsq(self.L, rhs, rcond=None)[0]
            
            # L2正則化を考慮した行列の事前計算
            n = self.L.shape[1]
            ATA_l2 = self.L_T @ self.L + alpha_l2 * jnp.eye(n)
            ATb = self.L_T @ rhs
            
            # リプシッツ定数の推定
            lambda_max = jnp.linalg.norm(ATA_l2, ord=2)
            step_size = 1.0 / lambda_max
            
            # JAX対応の反復処理：lax.scanを使用
            def body_fun(x_prev, _):
                # 勾配ステップ（L2正則化項を含む）
                grad = ATA_l2 @ x_prev - ATb
                x_grad = x_prev - step_size * grad
                
                # 近位演算子ステップ（L1正則化のみ）
                x_new = self._soft_threshold(x_grad, alpha_l1 * step_size)
                return x_new, None
            
            # lax.scanで反復処理を実行
            x_final, _ = lax.scan(body_fun, x, jnp.arange(self.iterations))
            return x_final
        
        return self.L, self.K, solver_func
    
    def _soft_threshold(self, x, threshold):
        """軟閾値処理（soft thresholding）"""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)