"""
Total Variation 正則化戦略

CCD法のTotal Variation 正則化を提供します。
解の微分に対するペナルティを課し、不連続点を保存しながらノイズを除去します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class TotalVariationRegularization(RegularizationStrategy):
    """Total Variation 正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.alpha = kwargs.get('alpha', 1e-4)
        self.iterations = kwargs.get('iterations', 50)
        self.tol = kwargs.get('tol', 1e-6)
        self.L_T = self.L.T
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            'alpha': {
                'type': float,
                'default': 1e-4,
                'help': '正則化パラメータ'
            },
            'iterations': {
                'type': int,
                'default': 50,
                'help': '反復回数'
            },
            'tol': {
                'type': float,
                'default': 1e-6,
                'help': '収束判定閾値'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        Total Variation 正則化を適用
        
        解の微分に対するペナルティを課す正則化手法で、
        不連続点を保存しながらノイズを除去する特徴があります。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
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
        
        # ソルバー関数（ADMM: Alternating Direction Method of Multipliers）
        def solver_func(rhs):
            # 初期解を標準的な最小二乗解に設定
            x = jnp.linalg.lstsq(self.L, rhs, rcond=None)[0]
            z = D @ x
            u = jnp.zeros_like(z)  # 双対変数
            
            # ADMM反復
            for _ in range(self.iterations):
                # x-update (最小二乗問題)
                A = self.L_T @ self.L + self.alpha * D.T @ D
                b = self.L_T @ rhs + self.alpha * D.T @ (z - u)
                x_new = jnp.linalg.solve(A, b)
                
                # z-update (縮小演算子)
                Dx = D @ x_new
                z_new = self._shrinkage(Dx + u, 1.0/self.alpha)
                
                # u-update (双対変数の更新)
                u_new = u + Dx - z_new
                
                # 収束判定
                primal_residual = jnp.linalg.norm(Dx - z_new)
                dual_residual = self.alpha * jnp.linalg.norm(D.T @ (z_new - z))
                
                if primal_residual < self.tol and dual_residual < self.tol:
                    break
                
                # 変数の更新
                x, z, u = x_new, z_new, u_new
            
            return x
        
        return self.L, self.K, solver_func
    
    def _shrinkage(self, x, kappa):
        """縮小演算子（soft thresholding）"""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - kappa, 0)


# 正則化戦略をレジストリに登録
regularization_registry.register("total_variation", TotalVariationRegularization)
