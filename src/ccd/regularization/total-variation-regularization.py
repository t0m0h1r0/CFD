"""
Total Variation 正則化戦略

CCD法のTotal Variation 正則化を提供します。
解の微分に対するペナルティを課し、不連続点を保存しながらノイズを除去します。
JAX互換の実装です。
"""

import jax.numpy as jnp
import jax.lax as lax
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
        # クラス変数をローカル変数に保存（ループ内での参照のため）
        L = self.L
        L_T = self.L_T
        alpha = self.alpha
        tol = self.tol
        
        # 差分行列（1階微分演算子）の構築
        n = L.shape[1] // 3  # 各グリッド点の自由度は3
        
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
        
        # JAX互換のソルバー関数（ADMM: Alternating Direction Method of Multipliers）
        def solver_func(rhs):
            # 初期解を標準的な最小二乗解に設定
            x_init = jnp.linalg.lstsq(L, rhs, rcond=None)[0]
            z_init = D @ x_init
            u_init = jnp.zeros_like(z_init)  # 双対変数
            
            # ADMM反復のボディ関数
            def admm_body(i, loop_state):
                x, z, u, _best_x, _best_residual = loop_state
                
                # x-update (最小二乗問題)
                A = L_T @ L + alpha * D.T @ D
                b = L_T @ rhs + alpha * D.T @ (z - u)
                x_new = jnp.linalg.solve(A, b)
                
                # z-update (縮小演算子)
                Dx = D @ x_new
                z_new = _shrinkage(Dx + u, 1.0/alpha)
                
                # u-update (双対変数の更新)
                u_new = u + Dx - z_new
                
                # 残差計算
                primal_residual = jnp.linalg.norm(Dx - z_new)
                dual_residual = alpha * jnp.linalg.norm(D.T @ (z_new - z))
                combined_residual = primal_residual + dual_residual
                
                # 最良解の更新
                cond = combined_residual < _best_residual
                best_x = jnp.where(cond, x_new, _best_x)
                best_residual = jnp.where(cond, combined_residual, _best_residual)
                
                # ループ状態を更新
                return x_new, z_new, u_new, best_x, best_residual
            
            # 初期ループ状態
            initial_state = (x_init, z_init, u_init, x_init, jnp.array(float('inf')))
            
            # ADMMアルゴリズムを実行
            final_state = lax.fori_loop(0, self.iterations, admm_body, initial_state)
            
            # 最良解を返す
            return final_state[3]  # best_x
        
        # 縮小演算子（JAX互換）
        def _shrinkage(x, kappa):
            """縮小演算子（soft thresholding）"""
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - kappa, 0)
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("total_variation", TotalVariationRegularization)