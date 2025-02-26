"""
L1正則化戦略（LASSO）

CCD法のL1正則化（LASSO）を提供します。
解のL1ノルムに対するペナルティを課します。
JAX互換の実装です。
"""

import jax.numpy as jnp
import jax.lax as lax
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class L1Regularization(RegularizationStrategy):
    """L1正則化（LASSO）"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.alpha = kwargs.get('alpha', 1e-4)
        self.iterations = kwargs.get('iterations', 100)
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
                'default': 100,
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
        L1正則化（LASSO）を適用
        
        解のL1ノルムに対するペナルティを課す正則化手法で、
        スパース性（多くの要素がゼロ）を持つ解を生成します。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        # クラス変数をローカル変数に保存（ループ内での参照のため）
        L = self.L
        L_T = self.L_T
        alpha = self.alpha
        tol = self.tol
        
        # ソルバー関数（近位勾配法） - JAX互換バージョン
        def solver_func(rhs):
            # 初期解を標準的な最小二乗解に設定
            x_init = jnp.linalg.lstsq(L, rhs, rcond=None)[0]
            
            # 行列AとATAの事前計算
            ATA = L_T @ L
            ATb = L_T @ rhs
            
            # リプシッツ定数の推定（勾配降下法のステップサイズに関連）
            lambda_max = jnp.linalg.norm(ATA, ord=2)
            step_size = 1.0 / lambda_max
            
            # 近位勾配法のボディ関数
            def ista_body(i, loop_state):
                x, _best_x, _best_residual = loop_state
                
                # 勾配ステップ
                grad = ATA @ x - ATb
                x_grad = x - step_size * grad
                
                # 近位演算子ステップ（軟閾値処理）
                x_new = _soft_threshold(x_grad, alpha * step_size)
                
                # 残差計算
                residual = jnp.linalg.norm(x_new - x)
                
                # 最良解の更新
                cond = residual < _best_residual
                best_x = jnp.where(cond, x_new, _best_x)
                best_residual = jnp.where(cond, residual, _best_residual)
                
                # ループ状態を更新
                return x_new, best_x, best_residual
            
            # 初期ループ状態
            initial_state = (x_init, x_init, jnp.array(float('inf')))
            
            # ISTAアルゴリズムを実行
            final_state = lax.fori_loop(0, self.iterations, ista_body, initial_state)
            
            # トレランスより小さい残差になったベスト解を返す
            # 注：実際のコードでは最適な解を返すため、best_xを使用
            return final_state[1]  # best_x
        
        # 軟閾値処理（soft thresholding）JAX互換バージョン
        def _soft_threshold(x, threshold):
            """軟閾値処理（soft thresholding）"""
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("l1", L1Regularization)