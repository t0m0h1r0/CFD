"""
Elastic Net 正則化戦略

CCD法のElastic Net 正則化を提供します。
L1正則化とL2正則化を組み合わせた手法です。
JAX互換の実装です。
"""

import jax.numpy as jnp
import jax.lax as lax
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class ElasticNetRegularization(RegularizationStrategy):
    """Elastic Net 正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.alpha = kwargs.get('alpha', 1e-4)
        self.l1_ratio = kwargs.get('l1_ratio', 0.5)
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
                'help': '正則化パラメータの強さ'
            },
            'l1_ratio': {
                'type': float,
                'default': 0.5,
                'help': 'L1正則化の割合（0=L2のみ、1=L1のみ）'
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
        Elastic Net 正則化を適用
        
        L1正則化とL2正則化を組み合わせた手法で、スパース性を保ちながら
        相関の強い特徴間で選択の安定性を向上させます。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        # クラス変数をローカル変数に保存（ループ内での参照のため）
        L = self.L
        L_T = self.L_T
        alpha = self.alpha
        l1_ratio = self.l1_ratio
        tol = self.tol
        
        # L1とL2の重みを計算
        alpha_l1 = alpha * l1_ratio
        alpha_l2 = alpha * (1 - l1_ratio)
        
        # ソルバー関数（近位勾配法） - JAX互換バージョン
        def solver_func(rhs):
            # 初期解を標準的な最小二乗解に設定
            x_init = jnp.linalg.lstsq(L, rhs, rcond=None)[0]
            
            # L2正則化を考慮した行列の事前計算
            n = L.shape[1]
            ATA_l2 = L_T @ L + alpha_l2 * jnp.eye(n)
            ATb = L_T @ rhs
            
            # リプシッツ定数の推定
            lambda_max = jnp.linalg.norm(ATA_l2, ord=2)
            step_size = 1.0 / lambda_max
            
            # 近位勾配法のボディ関数
            def ista_body(i, loop_state):
                x, _best_x, _best_residual = loop_state
                
                # 勾配ステップ（L2正則化項を含む）
                grad = ATA_l2 @ x - ATb
                x_grad = x - step_size * grad
                
                # 近位演算子ステップ（L1正則化のみ）
                x_new = _soft_threshold(x_grad, alpha_l1 * step_size)
                
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
            
            # 最良解を返す
            return final_state[1]  # best_x
        
        # 軟閾値処理（soft thresholding）JAX互換バージョン
        def _soft_threshold(x, threshold):
            """軟閾値処理（soft thresholding）"""
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("elastic_net", ElasticNetRegularization)