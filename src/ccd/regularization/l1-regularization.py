"""
L1正則化戦略（LASSO）

CCD法のL1正則化（LASSO）を提供します。
解のL1ノルムに対するペナルティを課します。
"""

import jax.numpy as jnp
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
            
            # 近位勾配法（ISTA: Iterative Shrinkage-Thresholding Algorithm）
            for _ in range(self.iterations):
                # 勾配ステップ
                grad = ATA @ x - ATb
                x_grad = x - step_size * grad
                
                # 近位演算子ステップ（軟閾値処理）
                x_new = self._soft_threshold(x_grad, self.alpha * step_size)
                
                # 収束判定
                if jnp.linalg.norm(x_new - x) < self.tol:
                    break
                
                x = x_new
            
            return x
        
        return self.L, self.K, solver_func
    
    def _soft_threshold(self, x, threshold):
        """軟閾値処理（soft thresholding）"""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)


# 正則化戦略をレジストリに登録
regularization_registry.register("l1", L1Regularization)
