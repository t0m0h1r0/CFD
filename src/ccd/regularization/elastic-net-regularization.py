"""
Elastic Net 正則化戦略

CCD法のElastic Net 正則化を提供します。
L1正則化とL2正則化を組み合わせた手法です。
"""

import jax.numpy as jnp
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
            
            # 近位勾配法（ISTA）
            for _ in range(self.iterations):
                # 勾配ステップ（L2正則化項を含む）
                grad = ATA_l2 @ x - ATb
                x_grad = x - step_size * grad
                
                # 近位演算子ステップ（L1正則化のみ）
                x_new = self._soft_threshold(x_grad, alpha_l1 * step_size)
                
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
regularization_registry.register("elastic_net", ElasticNetRegularization)
