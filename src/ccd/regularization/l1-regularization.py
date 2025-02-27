"""
L1正則化戦略（LASSO）

CCD法のL1正則化（LASSO）を提供します。
解のL1ノルムに対するペナルティを課します。
JAX互換の実装です。
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
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        L1正則化（LASSO）を適用
        
        解のL1ノルムに対するペナルティを課す正則化手法で、
        スパース性（多くの要素がゼロ）を持つ解を生成します。
        
        Returns:
            正則化された行列L、逆変換関数
        """
        # クラス変数をローカル変数に保存（ループ内での参照のため）
        L = self.L
        L_T = self.L_T
        alpha = self.alpha
        
        # 行列AとATAの事前計算
        ATA = L_T @ L
        
        # リプシッツ定数の推定（勾配降下法のステップサイズに関連）
        lambda_max = jnp.linalg.norm(ATA, ord=2)
        
        # 正則化された行列を計算
        # L1正則化では対角成分に正則化パラメータを加算
        n = L.shape[1]
        # 単位行列に正則化パラメータをスケールして加算
        L_reg = L + alpha * jnp.eye(n)
        
        # 逆変換関数
        def inverse_scaling(x_scaled):
            return x_scaled
        
        return L_reg, inverse_scaling


# 正則化戦略をレジストリに登録
regularization_registry.register("l1", L1Regularization)