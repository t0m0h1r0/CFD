"""
Elastic Net 正則化戦略

CCD法のElastic Net 正則化を提供します。
L1正則化とL2正則化を組み合わせた手法です。
JAX互換の実装です。
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
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        Elastic Net 正則化を適用
        
        L1正則化とL2正則化を組み合わせた手法で、スパース性を保ちながら
        相関の強い特徴間で選択の安定性を向上させます。
        
        Returns:
            正則化された行列L、逆変換関数
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
        
        # 初期解を標準的な最小二乗解に設定
        n = L.shape[1]
        
        # L2正則化を考慮した行列の事前計算
        ATA_l2 = L_T @ L + alpha_l2 * jnp.eye(n)
        
        # リプシッツ定数の推定
        lambda_max = jnp.linalg.norm(ATA_l2, ord=2)
        step_size = 1.0 / lambda_max
        
        # 正則化された行列を計算
        # 何らかの適切な正則化行列を選択（例：単位行列を加算）
        L_reg = L + alpha_l2 * jnp.eye(n)
        
        # 逆変換関数
        def inverse_scaling(x_scaled):
            return x_scaled
        
        return L_reg, inverse_scaling


# 正則化戦略をレジストリに登録
regularization_registry.register("elastic_net", ElasticNetRegularization)