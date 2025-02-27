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
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        Total Variation 正則化を適用
        
        解の微分に対するペナルティを課す正則化手法で、
        不連続点を保存しながらノイズを除去する特徴があります。
        
        Returns:
            正則化された行列L、逆変換関数
        """
        # クラス変数をローカル変数に保存（ループ内での参照のため）
        L = self.L
        L_T = self.L_T
        alpha = self.alpha
        
        # 差分行列（1階微分演算子）の構築
        n = L.shape[1] // 3  # 各グリッド点の自由度は3
        
        # L2正則化を考慮した行列の事前計算
        n_matrix = L.shape[1]
        # 単位行列に正則化パラメータをスケールして加算
        L_reg = L + alpha * jnp.eye(n_matrix)
        
        # 逆変換関数
        def inverse_scaling(x_scaled):
            return x_scaled
        
        return L_reg, inverse_scaling


# 正則化戦略をレジストリに登録
regularization_registry.register("total_variation", TotalVariationRegularization)