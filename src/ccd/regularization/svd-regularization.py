"""
SVD切断法による正則化戦略

CCD法のSVD切断法による正則化戦略を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class SVDRegularization(RegularizationStrategy):
    """SVD切断法による正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.threshold = kwargs.get('threshold', 1e-10)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            'threshold': {
                'type': float,
                'default': 1e-10,
                'help': '特異値の切断閾値（この値より小さい特異値は切断値に置き換える）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        SVD切断法による正則化を適用
        
        特異値分解を使用し、一定の閾値より小さい特異値を切断値に置き換えて
        数値的な安定性を向上させる正則化手法です。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        # 特異値分解を実行
        U, s, Vh = jnp.linalg.svd(self.L, full_matrices=False)
        
        # 特異値のフィルタリング
        s_filtered = jnp.where(s > self.threshold, s, self.threshold)
        
        # 擬似逆行列を計算
        pinv = Vh.T @ jnp.diag(1.0 / s_filtered) @ U.T
        
        # ソルバー関数
        def solver_func(rhs):
            return pinv @ rhs
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("svd", SVDRegularization)
