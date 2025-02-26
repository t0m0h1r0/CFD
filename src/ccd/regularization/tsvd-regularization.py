"""
切断特異値分解（TSVD）による正則化戦略

CCD法の切断特異値分解（TSVD）による正則化戦略を提供します。
JAX互換の実装です。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class TSVDRegularization(RegularizationStrategy):
    """切断特異値分解（Truncated SVD）による正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.rank = kwargs.get('rank', None)
        self.threshold_ratio = kwargs.get('threshold_ratio', 1e-5)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            'rank': {
                'type': int,
                'default': None,
                'help': '保持する特異値の数（Noneの場合は閾値比率で決定）'
            },
            'threshold_ratio': {
                'type': float,
                'default': 1e-5,
                'help': '最大特異値との比率による閾値（rank=Noneの場合のみ使用）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        切断特異値分解（TSVD）による正則化を適用
        
        指定したランク数以上の特異値を完全に切り捨てる手法です。
        SVD切断法とは異なり、小さな特異値は保持せず0に置き換えます。
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        # 特異値分解を実行
        U, s, Vh = jnp.linalg.svd(self.L, full_matrices=False)
        
        # 使用するランクを決定（JAX互換）
        if self.rank is None:
            # 閾値比率に基づいてランクを決定
            threshold = jnp.max(s) * self.threshold_ratio
            # JAX互換の方法でカウント
            mask = s > threshold
            rank = jnp.sum(mask)
        else:
            # ランクが行列の最小次元を超えないようにする
            rank = jnp.minimum(self.rank, jnp.minimum(self.L.shape[0], self.L.shape[1]))
        
        # JAX互換の方法で特異値をトランケート
        # 不要な特異値にはゼロを設定
        s_truncated = jnp.where(
            jnp.arange(s.shape[0]) < rank,
            s,
            jnp.zeros_like(s)
        )
        
        # 擬似逆行列を計算（0除算を避けるため、逆数を計算する前に特異値が0でないか確認）
        s_inv = jnp.where(s_truncated > 0, 1.0 / s_truncated, 0.0)
        pinv = Vh.T @ jnp.diag(s_inv) @ U.T
        
        # ソルバー関数
        def solver_func(rhs):
            return pinv @ rhs
        
        return self.L, self.K, solver_func


# 正則化戦略をレジストリに登録
regularization_registry.register("tsvd", TSVDRegularization)