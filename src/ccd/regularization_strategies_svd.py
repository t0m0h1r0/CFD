"""
SVDベースの正則化戦略モジュール

CCD法のSVDベースの正則化戦略（SVD, TSVD）を提供します。
"""

import jax.numpy as jnp

from regularization_strategies_base import RegularizationStrategy


class SVDRegularization(RegularizationStrategy):
    """SVD切断法による正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, threshold: float = 1e-10):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            threshold: 特異値の切断閾値（この値より小さい特異値は切断値に置き換える）
        """
        super().__init__(L, K)
        self.threshold = threshold
    
    def apply_regularization(self):
        """
        SVD切断法による正則化を適用
        
        特異値分解を使用し、一定の閾値より小さい特異値を切断値に置き換えて
        数値的な安定性を向上させる正則化手法です。
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


class TSVDRegularization(RegularizationStrategy):
    """切断特異値分解（Truncated SVD）による正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, rank: int = None, threshold_ratio: float = 1e-5):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            rank: 保持する特異値の数（Noneの場合は閾値比率で決定）
            threshold_ratio: 最大特異値との比率による閾値（rank=Noneの場合のみ使用）
        """
        super().__init__(L, K)
        self.rank = rank
        self.threshold_ratio = threshold_ratio
    
    def apply_regularization(self):
        """
        切断特異値分解（TSVD）による正則化を適用
        
        指定したランク数以上の特異値を完全に切り捨てる手法です。
        SVD切断法とは異なり、小さな特異値は保持せず0に置き換えます。
        """
        # 特異値分解を実行
        U, s, Vh = jnp.linalg.svd(self.L, full_matrices=False)
        
        # 使用するランクを決定
        if self.rank is None:
            # 閾値比率に基づいてランクを決定
            threshold = jnp.max(s) * self.threshold_ratio
            self.rank = jnp.sum(s > threshold)
        else:
            # ランクが行列の最小次元を超えないようにする
            self.rank = min(self.rank, min(self.L.shape))
        
        # ランク外の特異値を0にする
        s_truncated = jnp.concatenate([s[:self.rank], jnp.zeros(len(s) - self.rank)])
        
        # 擬似逆行列を計算（0除算を避けるため、逆数を計算する前に特異値が0でないか確認）
        s_inv = jnp.where(s_truncated > 0, 1.0 / s_truncated, 0.0)
        pinv = Vh.T @ jnp.diag(s_inv) @ U.T
        
        # SVD成分を保存（診断用）
        self.U = U
        self.s = s
        self.Vh = Vh
        self.actual_rank = self.rank
        
        # ソルバー関数
        def solver_func(rhs):
            return pinv @ rhs
        
        return self.L, self.K, solver_func