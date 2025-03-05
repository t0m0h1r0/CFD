"""
切断特異値分解（TSVD）正則化戦略

特定のランク以上の特異値を完全に切り捨てる正則化戦略を提供します。
"""

import cupy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategy import RegularizationStrategy, regularization_registry


class TSVDRegularization(RegularizationStrategy):
    """
    切断特異値分解（Truncated SVD）による正則化

    指定したランク以上の特異値を完全に切り捨てる
    """

    def _init_params(self, **kwargs):
        """
        パラメータの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        self.rank = kwargs.get("rank", None)
        self.threshold_ratio = kwargs.get("threshold_ratio", 1e-5)

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {
            "rank": {
                "type": int,
                "default": None,
                "help": "保持する特異値の数（Noneの場合は閾値比率で決定）",
            },
            "threshold_ratio": {
                "type": float,
                "default": 1e-5,
                "help": "最大特異値との比率による閾値（rank=Noneの場合のみ使用）",
            },
        }

    def apply_regularization(
        self,
    ) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        切断特異値分解（TSVD）による正則化を適用

        Returns:
            (正則化された行列, 逆変換関数)
        """
        # 行列のスケールを確認
        matrix_norm = jnp.linalg.norm(self.matrix, ord=2)

        # 行列のスケールが大きい場合はスケーリング
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.matrix * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.matrix

        # 特異値分解を実行
        U, s, Vh = jnp.linalg.svd(L_scaled, full_matrices=False)

        # 使用するランクを決定（JAX互換）
        if self.rank is None:
            # 閾値比率に基づいてランクを決定
            threshold = jnp.max(s) * self.threshold_ratio
            # JAX互換の方法でカウント
            mask = s > threshold
            rank = jnp.sum(mask)
        else:
            # ランクが行列の最小次元を超えないようにする
            rank = jnp.minimum(
                self.rank, jnp.minimum(L_scaled.shape[0], L_scaled.shape[1])
            )

        # JAX互換の方法で特異値をトランケート
        # 不要な特異値にはゼロを設定
        s_truncated = jnp.where(jnp.arange(s.shape[0]) < rank, s, jnp.zeros_like(s))

        # 正則化された行列を計算
        L_reg = U @ jnp.diag(s_truncated) @ Vh

        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor

        return L_reg, inverse_transform


# 正則化戦略をレジストリに登録
regularization_registry.register("tsvd", TSVDRegularization)
