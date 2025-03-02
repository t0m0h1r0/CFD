"""
Elastic Net 正則化戦略

L1正則化とL2正則化を組み合わせた手法を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategy import RegularizationStrategy, regularization_registry


class ElasticNetRegularization(RegularizationStrategy):
    """
    Elastic Net 正則化

    L1正則化とL2正則化を組み合わせた手法
    """

    def _init_params(self, **kwargs):
        """
        パラメータの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        self.alpha = kwargs.get("alpha", 1e-4)
        self.l1_ratio = kwargs.get("l1_ratio", 0.5)

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {
            "alpha": {"type": float, "default": 1e-4, "help": "正則化パラメータの強さ"},
            "l1_ratio": {
                "type": float,
                "default": 0.5,
                "help": "L1正則化の割合（0=L2のみ、1=L1のみ）",
            },
        }

    def apply_regularization(
        self,
    ) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        Elastic Net 正則化を適用

        L1正則化とL2正則化を組み合わせた手法で、スパース性を保ちながら
        相関の強い特徴間で選択の安定性を向上させます。

        Returns:
            (正則化された行列, 逆変換関数)
        """
        # 行列のスケールを確認
        matrix_norm = jnp.linalg.norm(self.matrix, ord=2)

        # 行列のスケールが大きい場合はスケーリング
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.matrix * self.reg_factor
            alpha_scaled = self.alpha * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.matrix
            alpha_scaled = self.alpha

        # L1とL2の重みを計算
        alpha_l1 = alpha_scaled * self.l1_ratio
        alpha_l2 = alpha_scaled * (1 - self.l1_ratio)

        # L2正則化を適用した行列を計算
        n = L_scaled.shape[1]
        L_reg = L_scaled + alpha_l2 * jnp.eye(n)

        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor

        return L_reg, inverse_transform


# 正則化戦略をレジストリに登録
regularization_registry.register("elastic_net", ElasticNetRegularization)
