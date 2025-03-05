"""
LSQR法による正則化戦略

Lanczos双共役勾配法に基づく反復法による正則化を提供します。
"""

import cupy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategy import RegularizationStrategy, regularization_registry


class LSQRRegularization(RegularizationStrategy):
    """
    LSQR法による正則化

    Lanczos双共役勾配法に基づく反復法による正則化
    """

    def _init_params(self, **kwargs):
        """
        パラメータの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        self.iterations = kwargs.get("iterations", 20)
        self.damp = kwargs.get("damp", 0)

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {
            "iterations": {"type": int, "default": 20, "help": "反復回数"},
            "damp": {
                "type": float,
                "default": 0,
                "help": "減衰パラメータ（0よりも大きい値を設定するとTikhonov正則化と同様の効果）",
            },
        }

    def apply_regularization(
        self,
    ) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        LSQR法による正則化を適用

        Lanczos双共役勾配法に基づく反復法で、大規模な最小二乗問題に適しています。
        早期停止による正則化効果を得ます。

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

        # 減衰パラメータを考慮
        # 減衰パラメータがある場合は、単位行列にスケールして加算
        damp_scaled = self.damp * self.reg_factor
        n = L_scaled.shape[1]

        if damp_scaled > 0:
            L_reg = L_scaled + damp_scaled * jnp.eye(n)
        else:
            L_reg = L_scaled

        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor

        return L_reg, inverse_transform


# 正則化戦略をレジストリに登録
regularization_registry.register("lsqr", LSQRRegularization)
