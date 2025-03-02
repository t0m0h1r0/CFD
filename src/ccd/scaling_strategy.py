"""
スケーリング戦略モジュール

行列のスケーリングに関する戦略を定義します。
"""

import jax.numpy as jnp
from typing import Tuple, Callable

from strategy_interface import TransformationStrategy
from plugin_registry import PluginRegistry


class ScalingStrategy(TransformationStrategy):
    """
    スケーリング戦略の基底クラス

    行列のスケーリングを行うための共通インターフェース
    """

    def transform_matrix(
        self, matrix=None
    ) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        スケーリングを適用し、逆変換関数を返す

        Args:
            matrix: 変換する行列（指定がない場合は初期化時の行列を使用）

        Returns:
            (スケーリングされた行列, 逆スケーリング関数)
        """
        if matrix is not None:
            self.matrix = matrix
        return self.apply_scaling()

    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        スケーリングを適用する具体的な実装

        Returns:
            (スケーリングされた行列, 逆スケーリング関数)
        """
        # デフォルトでは何もしない
        return self.matrix, lambda x: x


class NoneScaling(ScalingStrategy):
    """
    スケーリングなし

    元の行列をそのまま返す
    """

    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        スケーリングを適用しない

        Returns:
            (元の行列, 恒等関数)
        """
        return self.matrix, lambda x: x


# スケーリング戦略のレジストリを作成
scaling_registry = PluginRegistry(ScalingStrategy, "スケーリング戦略")

# デフォルトのスケーリング戦略を登録
scaling_registry.register("none", NoneScaling)
