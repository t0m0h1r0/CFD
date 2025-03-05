"""
正則化戦略モジュール

CuPy対応の行列の正則化に関する戦略を定義します。
"""

import cupy as cp
from typing import Tuple, Callable

from strategy_interface import TransformationStrategy
from plugin_registry import PluginRegistry


class RegularizationStrategy(TransformationStrategy):
    """
    正則化戦略の基底クラス

    行列の正則化を行うための共通インターフェース（CuPy対応）
    """

    def __init__(self, matrix: cp.ndarray, **kwargs):
        """
        初期化

        Args:
            matrix: 正則化する行列
            **kwargs: 正則化パラメータ
        """
        super().__init__(cp.asarray(matrix), **kwargs)
        # 正則化のためのスケーリング係数
        self.reg_factor = 1.0

    def transform_matrix(
        self, matrix=None
    ) -> Tuple[cp.ndarray, Callable[[cp.ndarray], cp.ndarray]]:
        """
        正則化を適用し、逆変換関数を返す

        Args:
            matrix: 変換する行列（指定がない場合は初期化時の行列を使用）

        Returns:
            (正則化された行列, 逆正則化関数)
        """
        if matrix is not None:
            self.matrix = cp.asarray(matrix)
        return self.apply_regularization()

    def apply_regularization(
        self,
    ) -> Tuple[cp.ndarray, Callable[[cp.ndarray], cp.ndarray]]:
        """
        正則化を適用する具体的な実装

        Returns:
            (正則化された行列, 逆正則化関数)
        """
        # デフォルトでは何もしない
        return self.matrix, lambda x: x

    def transform_rhs(self, rhs: cp.ndarray) -> cp.ndarray:
        """
        右辺ベクトルに正則化の変換を適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        # デフォルトでは元の正則化係数をかける
        return rhs * self.reg_factor


class NoneRegularization(RegularizationStrategy):
    """
    正則化なし

    元の行列をそのまま返す（CuPy対応）
    """

    def apply_regularization(
        self,
    ) -> Tuple[cp.ndarray, Callable[[cp.ndarray], cp.ndarray]]:
        """
        正則化を適用しない

        Returns:
            (元の行列, 恒等関数)
        """
        return self.matrix, lambda x: x


# 正則化戦略のレジストリを作成
regularization_registry = PluginRegistry(RegularizationStrategy, "正則化戦略")

# デフォルトの正則化戦略を登録
regularization_registry.register("none", NoneRegularization)