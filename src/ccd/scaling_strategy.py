"""
スケーリング戦略モジュール

CuPy対応の行列のスケーリングに関する戦略を定義します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Tuple, Callable, Union, Optional

from strategy_interface import TransformationStrategy
from plugin_registry import PluginRegistry


class ScalingStrategy(TransformationStrategy):
    """
    スケーリング戦略の基底クラス

    行列のスケーリングを行うための共通インターフェース（CuPy疎行列対応）
    """

    def __init__(self, matrix: Union[cp.ndarray, cpx_sparse.spmatrix], **kwargs):
        """
        初期化

        Args:
            matrix: スケーリングする行列
            **kwargs: スケーリングパラメータ
        """
        # 疎行列かどうかを記録
        self.is_sparse = cpx_sparse.issparse(matrix)
        # 元の行列を保存（疎行列の場合はそのまま、密行列の場合はCuPy配列に変換）
        self.matrix = matrix if self.is_sparse else cp.asarray(matrix)
        self._init_params(**kwargs)

    def transform_matrix(
        self, matrix: Optional[Union[cp.ndarray, cpx_sparse.spmatrix]] = None
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        スケーリングを適用し、逆変換関数を返す

        Args:
            matrix: 変換する行列（指定がない場合は初期化時の行列を使用）

        Returns:
            (スケーリングされた行列, 逆スケーリング関数)
        """
        if matrix is not None:
            self.is_sparse = cpx_sparse.issparse(matrix)
            self.matrix = matrix
        return self.apply_scaling()

    def apply_scaling(self) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
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

    元の行列をそのまま返す（CuPy疎行列対応）
    """

    def apply_scaling(self) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
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