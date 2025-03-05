"""
対角優位スケーリング戦略

対角要素が1になるようスケーリングする手法を提供します。
"""

import cupy as cp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class DiagonalDominanceScaling(ScalingStrategy):
    """
    対角優位スケーリング

    対角要素が1になるようスケーリングします
    """

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {}

    def apply_scaling(self) -> Tuple[cp.ndarray, Callable[[cp.ndarray], cp.ndarray]]:
        """
        対角優位スケーリングを適用

        対角要素が1になるようスケーリングし、他の要素は相対的に小さくなります。
        数値解法の収束特性の改善に役立ちます。

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        n = self.matrix.shape[0]
        diag_elements = cp.diag(self.matrix)

        # 対角要素が0の場合に備えて小さな値を加える
        diag_elements = cp.where(cp.abs(diag_elements) < 1e-10, 1e-10, diag_elements)

        # スケーリング行列を作成
        D = cp.diag(1.0 / diag_elements)

        # 行と列のスケーリング行列として同じ行列を使用
        self.scaling_matrix_row = D
        self.scaling_matrix_col = D

        # スケーリングを適用
        L_scaled = D @ self.matrix @ D

        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D @ X_scaled

        return L_scaled, inverse_scaling

    def transform_rhs(self, rhs: cp.ndarray) -> cp.ndarray:
        """
        右辺ベクトルにスケーリングを適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        # 行方向のスケーリングを適用
        if hasattr(self, "scaling_matrix_row"):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("diagonal_dominance", DiagonalDominanceScaling)
