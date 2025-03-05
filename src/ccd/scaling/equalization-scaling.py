"""
均等化スケーリング戦略

各行と列の最大絶対値を1にスケーリングする手法を提供します。
"""

import cupy as cp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class EqualizationScaling(ScalingStrategy):
    """
    均等化スケーリング

    各行と列の最大絶対値を1にスケーリングします
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
        均等化スケーリングを適用

        各行と列の最大絶対値を1にスケーリングします。
        行列の要素間のスケールを均一化することで数値的な安定性を向上させます。

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 1. 行の均等化
        row_max = cp.max(cp.abs(self.matrix), axis=1)
        # 0除算を防ぐため、非常に小さい値をクリップ
        row_max = cp.maximum(row_max, 1e-10)
        D_row = cp.diag(1.0 / row_max)
        L_row_eq = D_row @ self.matrix

        # 2. 列の均等化
        col_max = cp.max(cp.abs(L_row_eq), axis=0)
        # 0除算を防ぐため、非常に小さい値をクリップ
        col_max = cp.maximum(col_max, 1e-10)
        D_col = cp.diag(1.0 / col_max)

        # 3. スケーリングを適用
        L_scaled = L_row_eq @ D_col

        # スケーリング行列を保存
        self.scaling_matrix_row = D_row
        self.scaling_matrix_col = D_col

        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled

        return L_scaled, inverse_scaling

    def transform_rhs(self, rhs: cp.ndarray) -> cp.ndarray:
        """
        右辺ベクトルにスケーリングを適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        # 行方向のスケーリングのみ適用
        if hasattr(self, "scaling_matrix_row"):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("equalization", EqualizationScaling)
