"""
Rehu スケーリング戦略

行と列の最大絶対値によるスケーリングを提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class RehuScaling(ScalingStrategy):
    """
    Rehu法によるスケーリング

    各行と列の最大絶対値の平方根でスケーリングする効果的な方法
    """

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {}

    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        Rehu法（行と列の最大絶対値）によるスケーリングを適用

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 行列の各行と列の最大絶対値を計算
        max_values_row = jnp.max(jnp.abs(self.matrix), axis=1)
        max_values_col = jnp.max(jnp.abs(self.matrix), axis=0)

        # 0除算を防ぐため、非常に小さい値をクリップ
        max_values_row = jnp.maximum(max_values_row, 1e-10)
        max_values_col = jnp.maximum(max_values_col, 1e-10)

        # スケーリング行列を作成
        D_row = jnp.diag(1.0 / jnp.sqrt(max_values_row))
        D_col = jnp.diag(1.0 / jnp.sqrt(max_values_col))

        # スケーリング行列を保存（右辺ベクトル変換用）
        self.scaling_matrix_row = D_row
        self.scaling_matrix_col = D_col

        # スケーリングを適用
        L_scaled = D_row @ self.matrix @ D_col

        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled

        return L_scaled, inverse_scaling

    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルにスケーリングを適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        # 行方向のスケーリングのみ適用（列方向は解に影響）
        if hasattr(self, "scaling_matrix_row"):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("rehu", RehuScaling)
