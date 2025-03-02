"""
正規化スケーリング戦略

行と列のノルムに基づくスケーリングを提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class NormalizationScaling(ScalingStrategy):
    """
    行と列のL2ノルムによるスケーリング

    行と列の要素をそれぞれのL2ノルムで正規化します
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
        行と列のL2ノルムでスケーリングを適用

        各要素を行と列のL2ノルムの平方根で割ることでスケーリング

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 行と列のL2ノルムを計算
        row_norms = jnp.sqrt(jnp.sum(self.matrix * self.matrix, axis=1))
        col_norms = jnp.sqrt(jnp.sum(self.matrix * self.matrix, axis=0))

        # 0除算を防ぐため、非常に小さい値をクリップ
        row_norms = jnp.maximum(row_norms, 1e-10)
        col_norms = jnp.maximum(col_norms, 1e-10)

        # スケーリング行列を作成
        D_row = jnp.diag(1.0 / jnp.sqrt(row_norms))
        D_col = jnp.diag(1.0 / jnp.sqrt(col_norms))

        # スケーリングを適用
        L_scaled = D_row @ self.matrix @ D_col

        # スケーリング行列を保存（右辺ベクトル変換用）
        self.scaling_matrix_row = D_row
        self.scaling_matrix_col = D_col

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
scaling_registry.register("normalization", NormalizationScaling)
