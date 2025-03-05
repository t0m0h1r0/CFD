"""
最大成分スケーリング戦略

行列全体の最大絶対値が1になるようスケーリングする手法を提供します。
"""

import cupy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class MaxElementScaling(ScalingStrategy):
    """
    最大成分スケーリング

    行列全体の最大絶対値が1になるようスケーリングします
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
        最大成分スケーリングを適用

        行列全体の最大絶対値が1になるようスケーリングします。
        非常にシンプルなスケーリング手法です。

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 行列全体の最大絶対値を取得
        max_abs_value = jnp.max(jnp.abs(self.matrix))

        # 0除算を防ぐため、非常に小さい値をクリップ
        max_abs_value = jnp.maximum(max_abs_value, 1e-10)

        # スケーリング係数を保存
        self.scale_factor = max_abs_value

        # スケーリングを適用
        L_scaled = self.matrix / max_abs_value

        # 逆変換関数 - この場合はスケーリングが一様なので、追加の変換は不要
        def inverse_scaling(X_scaled):
            return X_scaled

        return L_scaled, inverse_scaling

    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルにスケーリングを適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        # 行列と同じスケールで右辺ベクトルもスケーリング
        if hasattr(self, "scale_factor"):
            return rhs / self.scale_factor
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("max_element", MaxElementScaling)
