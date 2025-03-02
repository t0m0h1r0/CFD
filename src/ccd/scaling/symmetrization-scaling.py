"""
対称化スケーリング戦略

CCD法の対称化スケーリング戦略を提供します。
行列を相似変換によって対称化し、数値的な安定性を向上させます。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class SymmetrizationScaling(ScalingStrategy):
    """対称化スケーリング（Symmetrization Scaling）"""

    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.epsilon = kwargs.get("epsilon", 1e-10)  # 0除算防止の小さな値
        self.perfect_symmetry = kwargs.get(
            "perfect_symmetry", True
        )  # 完全な対称性を強制するか

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            "epsilon": {
                "type": float,
                "default": 1e-10,
                "help": "0除算防止のための小さな値",
            },
            "perfect_symmetry": {
                "type": bool,
                "default": True,
                "help": "最終的に(A+A^T)/2で完全対称化するかどうか",
            },
        }

    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        対称化スケーリングを適用

        相似変換 D^(1/2) · L · D^(-1/2) によって行列を対称化します。
        ここでDは行列の対角要素からなる対角行列です。

        Returns:
            対称化された行列L、逆変換関数
        """
        # 対角要素を抽出
        diag_elements = jnp.diag(self.L)

        # 対角要素の符号を保存
        signs = jnp.sign(diag_elements)
        signs = jnp.where(signs == 0, 1.0, signs)  # 0の場合は正の符号とする

        # 対角要素の絶対値を取得し、小さな値でクリップ
        abs_diag = jnp.abs(diag_elements)
        abs_diag = jnp.maximum(abs_diag, self.epsilon)

        # 対角行列の平方根とその逆数を計算
        D_sqrt = jnp.diag(jnp.sqrt(abs_diag) * signs)
        D_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(abs_diag) * signs)

        # 相似変換による対称化
        L_sym = D_sqrt @ self.L @ D_inv_sqrt

        # 数値誤差による微小な非対称性を取り除く（オプション）
        if self.perfect_symmetry:
            L_sym = (L_sym + L_sym.T) / 2.0

        # スケーリング行列を保存
        self.scaling_matrix_row = D_sqrt
        self.scaling_matrix_col = D_inv_sqrt

        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_inv_sqrt @ X_scaled

        return L_sym, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("symmetrization", SymmetrizationScaling)
