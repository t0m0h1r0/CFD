"""
逆中央ブロックスケーリング戦略

中央ブロック行列Bの逆行列を左辺と右辺に適用するスケーリング手法を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class InverseMidBlockScaling(ScalingStrategy):
    """
    逆中央ブロックスケーリング

    中央ブロック行列Bの逆行列を左辺の各ブロックA,B,C,B0,C0,D0,BR,AR,ZRおよび
    右辺のブロックに掛けることでスケーリングする手法です。
    これにより中央ブロックが単位行列に近くなり、計算が容易になります。
    """

    def _init_params(self, **kwargs):
        """
        パラメータの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        self.block_size = kwargs.get("block_size", 4)  # ブロックサイズ（デフォルトは4）
        self.epsilon = kwargs.get("epsilon", 1e-10)  # 逆行列計算時の安定化パラメータ

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {
            "block_size": {
                "type": int,
                "default": 4,
                "help": "ブロック行列のサイズ（通常は4）",
            },
            "epsilon": {
                "type": float,
                "default": 1e-10,
                "help": "逆行列計算時の安定化パラメータ",
            },
        }

    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        逆中央ブロックスケーリングを適用

        中央ブロック行列Bの逆行列を左辺の各ブロック(A,B,C,B0,C0,D0,BR,AR,ZR)および
        右辺のブロックに掛けてスケーリングします。

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 行列のサイズを取得
        n_total = self.matrix.shape[0]
        n_blocks = n_total // self.block_size

        # 中央ブロック行列Bの定義（ccd_core.pyから取得）
        # 典型的なBは [[1, 0, 0, 0], [0, 1, 0, 0], [8, 0, 1, 0], [0, 0, 0, 1]]
        # ただし、第1行は微分モードに依存して変化する可能性がある

        # 係数やモードに依存して変化し得るため、テンプレートは使用せず
        # 実際の行列から内部の中央ブロックを抽出して使用する

        # シンプルに最初の内部ブロック（通常は[4:8,4:8]）を使用
        # 行列のサイズが十分でない場合に対する安全チェック
        if n_total >= 2 * self.block_size:
            # 最初の内部ブロックを抽出
            B_actual = self.matrix[
                self.block_size : 2 * self.block_size,
                self.block_size : 2 * self.block_size,
            ]
        else:
            # 行列サイズが小さすぎる場合は単位行列を使用
            B_actual = jnp.eye(self.block_size)

        # B_actualの正則性をチェックして安全に逆行列を計算
        try:
            # 条件数をチェックして特異かどうかを判断
            cond_B = jnp.linalg.cond(B_actual)
            if cond_B < 1e6:  # 十分に良好な条件数
                B_inv = jnp.linalg.inv(B_actual)
            else:
                # 条件が悪い場合は正則化
                # 対角要素にわずかな値を加えて特異性を回避
                reg_B = B_actual + self.epsilon * jnp.eye(self.block_size)
                B_inv = jnp.linalg.inv(reg_B)
        except:
            # エラーが発生した場合は単位行列に近い対角行列を使用
            print(
                "警告: 中央ブロックの逆行列計算に失敗しました。対角行列で代用します。"
            )
            # 対角成分の最大値を取得して、その逆数でスケーリング
            diag_values = jnp.diag(B_actual)
            abs_diag = jnp.abs(diag_values)
            scale = 1.0 / jnp.maximum(jnp.max(abs_diag), self.epsilon)
            B_inv = jnp.diag(scale * jnp.sign(diag_values) * jnp.ones(self.block_size))

        # JITに対応したベクトル化実装
        # ConcretizationTypeError対策として動的なnp.arangeを避ける

        # 全行変換（直接行列乗算を使用）
        # まずB_invを左から掛ける行変換行列を作成
        block_transform = jnp.kron(jnp.eye(n_blocks), B_inv)

        # 行列全体に変換を適用
        L_scaled = block_transform @ self.matrix

        # スケーリング行列を保存
        self.scaling_matrix_row = jnp.kron(
            jnp.eye(n_blocks), B_inv
        )  # 行スケーリング行列
        self.scaling_matrix_col = jnp.eye(n_total)  # 列スケーリングなし

        # 逆変換関数
        def inverse_scaling(X_scaled):
            # 解ベクトルxに対しては変換不要（列方向のスケーリングなし）
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
        # 行方向のスケーリングのみ適用
        if hasattr(self, "scaling_matrix_row"):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("inverse_midblock", InverseMidBlockScaling)
