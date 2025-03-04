"""
行列ビルダーモジュール

CCD法の左辺行列を生成するクラスを提供します。
"""

import jax.numpy as jnp
from typing import List, Optional, Tuple

from grid_config import GridConfig


class CCDLeftHandBuilder:
    """左辺ブロック行列を生成するクラス"""

    def _build_interior_blocks(
        self, coeffs: Optional[List[float]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """内部点のブロック行列A, B, Cを生成"""
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # 左ブロック行列 A
        A = jnp.array(
            [
                [0, 0, 0, 0],
                [35 / 32, 19 / 32, 1 / 8, 1 / 96],
                [-4, -29 / 16, -5 / 16, -1 / 48],
                [-105 / 16, -105 / 16, -15 / 8, -3 / 16],
            ]
        )

        # 中央ブロック行列 B - 第1行を係数で置き換え
        B = jnp.array([[a, b, c, d], [0, 1, 0, 0], [8, 0, 1, 0], [0, 0, 0, 1]])

        # 右ブロック行列 C
        C = jnp.array(
            [
                [0, 0, 0, 0],
                [-35 / 32, 19 / 32, -1 / 8, 1 / 96],
                [-4, 29 / 16, -5 / 16, 1 / 48],
                [105 / 16, -105 / 16, 15 / 8, -3 / 16],
            ]
        )

        return A, B, C

    def _build_boundary_blocks(
        self,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = True,
        neumann_enabled: bool = False,
    ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """境界点のブロック行列を生成"""
        # デフォルト係数
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # 基本のブロック行列
        B0 = jnp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [11 / 2, 1, 0, 0],  # ノイマン境界用
                [-51 / 2, 0, 1, 0],
                [387 / 4, 0, 0, 1],  # ディリクレ境界用
            ]
        )

        C0 = jnp.array(
            [
                [0, 0, 0, 0],
                [24, 24, 4, 4 / 3],
                [-264, -216, -44, -12],
                [1644, 1236, 282, 66],
            ]
        )

        D0 = jnp.array(
            [
                [0, 0, 0, 0],
                [-59 / 2, 10, -1, 0],
                [579 / 2, -99, 10, 0],
                [-6963 / 4, 1203 / 2, -123 / 2, 0],
            ]
        )

        ZR = jnp.array(
            [
                [0, 0, 0, 0],
                [59 / 2, 10, 1, 0],
                [579 / 2, 99, 10, 0],
                [6963 / 4, 1203 / 2, 123 / 2, 0],
            ]
        )

        AR = jnp.array(
            [
                [0, 0, 0, 0],
                [-24, 24, -4, 4 / 3],
                [-264, 216, -44, 12],
                [-1644, 1236, -282, 66],
            ]
        )

        BR = jnp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [-11 / 2, 1, 0, 0],  # ノイマン境界用
                [-51 / 2, 0, 1, 0],
                [-387 / 4, 0, 0, 1],  # ディリクレ境界用
            ]
        )

        # 境界条件に応じて行を更新
        # ディリクレ境界条件
        if dirichlet_enabled:
            # 左端の第4行
            B0 = B0.at[3].set([1, 0, 0, 0])
            C0 = C0.at[3].set([0, 0, 0, 0])
            D0 = D0.at[3].set([0, 0, 0, 0])

            # 右端の第4行
            BR = BR.at[3].set([1, 0, 0, 0])
            AR = AR.at[3].set([0, 0, 0, 0])
            ZR = ZR.at[3].set([0, 0, 0, 0])

        # ノイマン境界条件
        if neumann_enabled:
            # 左端の第2行
            B0 = B0.at[1].set([0, 1, 0, 0])
            C0 = C0.at[1].set([0, 0, 0, 0])
            D0 = D0.at[1].set([0, 0, 0, 0])

            # 右端の第2行
            BR = BR.at[1].set([0, 1, 0, 0])
            AR = AR.at[1].set([0, 0, 0, 0])
            ZR = ZR.at[1].set([0, 0, 0, 0])

        return B0, C0, D0, ZR, AR, BR

    def build_matrix(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> jnp.ndarray:
        """左辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h

        # coeffsが指定されていない場合はgrid_configから取得
        if coeffs is None:
            coeffs = grid_config.coeffs

        # 境界条件の状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet

        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann

        # ブロック行列を生成
        A, B, C = self._build_interior_blocks(coeffs)
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks(
            coeffs, dirichlet_enabled=dirichlet_enabled, neumann_enabled=neumann_enabled
        )

        # 次数行列
        DEGREE = jnp.array(
            [
                [1, 1, 1, 1],
                [h**-1, h**0, h**1, h**2],
                [h**-2, h**-1, h**0, h**1],
                [h**-3, h**-2, h**1, h**0],
            ]
        )

        # 次数を適用
        A *= DEGREE
        B *= DEGREE
        C *= DEGREE
        B0 *= DEGREE
        C0 *= DEGREE
        D0 *= DEGREE
        ZR *= DEGREE
        AR *= DEGREE
        BR *= DEGREE

        # 全体の行列を組み立て
        matrix_size = 4 * n
        L = jnp.zeros((matrix_size, matrix_size))

        # 左境界
        L = L.at[0:4, 0:4].set(B0)
        L = L.at[0:4, 4:8].set(C0)
        L = L.at[0:4, 8:12].set(D0)

        # 内部点
        for i in range(1, n - 1):
            row_start = 4 * i
            L = L.at[row_start : row_start + 4, row_start - 4 : row_start].set(A)
            L = L.at[row_start : row_start + 4, row_start : row_start + 4].set(B)
            L = L.at[row_start : row_start + 4, row_start + 4 : row_start + 8].set(C)

        # 右境界
        row_start = 4 * (n - 1)
        L = L.at[row_start : row_start + 4, row_start - 8 : row_start - 4].set(ZR)
        L = L.at[row_start : row_start + 4, row_start - 4 : row_start].set(AR)
        L = L.at[row_start : row_start + 4, row_start : row_start + 4].set(BR)

        return L
