"""
行列ビルダーモジュール

CCD法の左辺行列を生成するクラスを提供します。
"""

import cupy as cp
from typing import List, Optional, Tuple

from grid_config import GridConfig


class CCDLeftHandBuilder:
    """左辺ブロック行列を生成するクラス"""

    def _build_interior_blocks(
        self, coeffs: Optional[List[float]] = None
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """内部点のブロック行列A, B, Cを生成"""
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # CuPy配列として定義
        A = cp.array(
            [
                [0, 0, 0, 0],
                [35 / 32, 19 / 32, 1 / 8, 1 / 96],
                [-4, -29 / 16, -5 / 16, -1 / 48],
                [-105 / 16, -105 / 16, -15 / 8, -3 / 16],
            ]
        )

        # 中央ブロック行列 B - 第1行を係数で置き換え
        B = cp.array([[a, b, c, d], [0, 1, 0, 0], [8, 0, 1, 0], [0, 0, 0, 1]])

        # 右ブロック行列 C
        C = cp.array(
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
        cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray
    ]:
        """境界点のブロック行列を生成"""
        # デフォルト係数
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # 基本のブロック行列
        B0 = cp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [11 / 2, 1, 0, 0],  # ノイマン境界用
                [-51 / 2, 0, 1, 0],
                [387 / 4, 0, 0, 1],  # ディリクレ境界用
            ]
        )

        C0 = cp.array(
            [
                [0, 0, 0, 0],
                [24, 24, 4, 4 / 3],
                [-264, -216, -44, -12],
                [1644, 1236, 282, 66],
            ]
        )

        D0 = cp.array(
            [
                [0, 0, 0, 0],
                [-59 / 2, 10, -1, 0],
                [579 / 2, -99, 10, 0],
                [-6963 / 4, 1203 / 2, -123 / 2, 0],
            ]
        )

        ZR = cp.array(
            [
                [0, 0, 0, 0],
                [59 / 2, 10, 1, 0],
                [579 / 2, 99, 10, 0],
                [6963 / 4, 1203 / 2, 123 / 2, 0],
            ]
        )

        AR = cp.array(
            [
                [0, 0, 0, 0],
                [-24, 24, -4, 4 / 3],
                [-264, 216, -44, 12],
                [-1644, 1236, -282, 66],
            ]
        )

        BR = cp.array(
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
            B0_new = B0.copy()
            B0_new[3] = cp.array([1, 0, 0, 0])
            B0 = B0_new

            C0_new = C0.copy()
            C0_new[3] = cp.array([0, 0, 0, 0])
            C0 = C0_new

            D0_new = D0.copy()
            D0_new[3] = cp.array([0, 0, 0, 0])
            D0 = D0_new

            # 右端の第4行
            BR_new = BR.copy()
            BR_new[3] = cp.array([1, 0, 0, 0])
            BR = BR_new

            AR_new = AR.copy()
            AR_new[3] = cp.array([0, 0, 0, 0])
            AR = AR_new

            ZR_new = ZR.copy()
            ZR_new[3] = cp.array([0, 0, 0, 0])
            ZR = ZR_new

        # ノイマン境界条件
        if neumann_enabled:
            # 左端の第2行
            B0_new = B0.copy()
            B0_new[1] = cp.array([0, 1, 0, 0])
            B0 = B0_new

            C0_new = C0.copy()
            C0_new[1] = cp.array([0, 0, 0, 0])
            C0 = C0_new

            D0_new = D0.copy()
            D0_new[1] = cp.array([0, 0, 0, 0])
            D0 = D0_new

            # 右端の第2行
            BR_new = BR.copy()
            BR_new[1] = cp.array([0, 1, 0, 0])
            BR = BR_new

            AR_new = AR.copy()
            AR_new[1] = cp.array([0, 0, 0, 0])
            AR = AR_new

            ZR_new = ZR.copy()
            ZR_new[1] = cp.array([0, 0, 0, 0])
            ZR = ZR_new

        return B0, C0, D0, ZR, AR, BR

    def build_matrix(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> cp.ndarray:
        """左辺のブロック行列全体を生成（CuPy対応）"""
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
        DEGREE = cp.array(
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
        depth = 4
        matrix_size = depth * n
        L = cp.zeros((matrix_size, matrix_size))

        # 左境界
        L[:depth, :depth] = B0
        L[:depth, depth:2*depth] = C0
        L[:depth, 2*depth:3*depth] = D0

        # 内部点
        for i in range(1, n - 1):
            row_start = depth * i
            L[row_start:row_start + depth, row_start - depth:row_start] = A
            L[row_start:row_start + depth, row_start:row_start + depth] = B
            L[row_start:row_start + depth, row_start + depth:row_start + 2 * depth] = C

        # 右境界
        row_start = depth * (n - 1)
        L[row_start:row_start + depth, row_start - 2 * depth:row_start - depth] = ZR
        L[row_start:row_start + depth, row_start - depth:row_start] = AR
        L[row_start:row_start + depth, row_start:row_start + depth] = BR

        return L