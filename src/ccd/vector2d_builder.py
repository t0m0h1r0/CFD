"""
2次元CCD右辺ベクトルビルダーモジュール（修正版）

2次元CCD法の右辺ベクトルを生成するクラスを提供します。
"""

import cupy as cp
from typing import List, Optional, Dict, Tuple

from grid2d_config import Grid2DConfig


class CCD2DRightHandBuilder:
    """2次元CCD右辺ベクトルを生成するクラス（CuPy対応）"""

    def build_vector(
        self,
        grid_config: Grid2DConfig,
        values: cp.ndarray,
        coeffs: Optional[Dict[str, float]] = None,
        x_dirichlet_enabled: bool = None,
        y_dirichlet_enabled: bool = None,
        x_neumann_enabled: bool = None,
        y_neumann_enabled: bool = None,
    ) -> cp.ndarray:
        """
        2次元関数値から右辺ベクトルを生成（単純化版）
        """
        if coeffs is None:
            coeffs = grid_config.coeffs

        # 値の形状を確認
        nx, ny = grid_config.nx, grid_config.ny
        if values.shape != (nx, ny):
            raise ValueError(
                f"関数値の形状 {values.shape} が、グリッド設定の (nx, ny) = ({nx}, {ny}) と一致しません"
            )

        # 右辺ベクトルの正しいサイズは nx*ny*4
        total_size = nx * ny * 4
        rhs = cp.zeros(total_size, dtype=cp.float64)

        # 関数値を設定（4の倍数インデックス）
        idx = 0
        for j in range(ny):
            for i in range(nx):
                rhs[idx * 4] = values[i, j]
                idx += 1

        return rhs