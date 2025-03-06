"""
2次元ベクトルビルダーモジュール - 修正版

2次元CCDの右辺ベクトルを生成するクラス
"""

import cupy as cp
from typing import Optional, List

from grid_config_2d import GridConfig2D
from vector_builder import CCDRightHandBuilder  # 1次元のベクトルビルダー


class CCDRightHandBuilder2D:
    """2次元CCDの右辺ベクトルを生成するクラス"""

    def __init__(self):
        """初期化"""
        # 1次元ビルダーを内部で使用
        self.builder_1d = CCDRightHandBuilder()

    def build_vector(
        self,
        grid_config: GridConfig2D,
        values: cp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> cp.ndarray:
        """
        2次元関数値から右辺ベクトルを生成

        Args:
            grid_config: 2次元グリッド設定
            values: 関数値の2次元配列 (nx, ny)
            coeffs: 係数（省略時はgrid_configから取得）
            dirichlet_enabled: ディリクレ境界条件の有効/無効（省略時はgrid_configから判断）
            neumann_enabled: ノイマン境界条件の有効/無効（省略時はgrid_configから判断）

        Returns:
            右辺ベクトル
        """
        # 係数の設定
        if coeffs is None:
            coeffs = grid_config.coeffs

        # 境界条件の状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet()
        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann()

        nx, ny = grid_config.nx_points, grid_config.ny_points
        depth = 4  # 各点での状態の次元

        # 入力値が正しい形状か確認
        if values.shape != (nx, ny):
            values = values.reshape(nx, ny)

        # 右辺ベクトルを初期化 - ここがサイズの問題
        rhs = cp.zeros(nx * ny * depth)

        # 関数値をフラットなベクトルに設定
        for i in range(nx):
            for j in range(ny):
                # グローバルインデックスを計算 - これが正しい式
                idx = ((i * ny + j) * depth)
                rhs[idx] = values[i, j]

        # 境界条件の設定
        # x方向の境界（左右）
        for j in range(ny):
            # 左境界（i=0）
            left_idx = j * depth
            if dirichlet_enabled:
                left_val = grid_config.boundary_values.get("left", [0.0] * ny)[j]
                rhs[left_idx + 3] = left_val  # ディリクレ条件（4番目の要素）

            if neumann_enabled:
                left_neumann = grid_config.boundary_values.get("left_neumann", [0.0] * ny)[j]
                rhs[left_idx + 1] = left_neumann  # ノイマン条件（2番目の要素）

            # 右境界（i=nx-1）
            right_idx = ((nx - 1) * ny + j) * depth
            if dirichlet_enabled:
                right_val = grid_config.boundary_values.get("right", [0.0] * ny)[j]
                rhs[right_idx + 3] = right_val  # ディリクレ条件

            if neumann_enabled:
                right_neumann = grid_config.boundary_values.get("right_neumann", [0.0] * ny)[j]
                rhs[right_idx + 1] = right_neumann  # ノイマン条件

        # y方向の境界（上下）
        for i in range(nx):
            # 下境界（j=0）
            bottom_idx = i * ny * depth
            if dirichlet_enabled:
                bottom_val = grid_config.boundary_values.get("bottom", [0.0] * nx)[i]
                rhs[bottom_idx + 3] = bottom_val  # ディリクレ条件

            if neumann_enabled:
                bottom_neumann = grid_config.boundary_values.get("bottom_neumann", [0.0] * nx)[i]
                rhs[bottom_idx + 2] = bottom_neumann  # y方向ノイマン条件（3番目の要素）

            # 上境界（j=ny-1）
            top_idx = (i * ny + (ny - 1)) * depth
            if dirichlet_enabled:
                top_val = grid_config.boundary_values.get("top", [0.0] * nx)[i]
                rhs[top_idx + 3] = top_val  # ディリクレ条件

            if neumann_enabled:
                top_neumann = grid_config.boundary_values.get("top_neumann", [0.0] * nx)[i]
                rhs[top_idx + 2] = top_neumann  # y方向ノイマン条件

        return rhs