"""
2次元ベクトルビルダーモジュール

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

        # 2次元配列を1次元に展開
        values_flat = values.reshape(-1)

        # 右辺ベクトルを初期化
        rhs = cp.zeros(nx * ny * depth)

        # 関数値を設定
        for i in range(nx):
            for j in range(ny):
                idx = (i * ny + j) * depth
                rhs[idx] = values[i, j]

        # ここで境界条件を設定
        # 実際の実装では、grid_config.boundary_valuesを使用して
        # 適切な境界条件を設定する必要があります

        return rhs