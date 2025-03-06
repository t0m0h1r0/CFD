"""
2次元ベクトルビルダーモジュール - 根本的バグ修正版

2次元CCDの右辺ベクトルを生成するクラス
"""

import cupy as cp
from typing import Optional, List

from grid_config_2d import GridConfig2D


class CCDRightHandBuilder2D:
    """2次元CCDの右辺ベクトルを生成するクラス"""

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
            右辺ベクトル (nx*ny*4)
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
            print(f"警告: 入力値の形状 {values.shape} は期待値 ({nx}, {ny}) と異なります")
            values = values.reshape(nx, ny)

        # 正しいサイズの右辺ベクトルを初期化
        total_size = nx * ny * depth  # 正しいベクトルサイズ
        rhs = cp.zeros(total_size)

        print(f"右辺ベクトルを構築: 総要素数 = {total_size} (nx = {nx}, ny = {ny}, depth = {depth})")
        print(f"入力値の形状: {values.shape}")

        # 関数値を右辺ベクトルに設定
        for i in range(nx):
            for j in range(ny):
                # グローバルインデックスを計算（重要：状態ベクトル空間でのインデックス）
                idx = (i * ny + j) * depth
                
                # 各点の各状態に対して設定（主に関数値のみを設定）
                rhs[idx] = values[i, j]
                # 残りの3つの状態（導関数）は0のまま
                # rhs[idx+1] = 0.0  # x方向導関数
                # rhs[idx+2] = 0.0  # y方向導関数
                # rhs[idx+3] = 0.0  # 2階導関数

        # ディリクレ境界条件の設定
        if dirichlet_enabled and grid_config.boundary_values:
            # 境界値を取得
            # 左右境界のディリクレ条件
            left_values = grid_config.boundary_values.get("left", [0.0] * ny)
            right_values = grid_config.boundary_values.get("right", [0.0] * ny)
            
            # 上下境界のディリクレ条件
            bottom_values = grid_config.boundary_values.get("bottom", [0.0] * nx)
            top_values = grid_config.boundary_values.get("top", [0.0] * nx)
            
            # 左境界 (i=0)
            for j in range(ny):
                if j < len(left_values):
                    idx = (0 * ny + j) * depth
                    rhs[idx + 3] = left_values[j]  # 状態の4番目成分にディリクレ値を設定
            
            # 右境界 (i=nx-1)
            for j in range(ny):
                if j < len(right_values):
                    idx = ((nx-1) * ny + j) * depth
                    rhs[idx + 3] = right_values[j]
            
            # 下境界 (j=0)
            for i in range(nx):
                if i < len(bottom_values):
                    idx = (i * ny + 0) * depth
                    rhs[idx + 3] = bottom_values[i]
            
            # 上境界 (j=ny-1)
            for i in range(nx):
                if i < len(top_values):
                    idx = (i * ny + (ny-1)) * depth
                    rhs[idx + 3] = top_values[i]

        # ノイマン境界条件の設定
        if neumann_enabled and grid_config.boundary_values:
            # 左右境界のノイマン条件
            left_neumann = grid_config.boundary_values.get("left_neumann", [0.0] * ny)
            right_neumann = grid_config.boundary_values.get("right_neumann", [0.0] * ny)
            
            # 上下境界のノイマン条件
            bottom_neumann = grid_config.boundary_values.get("bottom_neumann", [0.0] * nx)
            top_neumann = grid_config.boundary_values.get("top_neumann", [0.0] * nx)
            
            # 左境界 (i=0)
            for j in range(ny):
                if j < len(left_neumann):
                    idx = (0 * ny + j) * depth
                    rhs[idx + 1] = left_neumann[j]  # 状態の2番目成分にx方向ノイマン値を設定
            
            # 右境界 (i=nx-1)
            for j in range(ny):
                if j < len(right_neumann):
                    idx = ((nx-1) * ny + j) * depth
                    rhs[idx + 1] = right_neumann[j]
            
            # 下境界 (j=0)
            for i in range(nx):
                if i < len(bottom_neumann):
                    idx = (i * ny + 0) * depth
                    rhs[idx + 2] = bottom_neumann[i]  # 状態の3番目成分にy方向ノイマン値を設定
            
            # 上境界 (j=ny-1)
            for i in range(nx):
                if i < len(top_neumann):
                    idx = (i * ny + (ny-1)) * depth
                    rhs[idx + 2] = top_neumann[i]

        return rhs