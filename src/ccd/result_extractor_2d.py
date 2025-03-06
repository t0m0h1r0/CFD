"""
2次元結果抽出モジュール - 根本的バグ修正版

2次元CCDの解ベクトルから各種偏導関数値を抽出するクラス
"""

import cupy as cp
from typing import Tuple

from grid_config_2d import GridConfig2D


class CCDResultExtractor2D:
    """2次元CCDの結果を抽出するクラス"""

    def extract_components(
        self, grid_config: GridConfig2D, solution: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        2次元解ベクトルから関数値と各偏導関数を抽出

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル (nx*ny*4)

        Returns:
            (f, f_x, f_y, f_xx, f_xy, f_yy)の形式のタプル
            各要素は2次元配列 (nx, ny)
        """
        nx, ny = grid_config.nx_points, grid_config.ny_points
        depth = 4  # 関数値と偏導関数の数
        
        # 結果を格納する配列を初期化
        f = cp.zeros((nx, ny))
        f_x = cp.zeros((nx, ny))
        f_y = cp.zeros((nx, ny))
        f_xx = cp.zeros((nx, ny))
        f_xy = cp.zeros((nx, ny))
        f_yy = cp.zeros((nx, ny))
        
        # チェック: solution が正しいサイズか確認
        expected_size = nx * ny * depth
        actual_size = solution.shape[0]
        
        if actual_size != expected_size:
            print(f"警告: ソリューションベクトルのサイズ({actual_size})が期待値({expected_size})と異なります")
        
        # ソリューションから各成分を抽出
        # CCDの各点情報は [f, f_x, f_y, f_xx] の順に格納されている
        for i in range(nx):
            for j in range(ny):
                idx = (i * ny + j) * depth
                
                # インデックスが有効範囲内かチェック
                if idx < actual_size:
                    f[i, j] = solution[idx]        # 関数値
                    
                    if idx + 1 < actual_size:
                        f_x[i, j] = solution[idx + 1]  # x方向偏導関数
                    
                    if idx + 2 < actual_size:
                        f_y[i, j] = solution[idx + 2]  # y方向偏導関数
                    
                    if idx + 3 < actual_size:
                        f_xx[i, j] = solution[idx + 3] # x方向2階偏導関数
        
        # 混合偏導関数 f_xy と y方向2階偏導関数 f_yy は直接格納されていないため計算
        
        # 混合偏導関数 f_xy (中心差分)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                f_xy[i, j] = (f_y[i+1, j] - f_y[i-1, j]) / (2 * grid_config.hx)
        
        # y方向2階偏導関数 f_yy (中心差分)
        for i in range(nx):
            for j in range(1, ny-1):
                f_yy[i, j] = (f[i, j+1] - 2*f[i, j] + f[i, j-1]) / (grid_config.hy**2)
        
        # 境界値の調整（必要に応じて）
        if grid_config.is_dirichlet() and grid_config.enable_boundary_correction:
            # ディリクレ境界条件がある場合、指定された境界値で更新
            if grid_config.boundary_values:
                # 左境界値
                if "left" in grid_config.boundary_values:
                    for j in range(ny):
                        if j < len(grid_config.boundary_values["left"]):
                            f[0, j] = grid_config.boundary_values["left"][j]
                
                # 右境界値
                if "right" in grid_config.boundary_values:
                    for j in range(ny):
                        if j < len(grid_config.boundary_values["right"]):
                            f[nx-1, j] = grid_config.boundary_values["right"][j]
                
                # 下境界値
                if "bottom" in grid_config.boundary_values:
                    for i in range(nx):
                        if i < len(grid_config.boundary_values["bottom"]):
                            f[i, 0] = grid_config.boundary_values["bottom"][i]
                
                # 上境界値
                if "top" in grid_config.boundary_values:
                    for i in range(nx):
                        if i < len(grid_config.boundary_values["top"]):
                            f[i, ny-1] = grid_config.boundary_values["top"][i]
        
        return f, f_x, f_y, f_xx, f_xy, f_yy