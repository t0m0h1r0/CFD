"""
2次元結果抽出モジュール

2次元CCD法の解ベクトルから各成分を抽出するクラスを提供します。
"""

import cupy as cp
from typing import Dict, Tuple, List

from grid2d_config import Grid2DConfig


class CCD2DResultExtractor:
    """2次元CCDソルバーの結果から各成分を抽出するクラス（CuPy対応）"""

    def extract_components(
        self, grid_config: Grid2DConfig, solution: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル

        Returns:
            各成分を含む辞書
            {
                "f": 関数値(nx, ny),
                "f_x": x方向1階微分(nx, ny),
                "f_y": y方向1階微分(nx, ny),
                "f_xx": x方向2階微分(nx, ny),
                ...
            }
        """
        nx, ny = grid_config.nx, grid_config.ny
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        
        # 1グリッドあたりの未知数
        vars_per_point = (x_order + 1) * (y_order + 1)  # フルシステム
        
        # 結果格納用の辞書
        results = {}
        
        # 関数値 (f)
        f = cp.zeros((nx, ny))
        f_x = cp.zeros((nx, ny))  # x方向1階微分
        f_y = cp.zeros((nx, ny))  # y方向1階微分
        f_xx = cp.zeros((nx, ny))  # x方向2階微分
        f_yy = cp.zeros((nx, ny))  # y方向2階微分
        f_xy = cp.zeros((nx, ny))  # 混合微分
        
        for i in range(nx):
            for j in range(ny):
                # 関数値
                idx_f = self._compute_vector_index(grid_config, i, j, 0)
                if idx_f < solution.size:
                    f[i, j] = solution[idx_f]
                
                # x方向1階微分 (f_x)
                if x_order >= 1:
                    idx_fx = self._compute_vector_index(grid_config, i, j, 1)
                    if idx_fx < solution.size:
                        f_x[i, j] = solution[idx_fx]
                
                # y方向1階微分 (f_y)
                if y_order >= 1:
                    idx_fy = self._compute_vector_index(grid_config, i, j, x_order + 1)
                    if idx_fy < solution.size:
                        f_y[i, j] = solution[idx_fy]
                
                # x方向2階微分 (f_xx)
                if x_order >= 2:
                    idx_fxx = self._compute_vector_index(grid_config, i, j, 2)
                    if idx_fxx < solution.size:
                        f_xx[i, j] = solution[idx_fxx]
                
                # y方向2階微分 (f_yy)
                if y_order >= 2:
                    idx_fyy = self._compute_vector_index(grid_config, i, j, x_order + 2)
                    if idx_fyy < solution.size:
                        f_yy[i, j] = solution[idx_fyy]
                
                # 混合微分 (f_xy) - 変数の配置に依存
                if x_order >= 1 and y_order >= 1:
                    # 変数の配置方法によって適切なインデックスを設定
                    idx_fxy = self._compute_vector_index(grid_config, i, j, (x_order + 1) + (y_order + 1))
                    if idx_fxy < solution.size:
                        try:
                            f_xy[i, j] = solution[idx_fxy]
                        except IndexError:
                            # インデックスが範囲外の場合は警告を出して続行
                            if i == 0 and j == 0:  # 最初の要素でのみ警告を出す
                                print(f"警告: 混合微分のインデックスが範囲外です: {idx_fxy} >= {solution.size}")
        
        # 結果辞書に格納
        results["f"] = f
        
        if x_order >= 1:
            results["f_x"] = f_x
        if y_order >= 1:
            results["f_y"] = f_y
        if x_order >= 2:
            results["f_xx"] = f_xx
        if y_order >= 2:
            results["f_yy"] = f_yy
        if x_order >= 1 and y_order >= 1:
            results["f_xy"] = f_xy
        
        # 3階導関数 (オプション)
        if x_order >= 3:
            f_xxx = cp.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    idx = self._compute_vector_index(grid_config, i, j, 3)
                    if idx < solution.size:
                        f_xxx[i, j] = solution[idx]
            results["f_xxx"] = f_xxx
        
        if y_order >= 3:
            f_yyy = cp.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    idx = self._compute_vector_index(grid_config, i, j, x_order + 3)
                    if idx < solution.size:
                        f_yyy[i, j] = solution[idx]
            results["f_yyy"] = f_yyy
        
        # ディリクレ境界条件が有効な場合、境界補正を適用
        if grid_config.is_dirichlet_x or grid_config.is_dirichlet_y:
            # 境界条件による補正を適用
            results["f"] = grid_config.apply_boundary_correction(results["f"])
        
        return results

    def _compute_vector_index(
        self, grid_config: Grid2DConfig, i: int, j: int, var_idx: int
    ) -> int:
        """
        2次元グリッド上の特定の変数のインデックスを計算

        Args:
            grid_config: 2次元グリッド設定
            i: x方向のインデックス
            j: y方向のインデックス
            var_idx: 変数のインデックス

        Returns:
            ベクトル内のインデックス
        """
        nx, ny = grid_config.nx, grid_config.ny
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        
        # フルシステムの場合
        vars_per_point = (x_order + 1) * (y_order + 1)
        
        # グリッド点のフラット化インデックス
        flat_idx = i * ny + j
        
        # 変数のオフセット
        return flat_idx * vars_per_point + var_idx

    def reshape_to_grid(
        self, grid_config: Grid2DConfig, vector: cp.ndarray, var_idx: int = 0
    ) -> cp.ndarray:
        """
        1次元ベクトルを2次元グリッド形状に再構成

        Args:
            grid_config: 2次元グリッド設定
            vector: 1次元ベクトル
            var_idx: 抽出する変数のインデックス
            
        Returns:
            2次元配列
        """
        nx, ny = grid_config.nx, grid_config.ny
        
        result = cp.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                idx = self._compute_vector_index(grid_config, i, j, var_idx)
                if idx < vector.size:
                    result[i, j] = vector[idx]
        
        return result