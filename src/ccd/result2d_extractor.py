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
        
        # 1グリッドあたりの未知数数
        vars_per_point = x_order + y_order + 1  # 関数値は1つだけカウント
        
        # 結果格納用の辞書
        results = {}
        
        # 関数値
        f = cp.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                idx = self._compute_vector_index(grid_config, i, j, 0)
                f[i, j] = solution[idx]
        results["f"] = f
        
        # x方向の導関数
        for d in range(1, x_order + 1):
            f_x = cp.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    idx = self._compute_vector_index(grid_config, i, j, d)
                    f_x[i, j] = solution[idx]
            results[f"f_{'x' * d}"] = f_x
        
        # y方向の導関数
        for d in range(1, y_order + 1):
            f_y = cp.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    idx = self._compute_vector_index(grid_config, i, j, x_order + d)
                    f_y[i, j] = solution[idx]
            results[f"f_{'y' * d}"] = f_y
        
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
            var_idx: 変数のインデックス（0: 関数値, 1: x方向1階微分, ...）

        Returns:
            ベクトル内のインデックス
        """
        nx, ny = grid_config.nx, grid_config.ny
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        
        # 1グリッドあたりの未知数数
        vars_per_point = x_order + y_order + 1  # 関数値は1つだけカウント
        
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
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        vars_per_point = x_order + y_order + 1
        
        result = cp.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                idx = self._compute_vector_index(grid_config, i, j, var_idx)
                if idx < vector.size:
                    result[i, j] = vector[idx]
        
        return result

    def extract_mixed_derivatives(
        self, grid_config: Grid2DConfig, solution: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """
        解ベクトルから混合微分を抽出
        
        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル
            
        Returns:
            混合微分を含む辞書
            {
                "f_xy": 混合1階微分(nx, ny),
                "f_xxy": 混合2階x,1階y微分(nx, ny),
                ...
            }
        """
        # 混合微分の実装は、全ての混合微分変数が解ベクトルに含まれていることを前提としています
        # より一般的な実装では、利用可能な変数に基づいて動的に処理する必要があります
        nx, ny = grid_config.nx, grid_config.ny
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        
        results = {}
        
        # 基本変数の後に混合微分を配置すると仮定
        base_vars = 1 + x_order + y_order  # 関数値 + x方向導関数 + y方向導関数
        
        # f_xy (1階x, 1階y)
        if x_order >= 1 and y_order >= 1:
            f_xy = cp.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    idx = self._compute_vector_index(grid_config, i, j, base_vars)
                    if idx < solution.size:
                        f_xy[i, j] = solution[idx]
            results["f_xy"] = f_xy
        
        # 必要に応じて他の混合微分も抽出
        # 混合微分の格納位置に応じて適切なインデックス計算が必要
        
        return results
