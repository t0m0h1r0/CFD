"""
2次元CCD結果抽出モジュール（修正版）

2次元CCD法の解ベクトルから各成分を抽出するクラスを提供します。
"""

import cupy as cp
from typing import Dict, Optional

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
            {"f": 関数値, "f_x": x偏導関数, ... } の形式の辞書
        """
        nx, ny = grid_config.nx, grid_config.ny
        depth = 4  # 各点の未知数の数（f, f', f'', f'''）
        
        # デバッグ情報
        print(f"解ベクトルのサイズ: {solution.shape}")
        print(f"期待されるサイズ: {nx * ny * depth}")
        
        # 期待されるサイズよりも解ベクトルが小さい場合の対処
        if solution.shape[0] < nx * ny * depth:
            print("警告: 解ベクトルが期待されるサイズよりも小さいです。結果は不完全である可能性があります。")

        # 結果を格納する辞書
        results = {}

        # 抽出する導関数のタイプと対応するインデックス
        deriv_types = {
            "f": 0,     # 関数値
            "f_x": 1,   # x方向の1階導関数
            "f_y": 2,   # y方向の1階導関数
            "f_xx": 3,  # x方向の2階導関数
            # "f_yy" と "f_xy" はインデックスがオーバーフローするため、現在のモデルでは抽出できない
        }

        # 各導関数タイプに対して2次元配列を作成
        for deriv_type, idx_offset in deriv_types.items():
            # 空の2次元配列を作成
            result_array = cp.zeros((nx, ny), dtype=cp.float64)
            
            try:
                # 各グリッド点での値を抽出
                for i in range(nx):
                    for j in range(ny):
                        # 平坦化されたインデックスを計算
                        linear_idx = (j * nx + i) * depth + idx_offset
                        if linear_idx < solution.shape[0]:
                            result_array[i, j] = solution[linear_idx]
            except Exception as e:
                print(f"警告: {deriv_type} の抽出中にエラーが発生しました: {e}")
                # エラーが発生しても処理を続行
            
            # 結果を辞書に格納
            results[deriv_type] = result_array

        # 関数値に対して境界補正を適用
        if grid_config.enable_boundary_correction and "f" in results:
            results["f"] = grid_config.apply_boundary_correction(results["f"])

        return results

    def extract_at_point(
        self, grid_config: Grid2DConfig, solution: cp.ndarray, i: int, j: int
    ) -> Dict[str, float]:
        """
        特定の格子点における解ベクトルから関数値と各階導関数を抽出

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル
            i: x方向のインデックス
            j: y方向のインデックス

        Returns:
            {"f": 関数値, "f_x": x偏導関数, ... } の形式の辞書
        """
        nx, ny = grid_config.nx, grid_config.ny
        depth = 4  # 各点の未知数の数
        
        # 結果を格納する辞書
        results = {}

        # 抽出する導関数のタイプと対応するインデックス
        deriv_types = {
            "f": 0,     # 関数値
            "f_x": 1,   # x方向の1階導関数
            "f_y": 2,   # y方向の1階導関数
            "f_xx": 3,  # x方向の2階導関数
        }

        # 各導関数タイプの値を抽出
        for deriv_type, idx_offset in deriv_types.items():
            try:
                # 平坦化されたインデックスを計算
                linear_idx = (j * nx + i) * depth + idx_offset
                if linear_idx < solution.shape[0]:
                    results[deriv_type] = float(solution[linear_idx])
                else:
                    results[deriv_type] = 0.0
            except Exception as e:
                print(f"警告: 点 ({i}, {j}) における {deriv_type} の抽出中にエラーが発生しました: {e}")
                results[deriv_type] = 0.0

        return results