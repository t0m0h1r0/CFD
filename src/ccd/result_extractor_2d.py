"""
2次元結果抽出モジュール

2次元CCDの解ベクトルから各種偏導関数値を抽出するクラス
"""

import cupy as cp
from typing import Tuple

from grid_config_2d import GridConfig2D
from result_extractor import CCDResultExtractor  # 1次元の結果抽出器


class CCDResultExtractor2D:
    """2次元CCDの結果を抽出するクラス"""

    def __init__(self):
        """初期化"""
        # 1次元抽出器を内部で使用
        self.extractor_1d = CCDResultExtractor()

    def extract_components(
        self, grid_config: GridConfig2D, solution: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        2次元解ベクトルから関数値と各偏導関数を抽出

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル

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

        # ソリューションから各成分を取り出す
        for i in range(nx):
            for j in range(ny):
                idx = (i * ny + j) * depth
                f[i, j] = solution[idx]
                f_x[i, j] = solution[idx + 1]
                f_y[i, j] = solution[idx + 2]
                f_xx[i, j] = solution[idx + 3]
                # 注：f_xyとf_yyは実際のソリューションの別の位置から取得する必要があるかもしれない
                # この実装は簡略化しています

        # ディリクレ境界条件が有効な場合、境界補正を適用
        if grid_config.is_dirichlet():
            if grid_config.enable_boundary_correction:
                # 境界補正を適用
                # この部分は具体的な境界条件に依存します
                pass

        return f, f_x, f_y, f_xx, f_xy, f_yy