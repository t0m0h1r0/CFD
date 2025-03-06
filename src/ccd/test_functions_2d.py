"""
2次元テスト関数モジュール

2次元CCDソルバーのテストに使用する関数群を定義します。
"""

import cupy as cp
from dataclasses import dataclass
from typing import Callable, List, Tuple

# 既存のテスト関数クラスを活用
from test_functions import TestFunction
from test_functions_adapter import TestFunction2D, create_test_functions


@dataclass
class TestFunction2DExplicit:
    """
    2次元テスト関数と明示的な導関数を保持するデータクラス
    """

    name: str
    f: Callable[[float, float], float]
    df_dx: Callable[[float, float], float]
    df_dy: Callable[[float, float], float]
    d2f_dx2: Callable[[float, float], float]
    d2f_dxdy: Callable[[float, float], float]
    d2f_dy2: Callable[[float, float], float]


class TestFunction2DFactory:
    """2次元テスト関数を生成するファクトリークラス"""

    @staticmethod
    def create_standard_functions() -> List[TestFunction2DExplicit]:
        """標準的な2次元テスト関数セットを生成"""
        test_functions = []

        # 1. ガウス関数
        test_functions.append(
            TestFunction2DExplicit(
                name="Gaussian",
                f=lambda x, y: cp.exp(-(x**2 + y**2)),
                df_dx=lambda x, y: -2 * x * cp.exp(-(x**2 + y**2)),
                df_dy=lambda x, y: -2 * y * cp.exp(-(x**2 + y**2)),
                d2f_dx2=lambda x, y: (-2 + 4 * x**2) * cp.exp(-(x**2 + y**2)),
                d2f_dxdy=lambda x, y: 4 * x * y * cp.exp(-(x**2 + y**2)),
                d2f_dy2=lambda x, y: (-2 + 4 * y**2) * cp.exp(-(x**2 + y**2)),
            )
        )

        # 2. サイン関数
        test_functions.append(
            TestFunction2DExplicit(
                name="Sine",
                f=lambda x, y: cp.sin(cp.pi * x) * cp.sin(cp.pi * y),
                df_dx=lambda x, y: cp.pi * cp.cos(cp.pi * x) * cp.sin(cp.pi * y),
                df_dy=lambda x, y: cp.pi * cp.sin(cp.pi * x) * cp.cos(cp.pi * y),
                d2f_dx2=lambda x, y: -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y),
                d2f_dxdy=lambda x, y: (cp.pi**2) * cp.cos(cp.pi * x) * cp.cos(cp.pi * y),
                d2f_dy2=lambda x, y: -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y),
            )
        )

        # 3. 多項式関数
        test_functions.append(
            TestFunction2DExplicit(
                name="Polynomial",
                f=lambda x, y: (1 - x**2) * (1 - y**2),
                df_dx=lambda x, y: -2 * x * (1 - y**2),
                df_dy=lambda x, y: -2 * y * (1 - x**2),
                d2f_dx2=lambda x, y: -2 * (1 - y**2),
                d2f_dxdy=lambda x, y: 4 * x * y,
                d2f_dy2=lambda x, y: -2 * (1 - x**2),
            )
        )

        # 4. 指数関数
        test_functions.append(
            TestFunction2DExplicit(
                name="Exponential",
                f=lambda x, y: cp.exp(x + y),
                df_dx=lambda x, y: cp.exp(x + y),
                df_dy=lambda x, y: cp.exp(x + y),
                d2f_dx2=lambda x, y: cp.exp(x + y),
                d2f_dxdy=lambda x, y: cp.exp(x + y),
                d2f_dy2=lambda x, y: cp.exp(x + y),
            )
        )

        return test_functions

    @staticmethod
    def evaluate_on_grid(
        test_func: TestFunction2DExplicit,
        x_grid: cp.ndarray,
        y_grid: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        テスト関数をグリッド上で評価し、関数値と各偏導関数値を返す

        Args:
            test_func: 2次元テスト関数
            x_grid: xグリッド点の配列
            y_grid: yグリッド点の配列

        Returns:
            (f, f_x, f_y, f_xx, f_xy, f_yy)のタプル、各要素は2次元配列
        """
        nx, ny = len(x_grid), len(y_grid)
        
        # 結果を格納する配列を初期化
        f = cp.zeros((nx, ny))
        f_x = cp.zeros((nx, ny))
        f_y = cp.zeros((nx, ny))
        f_xx = cp.zeros((nx, ny))
        f_xy = cp.zeros((nx, ny))
        f_yy = cp.zeros((nx, ny))
        
        # グリッド上の各点で関数を評価
        for i in range(nx):
            for j in range(ny):
                x, y = x_grid[i], y_grid[j]
                f[i, j] = test_func.f(x, y)
                f_x[i, j] = test_func.df_dx(x, y)
                f_y[i, j] = test_func.df_dy(x, y)
                f_xx[i, j] = test_func.d2f_dx2(x, y)
                f_xy[i, j] = test_func.d2f_dxdy(x, y)
                f_yy[i, j] = test_func.d2f_dy2(x, y)
        
        return f, f_x, f_y, f_xx, f_xy, f_yy