# equation/essential.py
import cupy as cp
from typing import Dict, Callable
from grid import Grid
from .base import Equation


class EssentialEquation(Equation):
    """
    Essential方程式クラス
    特定のグリッド点において、指定した未知数の係数を1に、それ以外を0にする方程式を表現
    """

    def __init__(self, k: int, f_func: Callable[[float], float]):
        """
        初期化

        Args:
            k: 係数を1にするインデックス (0: ψ, 1: ψ', 2: ψ'', 3: ψ''')
            value: 右辺の値
            target_point: この方程式を適用する特定のグリッド点 (None の場合は is_valid_at で決定)
        """
        if k not in [0, 1, 2, 3]:
            raise ValueError("Index k must be in [0, 1, 2, 3]")

        self.k = k
        self.f_func = f_func

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        グリッド点iにおけるステンシル係数を返す
        k番目の係数を1に、それ以外を0にする

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            {0: [a, b, c, d]} という形式の辞書、ただしk番目の値だけが1で他は0
        """
        # 全ての係数を0で初期化
        coeffs = cp.zeros(4)

        # k番目の係数を1に設定
        coeffs[self.k] = 1.0

        # 現在のグリッド点（オフセット0）のみに係数を設定
        return {0: coeffs}

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        方程式の右辺を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            指定された値
        """
        # グリッド点の座標値を取得
        x = grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        # デフォルトでは全てのグリッド点で有効
        return True
