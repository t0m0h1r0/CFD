# equation/poisson.py
import cupy as cp
from typing import Dict, Callable
from grid import Grid
from .base import Equation


class CustomEquation(Equation):
    """ポアソン方程式: psi''(x) = f(x)"""

    def __init__(self, f_func: Callable[[float], float], coeff=[1,0,0,0]):
        """
        方程式の初期化

        Args:
            f_func: 右辺の関数 f(x)
        """
        self.f_func = f_func
        self.coeff = coeff

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        方程式のステンシル係数を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            ステンシル係数の辞書
        """
        # ψ''項の係数を1に設定
        coeffs = {
            0: cp.array(self.coeff),
        }

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        右辺関数f(x)の値

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            右辺の値
        """
        # グリッド点の座標値を取得
        x = grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        return True  # ポアソン方程式は全ての点で適用可能
