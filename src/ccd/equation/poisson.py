# equation/poisson.py
import cupy as cp
from typing import Dict, Callable
from grid import Grid
from .base import Equation


class PoissonEquation(Equation):
    """ポアソン方程式: psi''(x) = f(x)"""

    def __init__(self, f_func: Callable[[float], float]):
        """
        ポアソン方程式の初期化

        Args:
            f_func: 右辺の関数 f(x)
        """
        self.f_func = f_func

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        ポアソン方程式のステンシル係数を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # ψ''項の係数を1に設定
        coeffs = {
            0: cp.array([0, 0, 1, 0]),  # 中央点 psi''に対応する係数を1に設定
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
        """
        ポアソン方程式が適用可能かを判定
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            方程式が適用可能な場合True
        """
        return True  # ポアソン方程式は全ての点で適用可能