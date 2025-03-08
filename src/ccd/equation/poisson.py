# equation/poisson.py
import cupy as np
from typing import Dict, Callable
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

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        """ポアソン方程式の係数を返す"""
        coeffs = {
            0: np.array([0, 0, 1, 0]),  # 中央点 psi''に対応する係数を1に設定
        }

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        """右辺関数f(x)の値"""
        x = i * h + self.grid.x_min if hasattr(self, "grid") else i * h
        return self.f_func(x)

    def is_valid_at(self, i: int, n: int) -> bool:
        """内部点でのみ有効"""
        return True  # ポアソン方程式は全ての点で適用可能
