import cupy as cp
from .base import Equation

class EssentialEquation(Equation):
    """特定の未知数に対する基本方程式"""

    def __init__(self, k, f_func):
        if k not in [0, 1, 2, 3]:
            k = 0
        self.k = k
        self.f_func = f_func

    def get_stencil_coefficients(self, grid, i):
        coeffs = cp.zeros(4)
        coeffs[self.k] = 1.0
        return {0: coeffs}

    def get_rhs(self, grid, i):
        x = grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, grid, i):
        return True