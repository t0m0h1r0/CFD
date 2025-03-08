import cupy as cp
from .base import Equation

class PoissonEquation(Equation):
    """ポアソン方程式: psi''(x) = f(x)"""

    def __init__(self, f_func):
        self.f_func = f_func

    def get_stencil_coefficients(self, grid, i):
        return {0: cp.array([0, 0, 1, 0])}

    def get_rhs(self, grid, i):
        x = grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, grid, i):
        return True