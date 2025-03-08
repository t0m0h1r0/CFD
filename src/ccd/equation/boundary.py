import cupy as cp
from .base import Equation

class DirichletBoundaryEquation(Equation):
    """ディリクレ境界条件: psi(x) = value"""

    def __init__(self, value):
        self.value = value

    def get_stencil_coefficients(self, grid, i):
        return {0: cp.array([1, 0, 0, 0])}

    def get_rhs(self, grid, i):
        return self.value

    def is_valid_at(self, grid, i):
        n = grid.n_points
        return i == 0 or i == n - 1


class NeumannBoundaryEquation(Equation):
    """ノイマン境界条件: psi'(x) = value"""

    def __init__(self, value):
        self.value = value

    def get_stencil_coefficients(self, grid, i):
        return {0: cp.array([0, 1, 0, 0])}

    def get_rhs(self, grid, i):
        return self.value

    def is_valid_at(self, grid, i):
        n = grid.n_points
        return i == 0 or i == n - 1