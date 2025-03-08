import cupy as cp
from .base import Equation

class Internal1stDerivativeEquation(Equation):
    """内部点での1階導関数関係式"""

    def get_stencil_coefficients(self, grid, i):
        h = grid.get_spacing()
        coeffs = {
            -1: cp.array([35/32, 19/32, 1/8, 1/96]) * cp.array([h**-1, h**0, h**1, h**2]),
            0: cp.array([0, 1, 0, 0]) * cp.array([h**-1, h**0, h**1, h**2]),
            1: cp.array([-35/32, 19/32, -1/8, 1/96]) * cp.array([h**-1, h**0, h**1, h**2])
        }
        return coeffs

    def get_rhs(self, grid, i):
        return 0.0

    def is_valid_at(self, grid, i):
        n = grid.n_points
        return 0 < i < n - 1


class Internal2ndDerivativeEquation(Equation):
    """内部点での2階導関数関係式"""

    def get_stencil_coefficients(self, grid, i):
        h = grid.get_spacing()
        coeffs = {
            -1: cp.array([-4, -29/16, -5/16, -1/48]) * cp.array([h**-2, h**-1, h**0, h**1]),
            0: cp.array([8, 0, 1, 0]) * cp.array([h**-2, h**-1, h**0, h**1]),
            1: cp.array([-4, 29/16, -5/16, 1/48]) * cp.array([h**-2, h**-1, h**0, h**1])
        }
        return coeffs

    def get_rhs(self, grid, i):
        return 0.0

    def is_valid_at(self, grid, i):
        n = grid.n_points
        return 0 < i < n - 1


class Internal3rdDerivativeEquation(Equation):
    """内部点での3階導関数関係式"""

    def get_stencil_coefficients(self, grid, i):
        h = grid.get_spacing()
        coeffs = {
            -1: cp.array([-105/16, -105/16, -15/8, -3/16]) * cp.array([h**-3, h**-2, h**-1, h**0]),
            0: cp.array([0, 0, 0, 1]) * cp.array([h**-3, h**-2, h**-1, h**0]),
            1: cp.array([105/16, -105/16, 15/8, -3/16]) * cp.array([h**-3, h**-2, h**-1, h**0])
        }
        return coeffs

    def get_rhs(self, grid, i):
        return 0.0

    def is_valid_at(self, grid, i):
        n = grid.n_points
        return 0 < i < n - 1