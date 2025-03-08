import cupy as cp
from .base1d import Equation

class LeftBoundary1stDerivativeEquation(Equation):
    """左境界点での1階導関数関係式"""

    def get_stencil_coefficients(self, grid, i):
        h = grid.get_spacing()
        coeffs = {
            0: cp.array([9/2, 1, 0, 0]) * cp.array([h**-1, h**0, h**1, h**2]),
            1: cp.array([-4, 4, -1, 1/3]) * cp.array([h**-1, h**0, h**1, h**2]),
            2: cp.array([-(1/2), 0, 0, 0]) * cp.array([h**-1, h**0, h**1, h**2])
        }
        return coeffs

    def get_rhs(self, grid, i):
        return 0.0

    def is_valid_at(self, grid, i):
        return i == 0


class LeftBoundary2ndDerivativeEquation(Equation):
    """左境界点での2階導関数関係式"""

    def get_stencil_coefficients(self, grid, i):
        h = grid.get_spacing()
        coeffs = {
            0: cp.array([-16, 0, 1, 0]) * cp.array([h**-2, h**-1, h**0, h**1]),
            1: cp.array([12, -20, 5, -(7/3)]) * cp.array([h**-2, h**-1, h**0, h**1]),
            2: cp.array([4, 0, 0, 0]) * cp.array([h**-2, h**-1, h**0, h**1])
        }
        return coeffs

    def get_rhs(self, grid, i):
        return 0.0

    def is_valid_at(self, grid, i):
        return i == 0


class LeftBoundary3rdDerivativeEquation(Equation):
    """左境界点での3階導関数関係式"""

    def get_stencil_coefficients(self, grid, i):
        h = grid.get_spacing()
        coeffs = {
            0: cp.array([42, 0, 0, 1]) * cp.array([h**-3, h**-2, h**-1, h**0]),
            1: cp.array([-24, 60, -12, 9]) * cp.array([h**-3, h**-2, h**-1, h**0]),
            2: cp.array([-18, 0, 0, 0]) * cp.array([h**-3, h**-2, h**-1, h**0])
        }
        return coeffs

    def get_rhs(self, grid, i):
        return 0.0

    def is_valid_at(self, grid, i):
        return i == 0