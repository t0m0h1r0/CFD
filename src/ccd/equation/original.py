import cupy as cp
from .base1d import Equation

class OriginalEquation(Equation):
    def __init__(self, f_func):
        self.f_func = f_func

    def get_stencil_coefficients(self, grid, i):
        return {0: cp.array([1, 0, 0, 0])}

    def get_rhs(self, grid, i):
        x = grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, grid, i):
        return True
    
from .base2d import Equation2D
class OriginalEquation2D(Equation2D):    
    def __init__(self, f_func):
        self.f_func = f_func
    
    def get_stencil_coefficients(self, grid, i, j):
        coeffs = {
            (0, 0): cp.array([1, 0, 0, 0, 0, 0, 0])
        }
        return coeffs
    
    def get_rhs(self, grid, i, j):
        x, y = grid.get_point(i, j)
        return self.f_func(x, y)
    
    def is_valid_at(self, grid, i, j):
        return grid.is_interior_point(i, j)
