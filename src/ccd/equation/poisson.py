import cupy as cp
from .base1d import Equation

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
    
from .base2d import Equation2D
class PoissonEquation2D(Equation2D):
    """2D Poisson equation: ψ_xx + ψ_yy = f(x,y)"""
    
    def __init__(self, f_func):
        """
        Initialize with right-hand side function
        
        Args:
            f_func: Function f(x,y) for right-hand side
        """
        self.f_func = f_func
    
    def get_stencil_coefficients(self, grid, i, j):
        """
        Get stencil coefficients at grid point (i,j)
        
        For 2D Poisson equation, we need coefficients for ψ_xx and ψ_yy
        """
        # Indices in the unknown vector:
        # 0: ψ, 1: ψ_x, 2: ψ_xx, 3: ψ_xxx, 4: ψ_y, 5: ψ_yy, 6: ψ_yyy
        
        # Set coefficient 1.0 for ψ_xx (index 2) and 1.0 for ψ_yy (index 5)
        coeffs = {
            (0, 0): cp.array([0, 0, 1, 0, 0, 1, 0])
        }
        return coeffs
    
    def get_rhs(self, grid, i, j):
        """Get right-hand side value f(x,y) at point (i,j)"""
        x, y = grid.get_point(i, j)
        return self.f_func(x, y)
    
    def is_valid_at(self, grid, i, j):
        """Poisson equation is valid at interior points"""
        return True
