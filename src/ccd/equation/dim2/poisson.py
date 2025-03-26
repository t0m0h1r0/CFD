import numpy as np
from .base import Equation2D

class PoissonEquation2D(Equation2D):
    """2D Poisson equation: ψ_xx + ψ_yy = f(x,y)"""
    
    def __init__(self, grid=None):
        """
        Initialize with right-hand side function
        
        Args:
            grid: Grid2D object
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        For 2D Poisson equation, we need coefficients for ψ_xx and ψ_yy
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
        
        Returns:
            Dictionary with stencil coefficients
        """
        # Indices in the unknown vector:
        # 0: ψ, 1: ψ_x, 2: ψ_xx, 3: ψ_xxx, 4: ψ_y, 5: ψ_yy, 6: ψ_yyy
        
        # Set coefficient 1.0 for ψ_xx (index 2) and 1.0 for ψ_yy (index 5)
        coeffs = {
            (0, 0): np.array([0, 0, 1, 0, 0, 1, 0])
        }
        return coeffs
        
    def is_valid_at(self, i=None, j=None):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Boolean indicating if equation is valid
        """
        # Poisson equation is valid at all points
        return True
        
    def get_equation_type(self):
        """
        2Dポアソン方程式の種類を返す - 支配方程式

        Returns:
            str: "governing"
        """
        return "governing"