import numpy as np
from .base import Equation3D

class PoissonEquation3D(Equation3D):
    """3D Poisson equation: ψ_xx + ψ_yy + ψ_zz = f(x,y,z)"""
    
    def __init__(self, grid=None):
        """
        Initialize with right-hand side function
        
        Args:
            grid: Grid3D object
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        Get stencil coefficients at grid point (i,j,k)
        
        For 3D Poisson equation, we need coefficients for ψ_xx, ψ_yy, and ψ_zz
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
        
        Returns:
            Dictionary with stencil coefficients
        """
        # Indices in the unknown vector:
        # 0: ψ, 1: ψ_x, 2: ψ_xx, 3: ψ_xxx, 4: ψ_y, 5: ψ_yy, 6: ψ_yyy, 7: ψ_z, 8: ψ_zz, 9: ψ_zzz
        
        # Set coefficient 1.0 for ψ_xx (index 2), 1.0 for ψ_yy (index 5), and 1.0 for ψ_zz (index 8)
        coeffs = {
            (0, 0, 0): np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        }
        return coeffs
        
    def is_valid_at(self, i=None, j=None, k=None):
        """
        Check if equation is valid at grid point (i,j,k)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
            
        Returns:
            Boolean indicating if equation is valid
        """
        # Poisson equation is valid at all points
        return True
        
    def get_equation_type(self):
        """
        3Dポアソン方程式の種類を返す - 支配方程式

        Returns:
            str: "governing"
        """
        return "governing"