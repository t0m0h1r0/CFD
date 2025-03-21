"""
Poisson equation implementation for 3D.

This module provides the Poisson equation implementation for 3D problems
in the Combined Compact Difference (CCD) method.
"""

import numpy as np
from ..base.base_equation import Equation3D


class PoissonEquation3D(Equation3D):
    """Poisson equation: ∇²ψ = ψ_xx + ψ_yy + ψ_zz = f(x,y,z)"""

    def __init__(self, grid=None):
        """
        Initialize Poisson equation
        
        Args:
            grid: 3D Grid object
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        Get stencil coefficients at grid point (i,j,k)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        # Indices in the unknown vector:
        # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        #  0   1    2     3      4    5     6      7    8     9
        
        # Set coefficient 1.0 for ψ_xx (index 2), ψ_yy (index 5), and ψ_zz (index 8)
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
            Boolean validity
        """
        # Poisson equation is valid at all points
        return True


class OriginalEquation3D(Equation3D):
    """Original function equation: ψ(x,y,z) = f(x,y,z)"""
    
    def __init__(self, grid=None):
        """
        Initialize original function equation
        
        Args:
            grid: 3D Grid object
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        Get stencil coefficients at grid point (i,j,k)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        coeffs = {
            (0, 0, 0): np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
            Boolean validity
        """
        # Original equation is valid at all points
        return True


class CustomEquation3D(Equation3D):
    """Custom equation with user-defined coefficients"""

    def __init__(self, f_func=None, coeff=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], grid=None):
        """
        Initialize custom equation
        
        Args:
            f_func: Right-hand side function f(x,y,z)
            coeff: Coefficients for [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
            grid: 3D Grid object
        """
        super().__init__(grid)
        self.f_func = f_func
        self.coeff = coeff

    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        Get stencil coefficients at grid point (i,j,k)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        # Stencil coefficients (independent of position)
        return {(0, 0, 0): np.array(self.coeff)}

    def get_rhs(self, i=None, j=None, k=None):
        """
        Get right-hand side value at grid point (i,j,k)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
            
        Returns:
            Right-hand side value
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None or k is None:
            raise ValueError("Grid indices i, j, and k must be specified.")
            
        if self.f_func is None:
            return 0.0
            
        # Get grid point coordinates
        x, y, z = self.grid.get_point(i, j, k)
        return self.f_func(x, y, z)

    def is_valid_at(self, i=None, j=None, k=None):
        """
        Check if equation is valid at grid point (i,j,k)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
            
        Returns:
            Boolean validity
        """
        # Custom equation is valid at all points
        return True
