"""
Poisson equation implementation for 2D.

This module provides the Poisson equation implementation for 2D problems
in the Combined Compact Difference (CCD) method.
"""

import numpy as np
from ..base.base_equation import Equation2D


class PoissonEquation2D(Equation2D):
    """Poisson equation: ∇²ψ = ψ_xx + ψ_yy = f(x,y)"""

    def __init__(self, grid=None):
        """
        Initialize Poisson equation
        
        Args:
            grid: 2D Grid object
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
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
            Boolean validity
        """
        # Poisson equation is valid at all points
        return True


class OriginalEquation2D(Equation2D):
    """Original function equation: ψ(x,y) = f(x,y)"""
    
    def __init__(self, grid=None):
        """
        Initialize original function equation
        
        Args:
            grid: 2D Grid object
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        coeffs = {
            (0, 0): np.array([1, 0, 0, 0, 0, 0, 0])
        }
        return coeffs

    def is_valid_at(self, i=None, j=None):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Boolean validity
        """
        # Original equation is valid at all points
        return True


class CustomEquation2D(Equation2D):
    """Custom equation with user-defined coefficients"""

    def __init__(self, f_func=None, coeff=[1, 0, 0, 0, 0, 0, 0], grid=None):
        """
        Initialize custom equation
        
        Args:
            f_func: Right-hand side function f(x,y)
            coeff: Coefficients for [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
            grid: 2D Grid object
        """
        super().__init__(grid)
        self.f_func = f_func
        self.coeff = coeff

    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        # Stencil coefficients (independent of position)
        return {(0, 0): np.array(self.coeff)}

    def get_rhs(self, i=None, j=None):
        """
        Get right-hand side value at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Right-hand side value
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
            
        if self.f_func is None:
            return 0.0
            
        # Get grid point coordinates
        x, y = self.grid.get_point(i, j)
        return self.f_func(x, y)

    def is_valid_at(self, i=None, j=None):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Boolean validity
        """
        # Custom equation is valid at all points
        return True
