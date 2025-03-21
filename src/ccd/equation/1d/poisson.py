"""
Poisson equation implementation for 1D.

This module provides the Poisson equation implementation for 1D problems
in the Combined Compact Difference (CCD) method.
"""

import numpy as np
from ..base.base_equation import Equation1D


class PoissonEquation(Equation1D):
    """Poisson equation: ψ''(x) = f(x)"""

    def __init__(self, grid=None):
        """
        Initialize Poisson equation
        
        Args:
            grid: 1D Grid object
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None):
        """
        Get stencil coefficients at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        # For Poisson equation, stencil coefficients are independent of the position
        # i is not actually used
        return {0: np.array([0, 0, 1, 0])}

    def is_valid_at(self, i=None):
        """
        Check if equation is valid at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Boolean validity
        """
        # Poisson equation is valid at all points
        return True


class OriginalEquation(Equation1D):
    """Original function equation: ψ(x) = f(x)"""
    
    def __init__(self, grid=None):
        """
        Initialize original function equation
        
        Args:
            grid: 1D Grid object
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None):
        """
        Get stencil coefficients at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        return {0: np.array([1, 0, 0, 0])}

    def is_valid_at(self, i=None):
        """
        Check if equation is valid at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Boolean validity
        """
        # Original equation is valid at all points
        return True


class CustomEquation(Equation1D):
    """Custom equation with user-defined coefficients"""

    def __init__(self, f_func=None, coeff=[1, 0, 0, 0], grid=None):
        """
        Initialize custom equation
        
        Args:
            f_func: Right-hand side function f(x)
            coeff: Coefficients [c0, c1, c2, c3] for [ψ, ψ', ψ'', ψ''']
            grid: 1D Grid object
        """
        super().__init__(grid)
        self.f_func = f_func
        self.coeff = coeff

    def get_stencil_coefficients(self, i=None):
        """
        Get stencil coefficients at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        # Stencil coefficients (independent of position)
        return {0: np.array(self.coeff)}

    def get_rhs(self, i=None):
        """
        Get right-hand side value at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Right-hand side value
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        if self.f_func is None:
            return 0.0
            
        # Get grid point coordinate
        x = self.grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, i=None):
        """
        Check if equation is valid at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Boolean validity
        """
        # Custom equation is valid at all points
        return True
