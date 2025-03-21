"""
Internal point equations for 1D compact difference method.

This module provides equations for computing derivatives at interior points
using the Combined Compact Difference (CCD) method in 1D.
"""

import numpy as np
from ..base.base_equation import Equation1D


class Internal1stDerivativeEquation(Equation1D):
    """1st derivative equation for interior points in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for interior points
        
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        h = self.grid.get_spacing()
        coeffs = {
            -1: np.array([35/32, 19/32, 1/8, 1/96]) * np.array([h**-1, h**0, h**1, h**2]),
            0: np.array([0, 1, 0, 0]) * np.array([h**-1, h**0, h**1, h**2]),
            1: np.array([-35/32, 19/32, -1/8, 1/96]) * np.array([h**-1, h**0, h**1, h**2])
        }
        return coeffs

    def is_valid_at(self, i=None):
        """
        Check if equation is valid at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        n = self.grid.n_points
        return 0 < i < n - 1


class Internal2ndDerivativeEquation(Equation1D):
    """2nd derivative equation for interior points in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for interior points
        
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        h = self.grid.get_spacing()
        coeffs = {
            -1: np.array([-4, -29/16, -5/16, -1/48]) * np.array([h**-2, h**-1, h**0, h**1]),
            0: np.array([8, 0, 1, 0]) * np.array([h**-2, h**-1, h**0, h**1]),
            1: np.array([-4, 29/16, -5/16, 1/48]) * np.array([h**-2, h**-1, h**0, h**1])
        }
        return coeffs

    def get_rhs(self, i=None):
        """
        Get right-hand side value at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Right-hand side value
        """
        return 0.0

    def is_valid_at(self, i=None):
        """
        Check if equation is valid at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        n = self.grid.n_points
        return 0 < i < n - 1


class Internal3rdDerivativeEquation(Equation1D):
    """3rd derivative equation for interior points in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for interior points
        
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        h = self.grid.get_spacing()
        coeffs = {
            -1: np.array([-105/16, -105/16, -15/8, -3/16]) * np.array([h**-3, h**-2, h**-1, h**0]),
            0: np.array([0, 0, 0, 1]) * np.array([h**-3, h**-2, h**-1, h**0]),
            1: np.array([105/16, -105/16, 15/8, -3/16]) * np.array([h**-3, h**-2, h**-1, h**0])
        }
        return coeffs

    def get_rhs(self, i=None):
        """
        Get right-hand side value at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Right-hand side value
        """
        return 0.0

    def is_valid_at(self, i=None):
        """
        Check if equation is valid at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        n = self.grid.n_points
        return 0 < i < n - 1
