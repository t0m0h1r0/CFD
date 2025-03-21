"""
Boundary point equations for 1D compact difference method.

This module provides equations for computing derivatives at boundary points
using the Combined Compact Difference (CCD) method in 1D.
"""

import numpy as np
from ..base.base_equation import Equation1D


class LeftBoundary1stDerivativeEquation(Equation1D):
    """1st derivative equation for left boundary point in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for left boundary point
        
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
            0: np.array([-42, 0, 0, 1]) * np.array([h**-3, h**-2, h**-1, h**0]),
            -1: np.array([24, 60, 12, 9]) * np.array([h**-3, h**-2, h**-1, h**0]),
            -2: np.array([18, 0, 0, 0]) * np.array([h**-3, h**-2, h**-1, h**0])
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
        return i == n - 1raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        h = self.grid.get_spacing()
        coeffs = {
            0: np.array([9/2, 1, 0, 0]) * np.array([h**-1, h**0, h**1, h**2]),
            1: np.array([-4, 4, -1, 1/3]) * np.array([h**-1, h**0, h**1, h**2]),
            2: np.array([-(1/2), 0, 0, 0]) * np.array([h**-1, h**0, h**1, h**2])
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
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        return i == 0


class LeftBoundary2ndDerivativeEquation(Equation1D):
    """2nd derivative equation for left boundary point in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for left boundary point
        
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
            0: np.array([-16, 0, 1, 0]) * np.array([h**-2, h**-1, h**0, h**1]),
            1: np.array([12, -20, 5, -(7/3)]) * np.array([h**-2, h**-1, h**0, h**1]),
            2: np.array([4, 0, 0, 0]) * np.array([h**-2, h**-1, h**0, h**1])
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
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        return i == 0


class LeftBoundary3rdDerivativeEquation(Equation1D):
    """3rd derivative equation for left boundary point in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for left boundary point
        
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
            0: np.array([42, 0, 0, 1]) * np.array([h**-3, h**-2, h**-1, h**0]),
            1: np.array([-24, 60, -12, 9]) * np.array([h**-3, h**-2, h**-1, h**0]),
            2: np.array([-18, 0, 0, 0]) * np.array([h**-3, h**-2, h**-1, h**0])
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
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        return i == 0


class RightBoundary1stDerivativeEquation(Equation1D):
    """1st derivative equation for right boundary point in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for right boundary point
        
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
            0: np.array([-(9/2), 1, 0, 0]) * np.array([h**-1, h**0, h**1, h**2]),
            -1: np.array([4, 4, 1, 1/3]) * np.array([h**-1, h**0, h**1, h**2]),
            -2: np.array([1/2, 0, 0, 0]) * np.array([h**-1, h**0, h**1, h**2])
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
        return i == n - 1


class RightBoundary2ndDerivativeEquation(Equation1D):
    """2nd derivative equation for right boundary point in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for right boundary point
        
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
            0: np.array([-16, 0, 1, 0]) * np.array([h**-2, h**-1, h**0, h**1]),
            -1: np.array([12, 20, 5, 7/3]) * np.array([h**-2, h**-1, h**0, h**1]),
            -2: np.array([4, 0, 0, 0]) * np.array([h**-2, h**-1, h**0, h**1])
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
        return i == n - 1


class RightBoundary3rdDerivativeEquation(Equation1D):
    """3rd derivative equation for right boundary point in 1D"""

    def __init__(self, grid=None):
        """
        Initialize equation for right boundary point
        
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
            