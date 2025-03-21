"""
Boundary condition equations for 1D problems.

This module provides equations for boundary conditions (Dirichlet, Neumann)
in 1D problems for the Combined Compact Difference (CCD) method.
"""

import numpy as np
from ..base.base_equation import Equation1D


class DirichletBoundaryEquation(Equation1D):
    """Dirichlet boundary condition: ψ(x) = value"""

    def __init__(self, grid=None):
        """
        Initialize Dirichlet boundary condition
        
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None:
            raise ValueError("Grid index i must be specified.")
            
        n = self.grid.n_points
        return i == 0 or i == n - 1


class NeumannBoundaryEquation(Equation1D):
    """Neumann boundary condition: ψ'(x) = value"""

    def __init__(self, grid=None):
        """
        Initialize Neumann boundary condition
        
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
        return {0: np.array([0, 1, 0, 0])}

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
        return i == 0 or i == n - 1
