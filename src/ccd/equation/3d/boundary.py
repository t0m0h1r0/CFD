"""
Boundary condition equations for 3D problems.

This module provides equations for boundary conditions (Dirichlet, Neumann)
in 3D problems for the Combined Compact Difference (CCD) method.
"""

import numpy as np
from ..base.base_equation import Equation3D


class DirichletBoundaryEquation3D(Equation3D):
    """Dirichlet boundary condition: ψ(x,y,z) = value"""

    def __init__(self, grid=None):
        """
        Initialize Dirichlet boundary condition
        
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
        # ψ (index 0) constraint
        coeffs = {(0, 0, 0): np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None or k is None:
            raise ValueError("Grid indices i, j, and k must be specified.")
            
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        
        # Valid at boundary points
        return (i == 0 or i == nx - 1 or 
                j == 0 or j == ny - 1 or 
                k == 0 or k == nz - 1)


class NeumannXBoundaryEquation3D(Equation3D):
    """Neumann boundary condition in x-direction: ∂ψ/∂x = value"""

    def __init__(self, grid=None):
        """
        Initialize Neumann boundary condition
        
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
        # ψ_x (index 1) constraint
        coeffs = {(0, 0, 0): np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])}
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None or k is None:
            raise ValueError("Grid indices i, j, and k must be specified.")
            
        nx = self.grid.nx_points
        
        # Valid at x-boundary points
        return i == 0 or i == nx - 1


class NeumannYBoundaryEquation3D(Equation3D):
    """Neumann boundary condition in y-direction: ∂ψ/∂y = value"""

    def __init__(self, grid=None):
        """
        Initialize Neumann boundary condition
        
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
        # ψ_y (index 4) constraint
        coeffs = {(0, 0, 0): np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])}
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None or k is None:
            raise ValueError("Grid indices i, j, and k must be specified.")
            
        ny = self.grid.ny_points
        
        # Valid at y-boundary points
        return j == 0 or j == ny - 1


class NeumannZBoundaryEquation3D(Equation3D):
    """Neumann boundary condition in z-direction: ∂ψ/∂z = value"""

    def __init__(self, grid=None):
        """
        Initialize Neumann boundary condition
        
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
        # ψ_z (index 7) constraint
        coeffs = {(0, 0, 0): np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])}
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None or k is None:
            raise ValueError("Grid indices i, j, and k must be specified.")
            
        nz = self.grid.nz_points
        
        # Valid at z-boundary points
        return k == 0 or k == nz - 1
