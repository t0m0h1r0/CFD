"""
Equation converter for dimension conversion.

This module provides utilities for converting 1D equations to 2D and 3D,
enabling code reuse and simplifying maintenance.
"""

import numpy as np
from .base.base_equation import Equation2D, Equation3D


class Equation1Dto2DConverter:
    """Factory class for converting 1D equations to 2D"""
    
    @staticmethod
    def to_x(equation_1d, grid=None, direction_only=False):
        """
        Convert a 1D equation to a 2D equation applied in x-direction
        
        Args:
            equation_1d: 1D equation instance
            grid: 2D Grid object
            direction_only: If True, only apply in x-direction
            
        Returns:
            2D equation instance for x-direction
        """
        return DirectionalEquation2D(equation_1d, 'x', direction_only, grid)
    
    @staticmethod
    def to_y(equation_1d, grid=None, direction_only=False):
        """
        Convert a 1D equation to a 2D equation applied in y-direction
        
        Args:
            equation_1d: 1D equation instance
            grid: 2D Grid object
            direction_only: If True, only apply in y-direction
            
        Returns:
            2D equation instance for y-direction
        """
        return DirectionalEquation2D(equation_1d, 'y', direction_only, grid)
    
    @staticmethod
    def to_xy(equation_1d_x, equation_1d_y=None, grid=None):
        """
        Convert 1D equations to a 2D equation applied in both directions
        
        Args:
            equation_1d_x: 1D equation for x-direction
            equation_1d_y: 1D equation for y-direction (default: same as x)
            grid: 2D Grid object
            
        Returns:
            2D equation instance for both directions
        """
        if equation_1d_y is None:
            equation_1d_y = equation_1d_x
            
        x_eq = DirectionalEquation2D(equation_1d_x, 'x', False, grid)
        y_eq = DirectionalEquation2D(equation_1d_y, 'y', False, grid)
        
        return CombinedDirectionalEquation2D(x_eq, y_eq, grid)


class Equation1Dto3DConverter:
    """Factory class for converting 1D equations to 3D"""
    
    @staticmethod
    def to_x(equation_1d, grid=None, direction_only=False):
        """
        Convert a 1D equation to a 3D equation applied in x-direction
        
        Args:
            equation_1d: 1D equation instance
            grid: 3D Grid object
            direction_only: If True, only apply in x-direction
            
        Returns:
            3D equation instance for x-direction
        """
        return DirectionalEquation3D(equation_1d, 'x', direction_only, grid)
    
    @staticmethod
    def to_y(equation_1d, grid=None, direction_only=False):
        """
        Convert a 1D equation to a 3D equation applied in y-direction
        
        Args:
            equation_1d: 1D equation instance
            grid: 3D Grid object
            direction_only: If True, only apply in y-direction
            
        Returns:
            3D equation instance for y-direction
        """
        return DirectionalEquation3D(equation_1d, 'y', direction_only, grid)
    
    @staticmethod
    def to_z(equation_1d, grid=None, direction_only=False):
        """
        Convert a 1D equation to a 3D equation applied in z-direction
        
        Args:
            equation_1d: 1D equation instance
            grid: 3D Grid object
            direction_only: If True, only apply in z-direction
            
        Returns:
            3D equation instance for z-direction
        """
        return DirectionalEquation3D(equation_1d, 'z', direction_only, grid)
    
    @staticmethod
    def to_xyz(equation_1d_x, equation_1d_y=None, equation_1d_z=None, grid=None):
        """
        Convert 1D equations to a 3D equation applied in all directions
        
        Args:
            equation_1d_x: 1D equation for x-direction
            equation_1d_y: 1D equation for y-direction (default: same as x)
            equation_1d_z: 1D equation for z-direction (default: same as x)
            grid: 3D Grid object
            
        Returns:
            3D equation instance for all directions
        """
        if equation_1d_y is None:
            equation_1d_y = equation_1d_x
        if equation_1d_z is None:
            equation_1d_z = equation_1d_x
            
        x_eq = DirectionalEquation3D(equation_1d_x, 'x', False, grid)
        y_eq = DirectionalEquation3D(equation_1d_y, 'y', False, grid)
        z_eq = DirectionalEquation3D(equation_1d_z, 'z', False, grid)
        
        return CombinedDirectionalEquation3D(x_eq, y_eq, z_eq, grid)


class DirectionalEquation2D(Equation2D):
    """Adapter class that converts a 1D equation to a 2D directional equation"""
    
    def __init__(self, equation_1d, direction='x', direction_only=False, grid=None):
        """
        Initialize with 1D equation and direction
        
        Args:
            equation_1d: 1D equation instance
            direction: Direction to apply ('x' or 'y')
            direction_only: If True, only apply in specified direction
            grid: 2D Grid object
        """
        super().__init__(grid)
        self.equation_1d = equation_1d
        self.direction = direction
        self.direction_only = direction_only
        
        # Index mapping for unknown vector
        # 2D unknown order: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        if direction == 'x':
            # Map 1D [ψ, ψ', ψ'', ψ'''] to 2D [ψ, ψ_x, ψ_xx, ψ_xxx]
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        else:  # direction == 'y'
            # Map 1D [ψ, ψ', ψ'', ψ'''] to 2D [ψ, ψ_y, ψ_yy, ψ_yyy]
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
            
        # Set 1D grid if available
        if self.grid is not None and hasattr(equation_1d, 'set_grid'):
            self._set_1d_grid_to_equation()
            
    def _set_1d_grid_to_equation(self):
        """Set appropriate 1D grid for the 1D equation"""
        if self.grid is None:
            return
            
        # Create 1D grid emulator
        class Grid1DEmulator:
            def __init__(self, points, spacing, n_points):
                self.points = points
                self.h = spacing
                self.n_points = n_points
            
            def get_point(self, idx):
                return self.points[idx]
            
            def get_points(self):
                return self.points
            
            def get_spacing(self):
                return self.h
                
        # Create emulator for the appropriate direction
        if self.direction == 'x':
            emulated_grid = Grid1DEmulator(
                self.grid.x, 
                self.grid.get_spacing()[0], 
                self.grid.nx_points
            )
        else:  # self.direction == 'y'
            emulated_grid = Grid1DEmulator(
                self.grid.y, 
                self.grid.get_spacing()[1], 
                self.grid.ny_points
            )
            
        self.equation_1d.set_grid(emulated_grid)
        
    def set_grid(self, grid):
        """
        Set the grid for this equation and 1D equation
        
        Args:
            grid: 2D Grid object
            
        Returns:
            self: For method chaining
        """
        super().set_grid(grid)
        
        # Set 1D grid for the 1D equation
        self._set_1d_grid_to_equation()
        
        return self
        
    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # Get appropriate 1D index based on direction
        if self.direction == 'x':
            i_1d = i
            is_boundary = i == 0 or i == self.grid.nx_points - 1
        else:  # self.direction == 'y'
            i_1d = j
            is_boundary = j == 0 or j == self.grid.ny_points - 1
        
        # Skip processing for boundary-only equations
        if self.direction_only and is_boundary:
            return {}
        
        # Get 1D stencil coefficients
        coeffs_1d = self.equation_1d.get_stencil_coefficients(i_1d)
        
        # Convert to 2D coefficients
        coeffs_2d = {}
        for offset_1d, coeff_array_1d in coeffs_1d.items():
            # Convert offset to 2D
            if self.direction == 'x':
                offset_2d = (offset_1d, 0)
            else:  # self.direction == 'y'
                offset_2d = (0, offset_1d)
            
            # Initialize 2D coefficient array [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
            coeff_array_2d = np.zeros(7)
            
            # Map coefficients using the index map
            for idx_1d, idx_2d in self.index_map.items():
                if idx_1d < len(coeff_array_1d):
                    coeff_array_2d[idx_2d] = coeff_array_1d[idx_1d]
            
            coeffs_2d[offset_2d] = coeff_array_2d
            
        return coeffs_2d
        
    def is_valid_at(self, i=None, j=None):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # For direction-only equations, check boundary perpendicular to direction
        if self.direction_only:
            if self.direction == 'x':
                # Invalid at y boundaries
                if j == 0 or j == self.grid.ny_points - 1:
                    return False
            else:  # self.direction == 'y'
                # Invalid at x boundaries
                if i == 0 or i == self.grid.nx_points - 1:
                    return False
        
        # Check validity based on direction
        if self.direction == 'x':
            return self.equation_1d.is_valid_at(i)
        else:  # self.direction == 'y'
            return self.equation_1d.is_valid_at(j)


class CombinedDirectionalEquation2D(Equation2D):
    """Combines x and y directional equations into one 2D equation"""
    
    def __init__(self, x_direction_eq, y_direction_eq, grid=None):
        """
        Initialize with x and y direction equations
        
        Args:
            x_direction_eq: Equation for x-direction
            y_direction_eq: Equation for y-direction
            grid: 2D Grid object
        """
        # Use grid from equations if not provided
        if grid is None:
            if hasattr(x_direction_eq, 'grid') and x_direction_eq.grid is not None:
                grid = x_direction_eq.grid
            elif hasattr(y_direction_eq, 'grid') and y_direction_eq.grid is not None:
                grid = y_direction_eq.grid
                
        super().__init__(grid)
        self.x_eq = x_direction_eq
        self.y_eq = y_direction_eq
        
        # Set grid for component equations
        if self.grid is not None:
            if hasattr(x_direction_eq, 'set_grid'):
                x_direction_eq.set_grid(self.grid)
            if hasattr(y_direction_eq, 'set_grid'):
                y_direction_eq.set_grid(self.grid)
    
    def set_grid(self, grid):
        """
        Set the grid for this equation and component equations
        
        Args:
            grid: 2D Grid object
            
        Returns:
            self: For method chaining
        """
        super().set_grid(grid)
        
        # Set grid for component equations
        if hasattr(self.x_eq, 'set_grid'):
            self.x_eq.set_grid(grid)
        if hasattr(self.y_eq, 'set_grid'):
            self.y_eq.set_grid(grid)
            
        return self
        
    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Dictionary with stencil coefficients
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        # Get coefficients from both directions
        x_coeffs = self.x_eq.get_stencil_coefficients(i, j)
        y_coeffs = self.y_eq.get_stencil_coefficients(i, j)
        
        # Combine coefficients
        combined_coeffs = {}
        
        # Add x-direction coefficients
        for offset, coeff in x_coeffs.items():
            combined_coeffs[offset] = coeff.copy()
        
        # Add y-direction coefficients (merging with x if needed)
        for offset, coeff in y_coeffs.items():
            if offset in combined_coeffs:
                combined_coeffs[offset] += coeff
            else:
                combined_coeffs[offset] = coeff
        
        return combined_coeffs
        
    def is_valid_at(self, i=None, j=None):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        # Valid only if both component equations are valid
        return (self.x_eq.is_valid_at(i, j) and 
                self.y_eq.is_valid_at(i, j))


class DirectionalEquation3D(Equation3D):
    """Adapter class that converts a 1D equation to a 3D directional equation"""
    
    def __init__(self, equation_1d, direction='x', direction_only=False, grid=None):
        """
        Initialize with 1D equation and direction
        
        Args:
            equation_1d: 1D equation instance
            direction: Direction to apply ('x', 'y', or 'z')
            direction_only: If True, only apply in specified direction
            grid: 3D Grid object
        """
        super().__init__(grid)
        self.equation_1d = equation_1d
        self.direction = direction
        self.direction_only = direction_only
        
        # Index mapping for unknown vector
        # 3D unknown order: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        if direction == 'x':
            # Map 1D [ψ, ψ', ψ'', ψ'''] to 3D [ψ, ψ_x, ψ_xx, ψ_xxx]
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        elif direction == 'y':
            # Map 1D [ψ, ψ', ψ'', ψ'''] to 3D [ψ, ψ_y, ψ_yy, ψ_yyy]
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
        else:  # direction == 'z'
            # Map 1D [ψ, ψ', ψ'', ψ'''] to 3D [ψ, ψ_z, ψ_zz, ψ_zzz]
            self.index_map = {0: 0, 1: 7, 2: 8, 3: 9}
            
        # Set 1D grid if available
        if self.grid is not None and hasattr(equation_1d, 'set_grid'):
            self._set_1d_grid_to_equation()
            
    def _set_1d_grid_to_equation(self):
        """Set appropriate 1D grid for the 1D equation"""
        if self.grid is None:
            return
            
        # Create 1D grid emulator
        class Grid1DEmulator:
            def __init__(self, points, spacing, n_points):
                self.points = points
                self.h = spacing
                self.n_points = n_points
            
            def get_point(self, idx):
                return self.points[idx]
            
            def get_points(self):
                return self.points
            
            def get_spacing(self):
                return self.h
                
        # Create emulator for the appropriate direction
        if self.direction == 'x':
            emulated_grid = Grid1DEmulator(
                self.grid.x, 
                self.grid.get_spacing()[0], 
                self.grid.nx_points
            )
        elif self.direction == 'y':
            emulated_grid = Grid1DEmulator(
                self.grid.y, 
                self.grid.get_spacing()[1], 
                self.grid.ny_points
            )
        else:  # self.direction == 'z'
            emulated_grid = Grid1DEmulator(
                self.grid.z, 
                self.grid.get_spacing()[2], 
                self.grid.nz_points
            )
            
        self.equation_1d.set_grid(emulated_grid)
        
    def set_grid(self, grid):
        """
        Set the grid for this equation and 1D equation
        
        Args:
            grid: 3D Grid object
            
        Returns:
            self: For method chaining
        """
        super().set_grid(grid)
        
        # Set 1D grid for the 1D equation
        self._set_1d_grid_to_equation()
        
        return self
        
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
            
        if i is None or j is None or k is None:
            raise ValueError("Grid indices i, j, and k must be specified.")
        
        # Get appropriate 1D index based on direction
        if self.direction == 'x':
            i_1d = i
            is_boundary = i == 0 or i == self.grid.nx_points - 1
        elif self.direction == 'y':
            i_1d = j
            is_boundary = j == 0 or j == self.grid.ny_points - 1
        else:  # self.direction == 'z'
            i_1d = k
            is_boundary = k == 0 or k == self.grid.nz_points - 1
        
        # Skip processing for boundary-only equations
        if self.direction_only and is_boundary:
            return {}
        
        # Get 1D stencil coefficients
        coeffs_1d = self.equation_1d.get_stencil_coefficients(i_1d)
        
        # Convert to 3D coefficients
        coeffs_3d = {}
        for offset_1d, coeff_array_1d in coeffs_1d.items():
            # Convert offset to 3D
            if self.direction == 'x':
                offset_3d = (offset_1d, 0, 0)
            elif self.direction == 'y':
                offset_3d = (0, offset_1d, 0)
            else:  # self.direction == 'z'
                offset_3d = (0, 0, offset_1d)
            
            # Initialize 3D coefficient array
            # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
            coeff_array_3d = np.zeros(10)
            
            # Map coefficients using the index map
            for idx_1d, idx_3d in self.index_map.items():
                if idx_1d < len(coeff_array_1d):
                    coeff_array_3d[idx_3d] = coeff_array_1d[idx_1d]
            
            coeffs_3d[offset_3d] = coeff_array_3d
            
        return coeffs_3d
        
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
        
        # For direction-only equations, check boundaries perpendicular to direction
        if self.direction_only:
            if self.direction == 'x':
                # Invalid at y and z boundaries
                if (j == 0 or j == self.grid.ny_points - 1 or
                    k == 0 or k == self.grid.nz_points - 1):
                    return False
            elif self.direction == 'y':
                # Invalid at x and z boundaries
                if (i == 0 or i == self.grid.nx_points - 1 or
                    k == 0 or k == self.grid.nz_points - 1):
                    return False
            else:  # self.direction == 'z'
                # Invalid at x and y boundaries
                if (i == 0 or i == self.grid.nx_points - 1 or
                    j == 0 or j == self.grid.ny_points - 1):
                    return False
        
        # Check validity based on direction
        if self.direction == 'x':
            return self.equation_1d.is_valid_at(i)
        elif self.direction == 'y':
            return self.equation_1d.is_valid_at(j)
        else:  # self.direction == 'z'
            return self.equation_1d.is_valid_at(k)


class CombinedDirectionalEquation3D(Equation3D):
    """Combines x, y, and z directional equations into one 3D equation"""
    
    def __init__(self, x_direction_eq, y_direction_eq, z_direction_eq, grid=None):
        """
        Initialize with x, y, and z direction equations
        
        Args:
            x_direction_eq: Equation for x-direction
            y_direction_eq: Equation for y-direction
            z_direction_eq: Equation for z-direction
            grid: 3D Grid object
        """
        # Use grid from equations if not provided
        if grid is None:
            if hasattr(x_direction_eq, 'grid') and x_direction_eq.grid is not None:
                grid = x_direction_eq.grid
            elif hasattr(y_direction_eq, 'grid') and y_direction_eq.grid is not None:
                grid = y_direction_eq.grid
            elif hasattr(z_direction_eq, 'grid') and z_direction_eq.grid is not None:
                grid = z_direction_eq.grid
                
        super().__init__(grid)
        self.x_eq = x_direction_eq
        self.y_eq = y_direction_eq
        self.z_eq = z_direction_eq
        
        # Set grid for component equations
        if self.grid is not None:
            if hasattr(x_direction_eq, 'set_grid'):
                x_direction_eq.set_grid(self.grid)
            if hasattr(y_direction_eq, 'set_grid'):
                y_direction_eq.set_grid(self.grid)
            if hasattr(z_direction_eq, 'set_grid'):
                z_direction_eq.set_grid(self.grid)
    
    def set_grid(self, grid):
        """
        Set the grid for this equation and component equations
        
        Args:
            grid: 3D Grid object
            
        Returns:
            self: For method chaining
        """
        super().set_grid(grid)
        
        # Set grid for component equations
        if hasattr(self.x_eq, 'set_grid'):
            self.x_eq.set_grid(grid)
        if hasattr(self.y_eq, 'set_grid'):
            self.y_eq.set_grid(grid)
        if hasattr(self.z_eq, 'set_grid'):
            self.z_eq.set_grid(grid)
            
        return self
        
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
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        # Get coefficients from all directions
        x_coeffs = self.x_eq.get_stencil_coefficients(i, j, k)
        y_coeffs = self.y_eq.get_stencil_coefficients(i, j, k)
        z_coeffs = self.z_eq.get_stencil_coefficients(i, j, k)
        
        # Combine coefficients
        combined_coeffs = {}
        
        # Add x-direction coefficients
        for offset, coeff in x_coeffs.items():
            combined_coeffs[offset] = coeff.copy()
        
        # Add y-direction coefficients (merging with x if needed)
        for offset, coeff in y_coeffs.items():
            if offset in combined_coeffs:
                combined_coeffs[offset] += coeff
            else:
                combined_coeffs[offset] = coeff
        
        # Add z-direction coefficients (merging with existing if needed)
        for offset, coeff in z_coeffs.items():
            if offset in combined_coeffs:
                combined_coeffs[offset] += coeff
            else:
                combined_coeffs[offset] = coeff
        
        return combined_coeffs
        
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
        
        # Valid only if all component equations are valid
        return (self.x_eq.is_valid_at(i, j, k) and 
                self.y_eq.is_valid_at(i, j, k) and
                self.z_eq.is_valid_at(i, j, k))