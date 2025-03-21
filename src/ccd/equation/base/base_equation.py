"""
Base equation classes for the CCD method.

This module provides the abstract base classes for equations used in the
Combined Compact Difference (CCD) method across all dimensions.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple


class BaseEquation(ABC):
    """Base abstract class for all equations regardless of dimension"""
    
    def __init__(self, grid=None):
        """
        Initialize with optional grid
        
        Args:
            grid: Grid object
        """
        self.grid = grid
    
    def set_grid(self, grid):
        """
        Set the grid for this equation
        
        Args:
            grid: Grid object
            
        Returns:
            self: For method chaining
        """
        self.grid = grid
        return self
    
    def __add__(self, other):
        """Add two equations"""
        return CombinedEquation(self, other, "+")
    
    def __sub__(self, other):
        """Subtract two equations"""
        return CombinedEquation(self, other, "-")
    
    def __mul__(self, scalar):
        """Multiply equation by scalar"""
        return ScaledEquation(self, scalar)
    
    __rmul__ = __mul__


class Equation1D(BaseEquation):
    """Base class for 1D difference equations"""
    
    @abstractmethod
    def get_stencil_coefficients(self, i=None):
        """
        Get stencil coefficients at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Dictionary with keys as offset integers and values as coefficient arrays
            Each coefficient array has 4 elements for [ψ, ψ', ψ'', ψ''']
        """
        pass
    
    @abstractmethod
    def is_valid_at(self, i=None):
        """
        Check if equation is valid at grid point i
        
        Args:
            i: Grid index
            
        Returns:
            Boolean indicating if equation is valid at i
        """
        pass


class Equation2D(BaseEquation):
    """Base class for 2D difference equations"""
    
    @abstractmethod
    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        Args:
            i, j: Grid indices
            
        Returns:
            Dictionary with keys as (di, dj) offsets and values as coefficient arrays
            Each coefficient array has 7 elements for [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        """
        pass
    
    @abstractmethod
    def is_valid_at(self, i=None, j=None):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            i, j: Grid indices
            
        Returns:
            Boolean indicating if equation is valid at (i,j)
        """
        pass


class Equation3D(BaseEquation):
    """Base class for 3D difference equations"""
    
    @abstractmethod
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        Get stencil coefficients at grid point (i,j,k)
        
        Args:
            i, j, k: Grid indices
            
        Returns:
            Dictionary with keys as (di, dj, dk) offsets and values as coefficient arrays
            Each coefficient array has 10 elements for 
            [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        """
        pass
    
    @abstractmethod
    def is_valid_at(self, i=None, j=None, k=None):
        """
        Check if equation is valid at grid point (i,j,k)
        
        Args:
            i, j, k: Grid indices
            
        Returns:
            Boolean indicating if equation is valid at (i,j,k)
        """
        pass


class CombinedEquation(BaseEquation):
    """Combination of two equations of the same dimension"""
    
    def __init__(self, eq1, eq2, operation="+", grid=None):
        """
        Initialize with two equations and operation
        
        Args:
            eq1, eq2: Equation objects of the same dimension
            operation: "+" or "-"
            grid: Grid object (optional)
        """
        # If grid is not provided, try to use grid from eq1 or eq2
        if grid is None:
            if hasattr(eq1, 'grid') and eq1.grid is not None:
                grid = eq1.grid
            elif hasattr(eq2, 'grid') and eq2.grid is not None:
                grid = eq2.grid
                
        super().__init__(grid)
        self.eq1 = eq1
        self.eq2 = eq2
        self.operation = operation
        
        # Set grid to both sub-equations if we have a grid
        if self.grid is not None:
            if hasattr(eq1, 'set_grid'):
                eq1.set_grid(self.grid)
            if hasattr(eq2, 'set_grid'):
                eq2.set_grid(self.grid)
    
    def set_grid(self, grid):
        """
        Set the grid for this equation and its sub-equations
        
        Args:
            grid: Grid object
            
        Returns:
            self: For method chaining
        """
        super().set_grid(grid)
        
        # Set grid to both sub-equations
        if hasattr(self.eq1, 'set_grid'):
            self.eq1.set_grid(grid)
        if hasattr(self.eq2, 'set_grid'):
            self.eq2.set_grid(grid)
            
        return self
    
    def get_stencil_coefficients(self, *args):
        """
        Combine stencil coefficients from both equations
        
        Args:
            *args: Grid indices appropriate for the dimension
            
        Returns:
            Combined stencil coefficients
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        # Get coefficients from both equations
        coeffs1 = self.eq1.get_stencil_coefficients(*args)
        coeffs2 = self.eq2.get_stencil_coefficients(*args)
        
        # Combine coefficients
        combined_coeffs = {}
        all_offsets = set(list(coeffs1.keys()) + list(coeffs2.keys()))
        
        for offset in all_offsets:
            if offset in coeffs1:
                coeff1 = coeffs1[offset]
            else:
                coeff1 = np.zeros_like(next(iter(coeffs2.values())))
                
            if offset in coeffs2:
                coeff2 = coeffs2[offset]
            else:
                coeff2 = np.zeros_like(next(iter(coeffs1.values())))
            
            if self.operation == "+":
                combined_coeffs[offset] = coeff1 + coeff2
            else:
                combined_coeffs[offset] = coeff1 - coeff2
        
        return combined_coeffs
    
    def is_valid_at(self, *args):
        """
        Valid where both equations are valid
        
        Args:
            *args: Grid indices appropriate for the dimension
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        # Valid only if both equations are valid
        return self.eq1.is_valid_at(*args) and self.eq2.is_valid_at(*args)


class ScaledEquation(BaseEquation):
    """Equation scaled by a constant"""
    
    def __init__(self, equation, scalar, grid=None):
        """
        Initialize with equation and scaling factor
        
        Args:
            equation: Equation object
            scalar: Scaling factor
            grid: Grid object (optional)
        """
        # If grid is not provided, try to use grid from equation
        if grid is None and hasattr(equation, 'grid'):
            grid = equation.grid
            
        super().__init__(grid)
        self.equation = equation
        self.scalar = scalar
        
        # Set grid to sub-equation if we have a grid
        if self.grid is not None and hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
    
    def set_grid(self, grid):
        """
        Set the grid for this equation and its sub-equation
        
        Args:
            grid: Grid object
            
        Returns:
            self: For method chaining
        """
        super().set_grid(grid)
        
        # Set grid to sub-equation
        if hasattr(self.equation, 'set_grid'):
            self.equation.set_grid(grid)
            
        return self
    
    def get_stencil_coefficients(self, *args):
        """
        Scale all coefficients from sub-equation
        
        Args:
            *args: Grid indices appropriate for the dimension
            
        Returns:
            Scaled stencil coefficients
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        # Get coefficients from sub-equation and scale them
        coeffs = self.equation.get_stencil_coefficients(*args)
        return {offset: self.scalar * coeff for offset, coeff in coeffs.items()}
    
    def is_valid_at(self, *args):
        """
        Valid where the original equation is valid
        
        Args:
            *args: Grid indices appropriate for the dimension
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        # Valid only if sub-equation is valid
        return self.equation.is_valid_at(*args)
