from abc import ABC, abstractmethod
import cupy as cp

class Equation2D(ABC):
    """Base class for 2D difference equations"""
    
    def __init__(self, grid=None):
        """
        Initialize with optional grid
        
        Args:
            grid: Grid2D object
        """
        self.grid = grid
    
    def set_grid(self, grid):
        """
        Set the grid for this equation
        
        Args:
            grid: Grid2D object
            
        Returns:
            self: For method chaining
        """
        self.grid = grid
        return self
    
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
    def get_rhs(self, i=None, j=None):
        """
        Get right-hand side value at grid point (i,j)
        
        Args:
            i, j: Grid indices
            
        Returns:
            Right-hand side value at (i,j)
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
    
    def __add__(self, other):
        """Add two equations"""
        return CombinedEquation2D(self, other, "+")
    
    def __sub__(self, other):
        """Subtract two equations"""
        return CombinedEquation2D(self, other, "-")
    
    def __mul__(self, scalar):
        """Multiply equation by scalar"""
        return ScaledEquation2D(self, scalar)
    
    __rmul__ = __mul__


class CombinedEquation2D(Equation2D):
    """Combination of two 2D equations"""
    
    def __init__(self, eq1, eq2, operation="+", grid=None):
        """
        Initialize with two equations and operation
        
        Args:
            eq1, eq2: Equation2D objects
            operation: "+" or "-"
            grid: Grid2D object (optional)
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
            grid: Grid2D object
            
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
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        Combine stencil coefficients from both equations
        
        Args:
            i, j: Grid indices
            
        Returns:
            Combined stencil coefficients
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # Get coefficients from both equations
        coeffs1 = self.eq1.get_stencil_coefficients(i, j)
        coeffs2 = self.eq2.get_stencil_coefficients(i, j)
        
        # Combine coefficients
        combined_coeffs = {}
        all_offsets = set(list(coeffs1.keys()) + list(coeffs2.keys()))
        
        for offset in all_offsets:
            coeff1 = coeffs1.get(offset, cp.zeros(7))
            coeff2 = coeffs2.get(offset, cp.zeros(7))
            
            if self.operation == "+":
                combined_coeffs[offset] = coeff1 + coeff2
            else:
                combined_coeffs[offset] = coeff1 - coeff2
        
        return combined_coeffs
    
    def get_rhs(self, i=None, j=None):
        """
        Combine right-hand sides from both equations
        
        Args:
            i, j: Grid indices
            
        Returns:
            Combined right-hand side
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # Get RHS from both equations
        rhs1 = self.eq1.get_rhs(i, j)
        rhs2 = self.eq2.get_rhs(i, j)
        
        # Combine according to operation
        if self.operation == "+":
            return rhs1 + rhs2
        else:
            return rhs1 - rhs2
    
    def is_valid_at(self, i=None, j=None):
        """
        Valid where both equations are valid
        
        Args:
            i, j: Grid indices
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # Valid only if both equations are valid
        return self.eq1.is_valid_at(i, j) and self.eq2.is_valid_at(i, j)


class ScaledEquation2D(Equation2D):
    """Equation scaled by a constant"""
    
    def __init__(self, equation, scalar, grid=None):
        """
        Initialize with equation and scaling factor
        
        Args:
            equation: Equation2D object
            scalar: Scaling factor
            grid: Grid2D object (optional)
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
            grid: Grid2D object
            
        Returns:
            self: For method chaining
        """
        super().set_grid(grid)
        
        # Set grid to sub-equation
        if hasattr(self.equation, 'set_grid'):
            self.equation.set_grid(grid)
            
        return self
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        Scale all coefficients from sub-equation
        
        Args:
            i, j: Grid indices
            
        Returns:
            Scaled stencil coefficients
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # Get coefficients from sub-equation and scale them
        coeffs = self.equation.get_stencil_coefficients(i, j)
        return {offset: self.scalar * coeff for offset, coeff in coeffs.items()}
    
    def get_rhs(self, i=None, j=None):
        """
        Scale right-hand side from sub-equation
        
        Args:
            i, j: Grid indices
            
        Returns:
            Scaled right-hand side
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # Get RHS from sub-equation and scale it
        return self.scalar * self.equation.get_rhs(i, j)
    
    def is_valid_at(self, i=None, j=None):
        """
        Valid where the original equation is valid
        
        Args:
            i, j: Grid indices
            
        Returns:
            Boolean validity
        """
        if self.grid is None:
            raise ValueError("Grid is not set. Use set_grid() before calling this method.")
        
        if i is None or j is None:
            raise ValueError("Grid indices i and j must be specified.")
        
        # Valid only if sub-equation is valid
        return self.equation.is_valid_at(i, j)