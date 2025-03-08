from abc import ABC, abstractmethod
import cupy as cp

class Equation2D(ABC):
    """Base class for 2D difference equations"""
    
    @abstractmethod
    def get_stencil_coefficients(self, grid, i, j):
        """
        Get stencil coefficients at grid point (i,j)
        
        Args:
            grid: Grid2D object
            i, j: Grid indices
            
        Returns:
            Dictionary with keys as (di, dj) offsets and values as coefficient arrays
            Each coefficient array has 7 elements for [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        """
        pass
    
    @abstractmethod
    def get_rhs(self, grid, i, j):
        """
        Get right-hand side value at grid point (i,j)
        
        Args:
            grid: Grid2D object
            i, j: Grid indices
            
        Returns:
            Right-hand side value at (i,j)
        """
        pass
    
    @abstractmethod
    def is_valid_at(self, grid, i, j):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            grid: Grid2D object
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
    
    def __init__(self, eq1, eq2, operation="+"):
        """
        Initialize with two equations and operation
        
        Args:
            eq1, eq2: Equation2D objects
            operation: "+" or "-"
        """
        self.eq1 = eq1
        self.eq2 = eq2
        self.operation = operation
    
    def get_stencil_coefficients(self, grid, i, j):
        """Combine stencil coefficients from both equations"""
        coeffs1 = self.eq1.get_stencil_coefficients(grid, i, j)
        coeffs2 = self.eq2.get_stencil_coefficients(grid, i, j)
        
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
    
    def get_rhs(self, grid, i, j):
        """Combine right-hand sides from both equations"""
        rhs1 = self.eq1.get_rhs(grid, i, j)
        rhs2 = self.eq2.get_rhs(grid, i, j)
        
        if self.operation == "+":
            return rhs1 + rhs2
        else:
            return rhs1 - rhs2
    
    def is_valid_at(self, grid, i, j):
        """Valid where both equations are valid"""
        return self.eq1.is_valid_at(grid, i, j) and self.eq2.is_valid_at(grid, i, j)


class ScaledEquation2D(Equation2D):
    """Equation scaled by a constant"""
    
    def __init__(self, equation, scalar):
        """
        Initialize with equation and scaling factor
        
        Args:
            equation: Equation2D object
            scalar: Scaling factor
        """
        self.equation = equation
        self.scalar = scalar
    
    def get_stencil_coefficients(self, grid, i, j):
        """Scale all coefficients"""
        coeffs = self.equation.get_stencil_coefficients(grid, i, j)
        return {offset: self.scalar * coeff for offset, coeff in coeffs.items()}
    
    def get_rhs(self, grid, i, j):
        """Scale right-hand side"""
        return self.scalar * self.equation.get_rhs(grid, i, j)
    
    def is_valid_at(self, grid, i, j):
        """Valid where the original equation is valid"""
        return self.equation.is_valid_at(grid, i, j)