from abc import ABC, abstractmethod
from typing import Optional, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from ..common.types import BoundaryCondition
from ..common.grid import GridManager

class SpatialDiscretizationBase(ABC):
    """Base class for spatial discretization schemes."""
    
    def __init__(self, 
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None):
        """
        Initialize spatial discretization scheme.
        
        Args:
            grid_manager: Grid management object
            boundary_conditions: Dictionary of boundary conditions for each boundary
        """
        self.grid_manager = grid_manager
        self.boundary_conditions = boundary_conditions or {}
        
    @abstractmethod
    def discretize(self, 
                  field: ArrayLike,
                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute spatial derivatives of the field.
        
        Args:
            field: Input field to differentiate
            direction: Direction of differentiation ('x', 'y', or 'z')
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        pass
    
    @abstractmethod
    def apply_boundary_conditions(self,
                                field: ArrayLike,
                                derivatives: Tuple[ArrayLike, ArrayLike],
                                direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Apply boundary conditions to the computed derivatives.
        
        Args:
            field: Input field
            derivatives: Tuple of (first_derivative, second_derivative)
            direction: Direction of differentiation
            
        Returns:
            Tuple of corrected (first_derivative, second_derivative)
        """
        pass

class CompactDifferenceBase(SpatialDiscretizationBase):
    """Base class for compact difference schemes."""
    
    def __init__(self,
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None,
                 coefficients: Optional[dict] = None):
        """
        Initialize compact difference scheme.
        
        Args:
            grid_manager: Grid management object
            boundary_conditions: Dictionary of boundary conditions
            coefficients: Dictionary of difference coefficients
        """
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = coefficients or {}
        
    def build_coefficient_matrices(self, 
                                 direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Build coefficient matrices for the compact scheme.
        
        Args:
            direction: Direction for which to build matrices
            
        Returns:
            Tuple of (lhs_matrix, rhs_matrix)
        """
        dx = self.grid_manager.get_grid_spacing(direction)
        n_points = self.grid_manager.get_grid_points(direction)
        
        # Initialize matrices
        lhs = jnp.zeros((2*n_points, 2*n_points))
        rhs = jnp.zeros((2*n_points, n_points))
        
        return lhs, rhs
    
    @abstractmethod
    def solve_system(self,
                    lhs: ArrayLike,
                    rhs: ArrayLike,
                    field: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Solve the compact difference system.
        
        Args:
            lhs: Left-hand side matrix
            rhs: Right-hand side matrix
            field: Input field
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        pass