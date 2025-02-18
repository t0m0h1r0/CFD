from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import CompactDifferenceBase
from ...common.types import BoundaryCondition, BCType
from ...common.grid import GridManager

class CombinedCompactDifference(CompactDifferenceBase):
    """Combined Compact Difference (CCD) Scheme Implementation"""

    def __init__(self,
                 grid_manager: GridManager,
                 boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
                 order: int = 6):
        """Initialize CCD Scheme"""
        coefficients = self._derive_theoretical_coefficients(order)
        super().__init__(grid_manager, boundary_conditions, coefficients)
        self.order = order

    def _derive_theoretical_coefficients(self, order: int) -> Dict[str, float]:
        """Derive theoretical coefficients"""
        if order == 6:
            return {
                'alpha': 6/11,
                'beta': -4/11,
                'gamma': 1/11,
                'delta': 3/4,
                'epsilon': -3/8,
                'zeta': 1/8
            }
        else:
            raise NotImplementedError(f"Order {order} not supported")

    def _arrange_field_for_direction(self, 
                                   field: ArrayLike, 
                                   direction: str) -> ArrayLike:
        """Arrange field for the specified direction"""
        if direction == 'y':
            # Transpose for y-direction differentiation
            return jnp.transpose(field)
        return field

    def _restore_field_arrangement(self,
                                 field: ArrayLike,
                                 direction: str) -> ArrayLike:
        """Restore original field arrangement"""
        if direction == 'y':
            return jnp.transpose(field)
        return field

    def discretize(self, 
                   field: ArrayLike, 
                   direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute spatial derivatives
        
        Args:
            field: Input field
            direction: Differentiation direction ('x' or 'y')
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # Arrange field for the specified direction
        arranged_field = self._arrange_field_for_direction(field, direction)
        
        # Get grid spacing
        dx = self.grid_manager.get_grid_spacing(direction)
        if hasattr(dx, '__len__'):
            dx = dx[0]  # Use first value for non-uniform grids
            
        # Compute derivatives
        first_derivative = self._compact_first_derivative(arranged_field, dx)
        second_derivative = self._compact_second_derivative(arranged_field, dx)
        
        # Apply boundary conditions
        first_derivative, second_derivative = self.apply_boundary_conditions(
            arranged_field, (first_derivative, second_derivative), direction
        )
        
        # Restore original arrangement
        first_derivative = self._restore_field_arrangement(first_derivative, direction)
        second_derivative = self._restore_field_arrangement(second_derivative, direction)
        
        return first_derivative, second_derivative

    def _compact_first_derivative(self, 
                                field: ArrayLike, 
                                dx: float) -> ArrayLike:
        """Compute first derivative"""
        n = len(field)
        first_deriv = jnp.zeros_like(field)
        
        # Interior points
        first_deriv = first_deriv.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * dx)
        )
        
        # Boundary points
        first_deriv = first_deriv.at[0].set(
            (-3*field[0] + 4*field[1] - field[2]) / (2 * dx)  # Forward difference
        )
        first_deriv = first_deriv.at[-1].set(
            (field[-3] - 4*field[-2] + 3*field[-1]) / (2 * dx)  # Backward difference
        )
        
        return first_deriv

    def _compact_second_derivative(self, 
                                 field: ArrayLike, 
                                 dx: float) -> ArrayLike:
        """Compute second derivative"""
        n = len(field)
        second_deriv = jnp.zeros_like(field)
        
        # Interior points
        second_deriv = second_deriv.at[1:-1].set(
            (field[2:] - 2 * field[1:-1] + field[:-2]) / (dx**2)
        )
        
        # Boundary points - One-sided second order approximation
        second_deriv = second_deriv.at[0].set(
            (2*field[0] - 5*field[1] + 4*field[2] - field[3]) / (dx**2)
        )
        second_deriv = second_deriv.at[-1].set(
            (2*field[-1] - 5*field[-2] + 4*field[-3] - field[-4]) / (dx**2)
        )
        
        return second_deriv

    def apply_boundary_conditions(self,
                                field: ArrayLike,
                                derivatives: Tuple[ArrayLike, ArrayLike],
                                direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """Apply boundary conditions"""
        first_deriv, second_deriv = derivatives
        
        # Get boundary conditions for the specified direction
        if direction == 'x':
            bc_start = self.boundary_conditions.get('left')
            bc_end = self.boundary_conditions.get('right')
        else:  # direction == 'y'
            bc_start = self.boundary_conditions.get('bottom')
            bc_end = self.boundary_conditions.get('top')
            
        # Apply boundary conditions
        dx = self.grid_manager.get_grid_spacing(direction)
        if hasattr(dx, '__len__'):
            dx = dx[0]
            
        # Start boundary
        if bc_start and bc_start.type == BCType.DIRICHLET:
            first_deriv = first_deriv.at[0].set(
                (-3*field[0] + 4*field[1] - field[2]) / (2 * dx)
            )
            second_deriv = second_deriv.at[0].set(
                (2*field[0] - 5*field[1] + 4*field[2] - field[3]) / (dx**2)
            )
            
        # End boundary
        if bc_end and bc_end.type == BCType.DIRICHLET:
            first_deriv = first_deriv.at[-1].set(
                (field[-3] - 4*field[-2] + 3*field[-1]) / (2 * dx)
            )
            second_deriv = second_deriv.at[-1].set(
                (2*field[-1] - 5*field[-2] + 4*field[-3] - field[-4]) / (dx**2)
            )
            
        return first_deriv, second_deriv

    def solve_system(self,
                    lhs: ArrayLike, 
                    rhs: ArrayLike, 
                    field: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Solve the discretization system for derivatives
        
        Args:
            lhs: Left-hand side matrix
            rhs: Right-hand side matrix
            field: Input field
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # Get field shape and size
        n = field.shape[0]
        
        # Initialize solution arrays
        first_deriv = jnp.zeros_like(field)
        second_deriv = jnp.zeros_like(field)
        
        # Solve system using matrix operations
        if lhs.shape[0] > 0:  # If we have a non-empty system
            # System solution using JAX's linear algebra solver
            solution = jnp.linalg.solve(lhs, rhs @ field)
            
            # Extract derivatives from solution
            # Even indices correspond to first derivatives
            # Odd indices correspond to second derivatives
            first_deriv = first_deriv.at[:].set(solution[::2])
            second_deriv = second_deriv.at[:].set(solution[1::2])
        
        return first_deriv, second_deriv    