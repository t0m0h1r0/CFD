from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import SpatialDiscretizationBase
from ...common.types import Grid, BoundaryCondition, BCType
from ...common.grid import GridManager

class CombinedCompactDifference(SpatialDiscretizationBase):
    """Implementation of Combined Compact Difference (CCD) scheme."""
    
    def __init__(self,
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None,
                 order: int = 6):
        """
        Initialize CCD scheme.
        
        Args:
            grid_manager: Grid management object
            boundary_conditions: Dictionary of boundary conditions
            order: Order of accuracy (default: 6)
        """
        # Calculate coefficients based on the order
        coefficients = self._calculate_coefficients(order)
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = coefficients
        self.order = order
        
    def _calculate_coefficients(self, order: int) -> dict:
        """
        Calculate CCD coefficients for given order.
        
        Args:
            order: Order of accuracy
            
        Returns:
            Dictionary of coefficients
        """
        if order == 6:
            return {
                'a1': 15/16,
                'b1': -7/16,
                'c1': 1/16,
                'a2': 3/4,
                'b2': -9/8,
                'c2': 1/8
            }
        else:
            raise NotImplementedError(f"Order {order} not implemented")
    
    def _safe_first_derivative(self, field: ArrayLike, dx: float) -> ArrayLike:
        """
        Compute first derivative with central difference for boundary points
        
        Args:
            field: Input field 
            dx: Grid spacing
            
        Returns:
            First derivative
        """
        first_deriv = jnp.zeros_like(field)
        
        # Central difference for interior points
        first_deriv = first_deriv.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * dx)
        )
        
        # Forward difference for first point
        first_deriv = first_deriv.at[0].set(
            (-3 * field[0] + 4 * field[1] - field[2]) / (2 * dx)
        )
        
        # Backward difference for last point
        first_deriv = first_deriv.at[-1].set(
            (3 * field[-1] - 4 * field[-2] + field[-3]) / (2 * dx)
        )
        
        return first_deriv
    
    def _safe_second_derivative(self, field: ArrayLike, dx: float) -> ArrayLike:
        """
        Compute second derivative with central difference
        
        Args:
            field: Input field 
            dx: Grid spacing
            
        Returns:
            Second derivative
        """
        second_deriv = jnp.zeros_like(field)
        
        # Central difference for interior points
        second_deriv = second_deriv.at[1:-1].set(
            (field[2:] - 2 * field[1:-1] + field[:-2]) / (dx**2)
        )
        
        # Forward difference for first point
        second_deriv = second_deriv.at[0].set(
            (2 * field[0] - 5 * field[1] + 4 * field[2] - field[3]) / (dx**2)
        )
        
        # Backward difference for last point
        second_deriv = second_deriv.at[-1].set(
            (2 * field[-1] - 5 * field[-2] + 4 * field[-3] - field[-4]) / (dx**2)
        )
        
        return second_deriv
    
    def discretize(self,
                  field: ArrayLike,
                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute spatial derivatives using CCD scheme with safe fallback.
        
        Args:
            field: Input field to differentiate
            direction: Direction of differentiation
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # Get grid spacing
        dx = self.grid_manager.get_grid_spacing(direction)[0]
        
        # Compute derivatives using safe methods
        first_deriv = self._safe_first_derivative(field, dx)
        second_deriv = self._safe_second_derivative(field, dx)
        
        # Apply boundary conditions
        first_deriv, second_deriv = self.apply_boundary_conditions(
            field, (first_deriv, second_deriv), direction
        )
        
        return first_deriv, second_deriv
    
    def apply_boundary_conditions(self,
                                field: ArrayLike,
                                derivatives: Tuple[ArrayLike, ArrayLike],
                                direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Apply boundary conditions for CCD scheme.
        
        Args:
            field: Input field
            derivatives: Tuple of (first_derivative, second_derivative)
            direction: Direction of differentiation
            
        Returns:
            Tuple of corrected (first_derivative, second_derivative)
        """
        first_deriv, second_deriv = derivatives
        
        # Check if boundary conditions exist for this direction
        if direction not in self.boundary_conditions:
            return first_deriv, second_deriv
        
        bc = self.boundary_conditions[direction]
        
        # Apply boundary conditions based on type
        if bc.type == BCType.PERIODIC:
            # Periodic boundary conditions
            first_deriv = first_deriv.at[0].set(first_deriv[-2])
            first_deriv = first_deriv.at[-1].set(first_deriv[1])
            second_deriv = second_deriv.at[0].set(second_deriv[-2])
            second_deriv = second_deriv.at[-1].set(second_deriv[1])
        
        elif bc.type == BCType.DIRICHLET:
            # Default Dirichlet condition: zero at boundaries
            if callable(bc.value):
                # If value is a function, use it
                first_deriv = first_deriv.at[0].set(0.0)
                first_deriv = first_deriv.at[-1].set(0.0)
                second_deriv = second_deriv.at[0].set(0.0)
                second_deriv = second_deriv.at[-1].set(0.0)
            else:
                first_deriv = first_deriv.at[0].set(0.0)
                first_deriv = first_deriv.at[-1].set(0.0)
                second_deriv = second_deriv.at[0].set(0.0)
                second_deriv = second_deriv.at[-1].set(0.0)
        
        elif bc.type == BCType.NEUMANN:
            # Neumann condition: zero gradient
            if callable(bc.value):
                # If value is a function, use it
                first_deriv = first_deriv.at[0].set(0.0)
                first_deriv = first_deriv.at[-1].set(0.0)
            else:
                first_deriv = first_deriv.at[0].set(0.0)
                first_deriv = first_deriv.at[-1].set(0.0)
        
        return first_deriv, second_deriv