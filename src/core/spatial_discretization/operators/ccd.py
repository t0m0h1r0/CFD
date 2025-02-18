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
            
    def discretize(self,
                  field: ArrayLike,
                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute spatial derivatives using CCD scheme.
        
        Args:
            field: Input field to differentiate
            direction: Direction of differentiation
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # Extract coefficients
        a1, b1, c1 = (self.coefficients[k] for k in ['a1', 'b1', 'c1'])
        a2, b2, c2 = (self.coefficients[k] for k in ['a2', 'b2', 'c2'])
        
        # Get grid spacing
        dx = self.grid_manager.get_grid_spacing(direction)
        
        # First derivative computation
        first_deriv = jnp.zeros_like(field)
        for i in range(1, len(field)-1):
            first_deriv = first_deriv.at[i].set(
                (a1 * (field[i+1] - field[i-1]) / (2 * dx)) +  # Central difference
                b1 * (first_deriv[i+1] + first_deriv[i-1]) +  # Ghost point correction
                c1 * (first_deriv[i+1] - first_deriv[i-1]) / dx  # Cross terms
            )
        
        # Second derivative computation
        second_deriv = jnp.zeros_like(field)
        for i in range(1, len(field)-1):
            second_deriv = second_deriv.at[i].set(
                (a2 * (field[i+1] - 2*field[i] + field[i-1]) / (dx**2)) +  # Central difference
                b2 * (first_deriv[i+1] - first_deriv[i-1]) / dx +  # Cross terms
                c2 * (second_deriv[i+1] + second_deriv[i-1])  # Ghost point correction
            )
        
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
            # Dirichlet boundary conditions (constant value)
            if callable(bc.value):
                # If value is a function, use it at boundaries
                bc_value_left = bc.value(0, 0)  # Adjust coordinates as needed
                bc_value_right = bc.value(1, 0)  # Adjust coordinates as needed
            else:
                bc_value_left = bc_value_right = bc.value
            
            # Modify boundary derivatives to enforce Dirichlet condition
            first_deriv = first_deriv.at[0].set(
                (field[1] - bc_value_left) / (self.grid_manager.get_grid_spacing(direction)[0])
            )
            first_deriv = first_deriv.at[-1].set(
                (bc_value_right - field[-2]) / (self.grid_manager.get_grid_spacing(direction)[-1])
            )
        
        elif bc.type == BCType.NEUMANN:
            # Neumann boundary conditions (constant gradient)
            if callable(bc.value):
                # If value is a function, use it at boundaries
                bc_grad_left = bc.value(0, 0)  # Adjust coordinates as needed
                bc_grad_right = bc.value(1, 0)  # Adjust coordinates as needed
            else:
                bc_grad_left = bc_grad_right = bc.value
            
            # Set boundary derivatives to match specified gradient
            first_deriv = first_deriv.at[0].set(bc_grad_left)
            first_deriv = first_deriv.at[-1].set(bc_grad_right)
        
        return first_deriv, second_deriv