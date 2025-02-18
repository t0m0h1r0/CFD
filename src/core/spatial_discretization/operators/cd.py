from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import SpatialDiscretizationBase
from ...common.types import Grid, BoundaryCondition, BCType
from ...common.grid import GridManager

class CompactDifference(SpatialDiscretizationBase):
    """Implementation of standard Compact Difference scheme."""
    
    def __init__(self,
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None,
                 order: int = 4):
        """
        Initialize Compact Difference scheme.
        
        Args:
            grid_manager: Grid management object
            boundary_conditions: Dictionary of boundary conditions
            order: Order of accuracy (default: 4)
        """
        # Calculate coefficients based on the order
        coefficients = self._calculate_coefficients(order)
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = coefficients
        self.order = order
        
    def _calculate_coefficients(self, order: int) -> dict:
        """
        Calculate Compact Difference coefficients for given order.
        
        Args:
            order: Order of accuracy
            
        Returns:
            Dictionary of coefficients
        """
        if order == 4:
            return {
                # Standard 4th order compact difference coefficients
                'alpha': 1/4,  # Central coefficient for first derivative
                'beta_l': 1/5,  # Left coefficient
                'beta_r': 1/5,  # Right coefficient
                'first_alpha': 3/2,  # First derivative central coefficient
                'second_alpha': 10/12  # Second derivative central coefficient
            }
        elif order == 6:
            return {
                # 6th order compact difference coefficients
                'alpha': 1/3,
                'beta_l': 1/6,
                'beta_r': 1/6,
                'first_alpha': 11/12,
                'second_alpha': 5/6
            }
        else:
            raise NotImplementedError(f"Order {order} not implemented")
            
    def discretize(self,
                  field: ArrayLike,
                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute spatial derivatives using Compact Difference scheme.
        
        Args:
            field: Input field to differentiate
            direction: Direction of differentiation
            
        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # Extract coefficients
        alpha = self.coefficients['alpha']
        beta_l = self.coefficients['beta_l']
        beta_r = self.coefficients['beta_r']
        first_alpha = self.coefficients['first_alpha']
        second_alpha = self.coefficients['second_alpha']
        
        # Get grid spacing
        dx = self.grid_manager.get_grid_spacing(direction)
        
        # Initialize derivative arrays
        first_deriv = jnp.zeros_like(field)
        second_deriv = jnp.zeros_like(field)
        
        # First derivative computation using compact scheme
        for i in range(1, len(field)-1):
            # Compact scheme first derivative
            if i > 0 and i < len(field) - 1:
                first_deriv = first_deriv.at[i].set(
                    (field[i+1] - field[i-1]) / (2 * dx) +
                    beta_l * (first_deriv[i+1] + first_deriv[i-1]) / (2 * dx)
                )
        
        # Second derivative computation
        for i in range(1, len(field)-1):
            # Compact scheme second derivative
            if i > 0 and i < len(field) - 1:
                second_deriv = second_deriv.at[i].set(
                    (field[i+1] - 2*field[i] + field[i-1]) / (dx**2) +
                    beta_l * (second_deriv[i+1] + second_deriv[i-1]) / (dx**2)
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
        Apply boundary conditions for Compact Difference scheme.
        
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
            dx = self.grid_manager.get_grid_spacing(direction)[0]
            first_deriv = first_deriv.at[0].set(
                (field[1] - bc_value_left) / dx
            )
            first_deriv = first_deriv.at[-1].set(
                (bc_value_right - field[-2]) / dx
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