from typing import Tuple, Optional, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import IterativeSolverBase

class SORSolver(IterativeSolverBase):
    """Implementation of the Successive Over-Relaxation (SOR) method."""
    
    def __init__(self,
                 omega: float = 1.5,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 record_history: bool = False):
        """
        Initialize SOR solver.
        
        Args:
            omega: Relaxation parameter (1 < omega < 2 for over-relaxation)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            record_history: Whether to record convergence history
        """
        super().__init__(max_iterations, tolerance, record_history)
        self.omega = omega
        
    def solve(self,
             operator: Callable,
             b: ArrayLike,
             x0: Optional[ArrayLike] = None) -> Tuple[ArrayLike, dict]:
        """
        Solve the linear system Ax = b using SOR method.
        
        Args:
            operator: Function that implements the matrix-vector product
            b: Right hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        # Initialize solution and history
        x = x0 if x0 is not None else jnp.zeros_like(b)
        history = self.initialize_history()
        
        # Extract diagonal and off-diagonal parts
        diag = jnp.diag(operator(jnp.eye(len(b))))
        
        def sor_step(state, _):
            x, residual_norm = state
            
            # Perform SOR iteration
            x_new = x.copy()
            for i in range(len(x)):
                # Create unit vector for this component
                e_i = jnp.zeros_like(x).at[i].set(1.0)
                
                # Compute residual for this component
                r_i = b[i] - operator(x)[i]
                
                # Update solution
                x_new = x_new.at[i].set(
                    x[i] + self.omega * r_i / diag[i]
                )
            
            # Compute new residual
            new_residual = b - operator(x_new)
            new_residual_norm = jnp.linalg.norm(new_residual)
            
            return (x_new, new_residual_norm), new_residual_norm
        
        # Run the iteration using scan
        init_state = (x, jnp.linalg.norm(b - operator(x)))
        (x, residual_norm), residual_norms = jax.lax.scan(
            sor_step, init_state, None, length=self.max_iterations
        )
        
        # Check convergence and update history
        converged = self.check_convergence(residual_norm, self.max_iterations)
        history = self.update_history(
            history, residual_norm, self.max_iterations, converged
        )
        
        if self.record_history:
            history['residual_norms'] = residual_norms
            
        return x, history
    
    @partial(jax.jit, static_argnums=(0,))
    def solve_jit(self,
                 operator: Callable,
                 b: ArrayLike,
                 x0: Optional[ArrayLike] = None) -> Tuple[ArrayLike, dict]:
        """
        JIT-compiled version of solve method.
        
        Args:
            operator: Function that implements the matrix-vector product
            b: Right hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        return self.solve(operator, b, x0)
    
    def optimize_omega(self,
                      operator: Callable,
                      b: ArrayLike,
                      x0: Optional[ArrayLike] = None,
                      omega_range: Tuple[float, float] = (1.0, 1.9),
                      n_points: int = 10) -> float:
        """
        Find optimal relaxation parameter by testing different values.
        
        Args:
            operator: Matrix operator
            b: Right hand side vector
            x0: Initial guess (optional)
            omega_range: Range of omega values to test
            n_points: Number of points to test
            
        Returns:
            Optimal omega value found
        """
        def test_omega(omega):
            solver = SORSolver(
                omega=omega,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance
            )
            _, history = solver.solve(operator, b, x0)
            return history['iteration_count']
        
        # Test different omega values
        omegas = jnp.linspace(omega_range[0], omega_range[1], n_points)
        iterations = jnp.array([test_omega(omega) for omega in omegas])
        
        # Return omega with minimum iterations
        return omegas[jnp.argmin(iterations)]

class SymmetricSOR(SORSolver):
    """Implementation of the Symmetric SOR method."""
    
    def solve(self,
             operator: Callable,
             b: ArrayLike,
             x0: Optional[ArrayLike] = None) -> Tuple[ArrayLike, dict]:
        """
        Solve using Symmetric SOR (forward then backward sweep).
        
        Args:
            operator: Matrix operator
            b: Right hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        # Initialize solution and history
        x = x0 if x0 is not None else jnp.zeros_like(b)
        history = self.initialize_history()
        
        # Extract diagonal
        diag = jnp.diag(operator(jnp.eye(len(b))))
        
        def ssor_step(state, _):
            x, residual_norm = state
            
            # Forward sweep
            x_new = x.copy()
            for i in range(len(x)):
                r_i = b[i] - operator(x_new)[i]
                x_new = x_new.at[i].set(
                    x_new[i] + self.omega * r_i / diag[i]
                )
                
            # Backward sweep
            for i in reversed(range(len(x))):
                r_i = b[i] - operator(x_new)[i]
                x_new = x_new.at[i].set(
                    x_new[i] + self.omega * r_i / diag[i]
                )
            
            # Compute new residual
            new_residual = b - operator(x_new)
            new_residual_norm = jnp.linalg.norm(new_residual)
            
            return (x_new, new_residual_norm), new_residual_norm
            
        # Run iteration
        init_state = (x, jnp.linalg.norm(b - operator(x)))
        (x, residual_norm), residual_norms = jax.lax.scan(
            ssor_step, init_state, None, length=self.max_iterations
        )
        
        # Update history
        converged = self.check_convergence(residual_norm, self.max_iterations)
        history = self.update_history(
            history, residual_norm, self.max_iterations, converged
        )
        
        if self.record_history:
            history['residual_norms'] = residual_norms
            
        return x, history