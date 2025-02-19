from typing import Tuple, Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import IterativeSolverBase

class ConjugateGradient(IterativeSolverBase):
    """Implementation of the Conjugate Gradient method."""
    
    def __init__(self,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 record_history: bool = False,
                 preconditioner: Optional[Callable] = None):
        """
        Initialize CG solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            record_history: Whether to record convergence history
            preconditioner: Preconditioner function (optional)
        """
        super().__init__(max_iterations, tolerance, record_history)
        self.preconditioner = preconditioner or (lambda x: x)
        
    @partial(jax.jit, static_argnums=(0,1))
    def solve(self,
             operator: Callable,
             b: ArrayLike,
             x0: Optional[ArrayLike] = None) -> Tuple[ArrayLike, dict]:
        """
        Solve the linear system Ax = b using CG method.
        
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
        
        # Initial residual and search direction
        r = b - operator(x)
        z = self.preconditioner(r)
        p = z
        rz_old = jnp.sum(r * z)
        
        # Define the CG iteration step
        def cg_step(carry, _):
            x, r, p, rz_old = carry
            
            # Matrix-vector product
            Ap = operator(p)
            alpha = rz_old / jnp.sum(p * Ap)
            
            # Update solution and residual
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            
            # Apply preconditioner
            z_new = self.preconditioner(r_new)
            rz_new = jnp.sum(r_new * z_new)
            
            # Update search direction
            beta = rz_new / rz_old
            p_new = z_new + beta * p
            
            return (x_new, r_new, p_new, rz_new), jnp.sqrt(rz_new)
            
        # Run the iteration with jax.lax.scan
        init_carry = (x, r, p, rz_old)
        (x, r, p, rz), residual_norms = jax.lax.scan(
            cg_step, init_carry, None, length=self.max_iterations
        )
        
        # Check convergence and update history
        converged = self.check_convergence(residual_norms[-1], self.max_iterations)
        history = self.update_history(
            history, residual_norms[-1], self.max_iterations, converged
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
    
    def precondition(self,
                    preconditioner: Callable) -> 'ConjugateGradient':
        """
        Set the preconditioner for the CG method.
        
        Args:
            preconditioner: Preconditioner function
            
        Returns:
            Self with updated preconditioner
        """
        self.preconditioner = preconditioner
        return self

class PreconditionedCG(ConjugateGradient):
    """Preconditioned Conjugate Gradient with common preconditioners."""
    
    @staticmethod
    def jacobi_preconditioner(diagonal: ArrayLike) -> Callable:
        """
        Create Jacobi (diagonal) preconditioner.
        
        Args:
            diagonal: Diagonal elements of the matrix
            
        Returns:
            Preconditioner function
        """
        def precond(x):
            return x / diagonal
        return precond
    
    @staticmethod
    def ssor_preconditioner(
        operator: Callable,
        omega: float = 1.0
    ) -> Callable:
        """
        Create Symmetric SOR preconditioner.
        
        Args:
            operator: Original matrix operator
            omega: Relaxation parameter
            
        Returns:
            Preconditioner function
        """
        def precond(x):
            # Forward sweep
            y = jnp.zeros_like(x)
            for i in range(len(x)):
                y = y.at[i].set(
                    (x[i] - jnp.sum(operator(y))) / operator(jnp.eye(len(x))[i])[i]
                )
            y = omega * y
            
            # Backward sweep
            z = jnp.zeros_like(x)
            for i in reversed(range(len(x))):
                z = z.at[i].set(
                    (x[i] - jnp.sum(operator(z))) / operator(jnp.eye(len(x))[i])[i]
                )
            z = omega * z
            
            return y + z - omega * x
            
        return precond