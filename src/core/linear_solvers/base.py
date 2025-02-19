from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

class LinearSolverBase(ABC):
    """Base class for all linear solvers."""
    
    def __init__(self,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        """
        Initialize linear solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    @abstractmethod
    def solve(self,
             operator: Callable,
             b: ArrayLike,
             x0: Optional[ArrayLike] = None) -> Tuple[ArrayLike, dict]:
        """
        Solve the linear system Ax = b.
        
        Args:
            operator: Function that implements the matrix-vector product
            b: Right hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        pass
    
    def check_convergence(self,
                         residual_norm: float,
                         iteration: int) -> bool:
        """
        Check if the solver has converged.
        
        Args:
            residual_norm: Current residual norm
            iteration: Current iteration number
            
        Returns:
            True if converged, False otherwise
        """
        return (residual_norm < self.tolerance or 
                iteration >= self.max_iterations)

class IterativeSolverBase(LinearSolverBase):
    """Base class for iterative solvers."""
    
    def __init__(self,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 record_history: bool = False):
        """
        Initialize iterative solver.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            record_history: Whether to record convergence history
        """
        super().__init__(max_iterations, tolerance)
        self.record_history = record_history
        
    def initialize_history(self) -> dict:
        """
        Initialize convergence history dictionary.
        
        Returns:
            Empty history dictionary
        """
        return {
            'residual_norms': [],
            'iteration_count': 0,
            'converged': False,
            'final_residual': None
        }
    
    def update_history(self,
                      history: dict,
                      residual_norm: float,
                      iteration: int,
                      converged: bool) -> dict:
        """
        Update convergence history.
        
        Args:
            history: History dictionary to update
            residual_norm: Current residual norm
            iteration: Current iteration number
            converged: Whether solution has converged
            
        Returns:
            Updated history dictionary
        """
        if self.record_history:
            history['residual_norms'].append(residual_norm)
        
        history['iteration_count'] = iteration
        history['converged'] = converged
        history['final_residual'] = residual_norm
        
        return history

class DirectSolverBase(LinearSolverBase):
    """Base class for direct solvers."""
    
    def __init__(self):
        """Initialize direct solver."""
        super().__init__(max_iterations=1, tolerance=0.0)
        
    def solve(self,
             operator: Callable,
             b: ArrayLike,
             x0: Optional[ArrayLike] = None) -> Tuple[ArrayLike, dict]:
        """
        Solve the linear system using a direct method.
        
        Args:
            operator: Function that implements the matrix-vector product
            b: Right hand side vector
            x0: Initial guess (not used for direct solvers)
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        # Direct solvers don't use iterations or convergence checks
        history = {
            'iteration_count': 1,
            'converged': True,
            'final_residual': 0.0
        }
        
        # This method should be overridden by specific direct solver implementations
        raise NotImplementedError("Direct solver implementation needed")