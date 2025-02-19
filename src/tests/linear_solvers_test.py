import os
import unittest
from typing import Dict, Optional, Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

from src.core.linear_solvers.base import LinearSolverBase
from src.core.linear_solvers.iterative.cg import ConjugateGradient
from src.core.linear_solvers.iterative.sor import SORSolver

class LinearSolversTestSuite:
    """Test suite for linear solvers"""
    
    @staticmethod
    def generate_test_matrix(n: int, condition_number: float = 10.0) -> jnp.ndarray:
        """
        Generate a symmetric positive definite matrix with a specified condition number
        
        Args:
            n: Size of the matrix
            condition_number: Target condition number
        
        Returns:
            Symmetric positive definite matrix
        """
        # Create random key
        key = jax.random.PRNGKey(0)
        
        # Generate random symmetric matrix
        key, subkey = jax.random.split(key)
        A_rand = jax.random.normal(subkey, (n, n))
        A = A_rand @ A_rand.T
        
        # Compute eigenvalues
        evals, evecs = jnp.linalg.eigh(A)
        
        # Modify eigenvalues to achieve desired condition number
        min_eval = 1.0
        max_eval = condition_number
        modified_evals = jnp.linspace(min_eval, max_eval, n)
        
        # Reconstruct matrix with modified eigenvalues
        return evecs @ jnp.diag(modified_evals) @ evecs.T
    
    @classmethod
    def solve_linear_system(
        cls, 
        solver_class: type,
        matrix_size: int = 100,
        condition_number: float = 10.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> Tuple[dict, plt.Figure]:
        """
        Test a linear solver on a generated system
        
        Args:
            solver_class: Linear solver class to test
            matrix_size: Size of the test matrix
            condition_number: Condition number of the matrix
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
        
        Returns:
            Tuple of (test results, convergence plot)
        """
        # Generate test matrix
        A = cls.generate_test_matrix(matrix_size, condition_number)
        
        # Generate true solution and right-hand side
        key = jax.random.PRNGKey(1)
        x_true = jax.random.normal(key, (matrix_size,))
        b = A @ x_true
        
        # Define matrix-vector product operator
        @jax.jit
        def operator(x: jnp.ndarray) -> jnp.ndarray:
            return A @ x
        
        # Create solver
        solver = solver_class(
            max_iterations=max_iterations, 
            tolerance=tolerance
        )
        
        # Initial guess
        x0 = jnp.zeros_like(b)
        
        # Solve system
        x_solved, history = solver.solve(operator, b, x0)
        
        # Compute error metrics
        residual = jnp.linalg.norm(b - A @ x_solved)
        relative_error = residual / jnp.linalg.norm(b)
        solution_error = jnp.linalg.norm(x_true - x_solved)
        
        # Convergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot residual history
        if 'residual_norms' in history:
            ax.semilogy(history['residual_norms'], '-o')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual Norm')
            ax.set_title(f'{solver_class.__name__} Convergence')
            ax.grid(True)
        
        # Prepare results
        results = {
            'matrix_size': matrix_size,
            'condition_number': condition_number,
            'residual': float(residual),
            'relative_error': float(relative_error),
            'solution_error': float(solution_error),
            'iterations': history.get('iteration_count', 0),
            'converged': history.get('converged', False)
        }
        
        return results, fig
    
    @classmethod
    def run_tests(cls):
        """Run comprehensive linear solver tests"""
        # Create output directory
        os.makedirs('test_results/linear_solvers', exist_ok=True)
        
        # Solvers to test
        solvers = [
            ConjugateGradient,
            SORSolver
        ]
        
        # Test configurations
        test_configs = [
            {'matrix_size': 50, 'condition_number': 10.0},
            {'matrix_size': 100, 'condition_number': 100.0},
            {'matrix_size': 200, 'condition_number': 1000.0}
        ]
        
        # Store test results
        all_results = {}
        
        # Run tests
        for solver in solvers:
            solver_results = []
            
            for config in test_configs:
                print(f"Testing {solver.__name__} with {config}")
                
                # Run test
                result, fig = cls.solve_linear_system(
                    solver, 
                    matrix_size=config['matrix_size'], 
                    condition_number=config['condition_number']
                )
                
                # Save convergence plot
                fig.savefig(
                    f'test_results/linear_solvers/{solver.__name__.lower()}_'
                    f'{config["matrix_size"]}x{config["matrix_size"]}_'
                    f'cond{config["condition_number"]}.png'
                )
                plt.close(fig)
                
                # Store result
                solver_results.append(result)
            
            # Store results for this solver
            all_results[solver.__name__] = solver_results
        
        # Print results
        print("\nLinear Solvers Test Results:")
        for solver_name, results in all_results.items():
            print(f"\n{solver_name}:")
            for result in results:
                print(f"  Matrix Size: {result['matrix_size']}")
                print(f"    Condition Number: {result['condition_number']}")
                print(f"    Residual: {result['residual']:.6e}")
                print(f"    Relative Error: {result['relative_error']:.6e}")
                print(f"    Solution Error: {result['solution_error']:.6e}")
                print(f"    Iterations: {result['iterations']}")
                print(f"    Converged: {result['converged']}")
        
        return all_results

# Run tests when script is executed
if __name__ == '__main__':
    LinearSolversTestSuite.run_tests()