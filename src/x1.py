import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass

@dataclass
class CCDMatrix:
    """Class to hold CCD coefficient matrices"""
    D1: jnp.ndarray  # First derivative matrix
    D2: jnp.ndarray  # Second derivative matrix

class MatrixCCDSolver:
    def __init__(self, n: int, dx: float, left_bc: float = 0.0, right_bc: float = 0.0):
        """
        Initialize the Matrix Compact Difference Solver
        
        Args:
            n (int): Number of grid points
            dx (float): Grid spacing
            left_bc (float, optional): Left boundary condition. Defaults to 0.0.
            right_bc (float, optional): Right boundary condition. Defaults to 0.0.
        """
        self.n = n
        self.dx = dx
        self.left_bc = left_bc
        self.right_bc = right_bc
        
        # Build matrices during initialization
        self.matrices = self._build_matrices()

    def _build_matrices(self) -> CCDMatrix:
        """Build the coefficient matrices for first and second derivatives"""
        # Interior point coefficients from eq(2.6)
        alpha1 = 7.0/16.0
        beta1 = -1.0/16.0
        a1 = 15.0/8.0
        alpha2 = -1.0/8.0
        beta2 = 9.0/4.0
        a2 = 3.0
        
        n = self.n
        D1 = jnp.zeros((n, n))
        D2 = jnp.zeros((n, n))
        
        # Interior points
        for i in range(1, n-1):
            # First derivative - eq(2.7)
            D1 = D1.at[i, i-1].set(-a1/(2*self.dx))
            D1 = D1.at[i, i+1].set(a1/(2*self.dx))
            D1 = D1.at[i, i].set(0.0)
            
            # Second derivative - eq(2.8)
            D2 = D2.at[i, i-1].set(a2/self.dx**2)
            D2 = D2.at[i, i].set(-2*a2/self.dx**2)
            D2 = D2.at[i, i+1].set(a2/self.dx**2)
        
        # Boundary points with three-point stencil and enhanced accuracy
        # Left boundary
        D1 = D1.at[0, 0:3].set(jnp.array([
            -3.0/(2*self.dx),   # Coefficient for f_0
            4.0/(2*self.dx),    # Coefficient for f_1
            -1.0/(2*self.dx)    # Coefficient for f_2
        ]))
        
        D2 = D2.at[0, 0:3].set(jnp.array([
            2.0/(self.dx**2),   # Coefficient for f_0
            -5.0/(self.dx**2),  # Coefficient for f_1
            4.0/(self.dx**2)    # Coefficient for f_2
        ]))
        
        # Right boundary
        D1 = D1.at[-1, -3:].set(jnp.array([
            1.0/(2*self.dx),    # Coefficient for f_{N-2}
            -4.0/(2*self.dx),   # Coefficient for f_{N-1}
            3.0/(2*self.dx)     # Coefficient for f_N
        ]))
        
        D2 = D2.at[-1, -3:].set(jnp.array([
            4.0/(self.dx**2),   # Coefficient for f_{N-2}
            -5.0/(self.dx**2),  # Coefficient for f_{N-1}
            2.0/(self.dx**2)    # Coefficient for f_N
        ]))
        
        return CCDMatrix(D1, D2)
    
    def _compute(self, f: jnp.ndarray, D1: jnp.ndarray, D2: jnp.ndarray, 
                left_bc: float = 0.0, right_bc: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply boundaries and compute derivatives"""
        # Compute derivatives using the matrix representation
        f1 = D1 @ f
        f2 = D2 @ f
        
        return f1, f2

    def compute_derivatives(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute first and second derivatives using matrix multiplication
        Args:
            f: Input array of function values
        Returns:
            Tuple of (first derivative, second derivative)
        """
        return self._compute(f, self.matrices.D1, self.matrices.D2, 
                             self.left_bc, self.right_bc)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable, List
import numpy as np
from matplotlib.gridspec import GridSpec

@dataclass
class TestCase:
    """Test case definition"""
    name: str
    func: Callable[[jnp.ndarray], jnp.ndarray]
    d1_func: Callable[[jnp.ndarray], jnp.ndarray]
    d2_func: Callable[[jnp.ndarray], jnp.ndarray]
    domain: Tuple[float, float]

class CCDTestTool:
    """Test tool for CCD scheme validation"""
    
    def __init__(self, solver_class):
        self.solver_class = solver_class
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create standard test cases"""
        return [
            TestCase(
                name="Sine function",
                func=lambda x: jnp.sin(x),
                d1_func=lambda x: jnp.cos(x),
                d2_func=lambda x: -jnp.sin(x),
                domain=(0, 2*jnp.pi)
            ),
            TestCase(
                name="Gaussian function",
                func=lambda x: jnp.exp(-x**2/2),
                d1_func=lambda x: -x*jnp.exp(-x**2/2),
                d2_func=lambda x: (x**2 - 1)*jnp.exp(-x**2/2),
                domain=(-4, 4)
            ),
            TestCase(
                name="Polynomial function",
                func=lambda x: x**3 - 2*x**2 + x - 1,
                d1_func=lambda x: 3*x**2 - 4*x + 1,
                d2_func=lambda x: 6*x - 4,
                domain=(-2, 2)
            )
        ]
    
    def _compute_errors(self, computed: jnp.ndarray, exact: jnp.ndarray) -> Tuple[float, float, float]:
        """Compute L2, L1, and L∞ errors"""
        l2_error = jnp.sqrt(jnp.mean((computed - exact)**2))
        l1_error = jnp.mean(jnp.abs(computed - exact))
        linf_error = jnp.max(jnp.abs(computed - exact))
        return l2_error, l1_error, linf_error
    
    def run_convergence_study(self, test_case: TestCase, n_points_list: List[int]) -> dict:
        """Run convergence study for different grid resolutions"""
        results = {
            'n_points': n_points_list,
            'l2_errors_d1': [],
            'l2_errors_d2': [],
            'dx_list': []
        }
        
        for n in n_points_list:
            # Create grid
            x = jnp.linspace(test_case.domain[0], test_case.domain[1], n)
            dx = x[1] - x[0]
            
            # Compute exact solutions
            f = test_case.func(x)
            f1_exact = test_case.d1_func(x)
            f2_exact = test_case.d2_func(x)
            
            # Initialize solver and compute derivatives
            solver = self.solver_class(n, dx)
            f1_computed, f2_computed = solver.compute_derivatives(f)
            
            # Compute errors
            l2_error_d1, _, _ = self._compute_errors(f1_computed, f1_exact)
            l2_error_d2, _, _ = self._compute_errors(f2_computed, f2_exact)
            
            results['l2_errors_d1'].append(l2_error_d1)
            results['l2_errors_d2'].append(l2_error_d2)
            results['dx_list'].append(dx)
            
        return results
    
    def visualize_results(self, test_case: TestCase, n_points: int, save_path: str = None):
        """Visualize computation results and errors"""
        # Create grid and compute solutions
        x = jnp.linspace(test_case.domain[0], test_case.domain[1], n_points)
        dx = x[1] - x[0]
        
        f = test_case.func(x)
        f1_exact = test_case.d1_func(x)
        f2_exact = test_case.d2_func(x)
        
        solver = self.solver_class(n_points, dx)
        f1_computed, f2_computed = solver.compute_derivatives(f)
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Plot original function
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x, f, 'b-', label='Original')
        ax1.set_title('Original Function')
        ax1.legend()
        ax1.grid(True)
        
        # Plot first derivative
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x, f1_exact, 'b-', label='Exact')
        ax2.plot(x, f1_computed, 'r--', label='Computed')
        ax2.set_title('First Derivative')
        ax2.legend()
        ax2.grid(True)
        
        # Plot second derivative
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(x, f2_exact, 'b-', label='Exact')
        ax3.plot(x, f2_computed, 'r--', label='Computed')
        ax3.set_title('Second Derivative')
        ax3.legend()
        ax3.grid(True)
        
        # Plot errors
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax4.semilogy(x, np.abs(f1_computed - f1_exact), 'r-', label='First derivative error')
        ax4.semilogy(x, np.abs(f2_computed - f2_exact), 'b-', label='Second derivative error')
        ax4.set_title('Error Analysis')
        ax4.set_ylabel('Absolute Error (log scale)')
        ax4.legend()
        ax4.grid(True)
        
        # Add convergence study
        n_points_list = [50, 100, 200, 400]
        results = self.run_convergence_study(test_case, n_points_list)
        
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.loglog(results['dx_list'], results['l2_errors_d1'], 'ro-', label='First derivative')
        ax5.loglog(results['dx_list'], results['l2_errors_d2'], 'bo-', label='Second derivative')
        ax5.set_xlabel('dx')
        ax5.set_ylabel('L2 Error')
        ax5.set_title('Convergence Study')
        ax5.legend()
        ax5.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# Example usage
def run_tests():    
    test_tool = CCDTestTool(MatrixCCDSolver)
    
    # Run tests for each test case
    for i, test_case in enumerate(test_tool.test_cases):
        print(f"\nRunning test for: {test_case.name}")
        test_tool.visualize_results(
            test_case,
            n_points=200,
            save_path=f'ccd_test_results_{i}.png'
        )

if __name__ == "__main__":
    run_tests()
