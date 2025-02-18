import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Callable, Tuple

from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition

class SpatialDiscretizationTestSuite:
    """Spatial Discretization Scheme Test Suite"""
    
    @staticmethod
    def create_test_grid_manager(nx: int = 64, ny: int = 64) -> GridManager:
        """Create a uniform test grid"""
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(nx, ny, 1),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)
    
    @staticmethod
    def boundary_conditions() -> dict[str, BoundaryCondition]:
        """Define boundary conditions for tests"""
        return {
            'left': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='left'),
            'right': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='right'),
            'bottom': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='bottom'),
            'top': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='top')
        }
    
    @classmethod
    def test_derivative_accuracy(
        cls, 
        discretization: SpatialDiscretizationBase, 
        test_func: Callable, 
        derivative_func: Callable, 
        direction: str
    ) -> Tuple[float, plt.Figure]:
        """
        Test the accuracy of spatial derivatives
        
        Args:
            discretization: Spatial discretization scheme
            test_func: Function to differentiate
            derivative_func: Analytical derivative function
            direction: Differentiation direction
        
        Returns:
            Tuple of (relative error, error visualization figure)
        """
        # Create grid
        grid_manager = cls.create_test_grid_manager()
        x, y, _ = grid_manager.get_coordinates()
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        # Compute field
        field = test_func(X, Y)
        
        # Compute numerical derivatives
        numerical_deriv, _ = discretization.discretize(field, direction)
        
        # Compute analytical derivatives
        analytical_deriv = derivative_func(X, Y)
        
        # Compute relative error
        error = jnp.linalg.norm(numerical_deriv - analytical_deriv) / jnp.linalg.norm(analytical_deriv)
        
        # Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = ax1.pcolormesh(X, Y, numerical_deriv, shading='auto')
        ax1.set_title(f'Numerical {direction.upper()} Derivative')
        fig.colorbar(im1, ax=ax1)
        
        im2 = ax2.pcolormesh(X, Y, analytical_deriv, shading='auto')
        ax2.set_title(f'Analytical {direction.upper()} Derivative')
        fig.colorbar(im2, ax=ax2)
        
        error_map = jnp.abs(numerical_deriv - analytical_deriv)
        im3 = ax3.pcolormesh(X, Y, error_map, shading='auto')
        ax3.set_title(f'Absolute Error ({direction.upper()} Derivative)')
        fig.colorbar(im3, ax=ax3)
        
        plt.suptitle(f'Derivative Test - {direction.upper()}')
        plt.tight_layout()
        
        return error, fig
    
    @classmethod
    def run_tests(cls):
        """Run comprehensive spatial discretization tests"""
        # Create output directory
        os.makedirs('test_results/spatial_discretization', exist_ok=True)
        
        # Test functions
        def test_func1(x, y):
            """Test function 1: sin(πx)sin(πy)"""
            return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
        
        def dx_test_func1(x, y):
            """Analytical x-derivative of test function 1"""
            return jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
        
        def dy_test_func1(x, y):
            """Analytical y-derivative of test function 1"""
            return jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
        
        def test_func2(x, y):
            """Test function 2: exp(x)cos(y)"""
            return jnp.exp(x) * jnp.cos(y)
        
        def dx_test_func2(x, y):
            """Analytical x-derivative of test function 2"""
            return jnp.exp(x) * jnp.cos(y)
        
        def dy_test_func2(x, y):
            """Analytical y-derivative of test function 2"""
            return -jnp.exp(x) * jnp.sin(y)
        
        # Create discretization
        grid_manager = cls.create_test_grid_manager()
        discretization = CombinedCompactDifference(
            grid_manager=grid_manager,
            boundary_conditions=cls.boundary_conditions()
        )
        
        # Run tests for different test functions and directions
        test_cases = [
            ("Func1 X-Derivative", test_func1, dx_test_func1, 'x'),
            ("Func1 Y-Derivative", test_func1, dy_test_func1, 'y'),
            ("Func2 X-Derivative", test_func2, dx_test_func2, 'x'),
            ("Func2 Y-Derivative", test_func2, dy_test_func2, 'y')
        ]
        
        # Store test results
        test_results = {}
        
        for name, func, deriv_func, direction in test_cases:
            # Run test
            error, fig = cls.test_derivative_accuracy(
                discretization, func, deriv_func, direction
            )
            
            # Save figure
            fig.savefig(f'test_results/spatial_discretization/{name.lower().replace(" ", "_")}.png')
            plt.close(fig)
            
            # Store result
            test_results[name] = {
                'error': float(error),
                'passed': error < 1e-4  # Adjust tolerance as needed
            }
        
        # Print results
        print("Spatial Discretization Test Results:")
        for name, result in test_results.items():
            status = "PASSED" if result['passed'] else "FAILED"
            print(f"{name}: {status} (Error: {result['error']:.6f})")
        
        return test_results

# Run tests when script is executed
if __name__ == '__main__':
    SpatialDiscretizationTestSuite.run_tests()