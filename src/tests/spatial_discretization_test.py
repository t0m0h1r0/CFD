import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.spatial_discretization.operators.cd import CompactDifference
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition

class SpatialDiscretizationTestSuite:
    """Spatial Discretization Scheme Test Suite with Precise Verification"""
    
    @staticmethod
    def create_test_grid_manager(
        nx: int = 64, 
        ny: int = 64, 
        xmin: float = 0.0, 
        xmax: float = 1.0,
        ymin: float = 0.0, 
        ymax: float = 1.0
    ) -> GridManager:
        """
        Create a uniform test grid with precise configuration
        
        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            xmin: Minimum x coordinate
            xmax: Maximum x coordinate
            ymin: Minimum y coordinate
            ymax: Maximum y coordinate
        
        Returns:
            GridManager instance
        """
        grid_config = GridConfig(
            dimensions=(xmax-xmin, ymax-ymin, 1.0),
            points=(nx, ny, 1),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)
    
    @staticmethod
    def create_meshgrid(
        grid_manager: GridManager, 
        indexing: str = 'ij'
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create a meshgrid from grid manager with precise indexing
        
        Args:
            grid_manager: Grid management object
            indexing: Meshgrid indexing style
        
        Returns:
            Tuple of meshgrid coordinates
        """
        x, y, _ = grid_manager.get_coordinates()
        return jnp.meshgrid(x, y, indexing=indexing)
    
    @staticmethod
    def compute_analytical_derivative(
        func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        direction: str, 
        h: float = 1e-8
    ) -> jnp.ndarray:
        """
        Compute highly accurate finite difference derivative
        
        Args:
            func: Function to differentiate
            x: X coordinates
            y: Y coordinates
            direction: Differentiation direction ('x' or 'y')
            h: Small perturbation for finite difference
        
        Returns:
            Analytical derivative using central difference
        """
        if direction == 'x':
            # Central difference for x-derivative
            f_plus = func(x + h, y)
            f_minus = func(x - h, y)
            return (f_plus - f_minus) / (2 * h)
        elif direction == 'y':
            # Central difference for y-derivative
            f_plus = func(x, y + h)
            f_minus = func(x, y - h)
            return (f_plus - f_minus) / (2 * h)
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    @staticmethod
    def test_functions() -> list:
        """
        Provide a comprehensive set of test functions
        
        Returns:
            List of test functions
        """
        return [
            # Test function 1: Sinusoidal function
            {
                'func': lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y),
                'dx': lambda x, y: jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y),
                'dy': lambda x, y: jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y),
                'name': 'Sinusoidal'
            },
            # Test function 2: Exponential function
            {
                'func': lambda x, y: jnp.exp(x) * jnp.cos(y),
                'dx': lambda x, y: jnp.exp(x) * jnp.cos(y),
                'dy': lambda x, y: -jnp.exp(x) * jnp.sin(y),
                'name': 'Exponential'
            },
            # Test function 3: Polynomial function
            {
                'func': lambda x, y: x**2 * y**3,
                'dx': lambda x, y: 2 * x * y**3,
                'dy': lambda x, y: 3 * x**2 * y**2,
                'name': 'Polynomial'
            }
        ]
    
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
        test_func: dict,
        direction: str
    ) -> Tuple[float, plt.Figure, bool]:
        """
        Test the accuracy of spatial derivatives with comprehensive verification
        
        Args:
            discretization: Spatial discretization scheme
            test_func: Test function dictionary
            direction: Differentiation direction
        
        Returns:
            Tuple of (relative error, visualization figure, pass/fail flag)
        """
        # Create grid manager with high-resolution configuration
        grid_manager = cls.create_test_grid_manager(nx=128, ny=128)
        
        # Create meshgrid with precise indexing
        X, Y = cls.create_meshgrid(grid_manager, indexing='ij')
        
        # Compute field
        field = test_func['func'](X, Y)
        
        # Compute numerical derivatives
        numerical_deriv, _ = discretization.discretize(field, direction)
        
        # Compute highly accurate analytical derivative
        if direction == 'x':
            analytical_deriv = test_func['dx'](X, Y)
        else:
            analytical_deriv = test_func['dy'](X, Y)
        
        # Compute verification derivative using finite difference
        verification_deriv = cls.compute_analytical_derivative(
            test_func['func'], X, Y, direction
        )
        
        # Compute various error metrics
        abs_diff = jnp.abs(numerical_deriv - analytical_deriv)
        verification_diff = jnp.abs(numerical_deriv - verification_deriv)
        
        max_abs_diff = jnp.max(abs_diff)
        max_verification_diff = jnp.max(verification_diff)
        mean_abs_diff = jnp.mean(abs_diff)
        
        # Normalize by the magnitude of the derivative
        norm_factor = jnp.max(jnp.abs(analytical_deriv)) + 1e-10
        relative_error = max_abs_diff / norm_factor
        verification_error = max_verification_diff / norm_factor
        
        # Detailed diagnostics
        print(f"\n--- {test_func['name']} Function Derivative Test ---")
        print(f"Direction: {direction}")
        print(f"Max Absolute Difference: {max_abs_diff}")
        print(f"Verification Difference: {max_verification_diff}")
        print(f"Mean Absolute Difference: {mean_abs_diff}")
        print(f"Relative Error: {relative_error}")
        print(f"Verification Error: {verification_error}")
        
        # Comprehensive pass criteria
        passed = (
            relative_error < 1e-3 and  # Relative error
            verification_error < 1e-3 and  # Verification error
            mean_abs_diff / norm_factor < 1e-3  # Mean relative error
        )
        
        # Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = ax1.pcolormesh(X, Y, numerical_deriv, shading='auto')
        ax1.set_title(f'Numerical {direction.upper()} Derivative')
        fig.colorbar(im1, ax=ax1)
        
        im2 = ax2.pcolormesh(X, Y, analytical_deriv, shading='auto')
        ax2.set_title(f'Analytical {direction.upper()} Derivative')
        fig.colorbar(im2, ax=ax2)
        
        im3 = ax3.pcolormesh(X, Y, abs_diff, shading='auto')
        ax3.set_title(f'Absolute Error ({direction.upper()} Derivative)')
        fig.colorbar(im3, ax=ax3)
        
        plt.suptitle(f'Derivative Test - {test_func["name"]} Function')
        plt.tight_layout()
        
        return relative_error, fig, passed
    
    @classmethod
    def run_tests(cls):
        """Run comprehensive spatial discretization tests"""
        # Create output directory
        os.makedirs('test_results/spatial_discretization', exist_ok=True)
        
        # Discretization schemes to test
        discretization_schemes = [
            ('Combined Compact Difference', 
             lambda grid_manager: CombinedCompactDifference(
                 grid_manager=grid_manager,
                 boundary_conditions=cls.boundary_conditions()
             )),
            ('Compact Difference', 
             lambda grid_manager: CompactDifference(
                 grid_manager=grid_manager,
                 boundary_conditions=cls.boundary_conditions()
             ))
        ]
        
        # Overall test results
        overall_test_results = {}
        
        # Test each discretization scheme
        for scheme_name, discretization_factory in discretization_schemes:
            # Create grid and discretization
            grid_manager = cls.create_test_grid_manager()
            discretization = discretization_factory(grid_manager)
            
            # Test functions
            test_functions = cls.test_functions()
            
            # Store scheme-specific results
            overall_test_results[scheme_name] = {}
            
            # Test each function in both x and y directions
            for test_func in test_functions:
                for direction in ['x', 'y']:
                    test_name = f"{test_func['name']} {direction.upper()}-Derivative"
                    
                    # Run test
                    error, fig, passed = cls.test_derivative_accuracy(
                        discretization, test_func, direction
                    )
                    
                    # Save figure
                    scheme_safe_name = scheme_name.lower().replace(" ", "_")
                    fig_filename = f'test_results/spatial_discretization/{scheme_safe_name}_{test_name.lower().replace(" ", "_")}.png'
                    fig.savefig(fig_filename)
                    plt.close(fig)
                    
                    # Store result
                    overall_test_results[scheme_name][test_name] = {
                        'error': float(error),
                        'passed': passed
                    }
            
            # Print results for this scheme
            print(f"\nSpatial Discretization Test Results for {scheme_name}:")
            for name, result in overall_test_results[scheme_name].items():
                status = "PASSED" if result['passed'] else "FAILED"
                print(f"{name}: {status} (Error: {result['error']:.6f})")
        
        return overall_test_results

# Run tests when script is executed
if __name__ == '__main__':
    SpatialDiscretizationTestSuite.run_tests()