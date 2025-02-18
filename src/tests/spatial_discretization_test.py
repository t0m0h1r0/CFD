import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.spatial_discretization.operators.cd import CompactDifference
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition

class SpatialDiscretizationTestSuite:
    """Spatial Discretization Scheme Test Suite with Enhanced Debugging"""
    
    @staticmethod
    def create_test_grid_manager(nx: int = 64, ny: int = 64, 
                                 xmin: float = 0.0, xmax: float = 1.0,
                                 ymin: float = 0.0, ymax: float = 1.0) -> GridManager:
        """
        Create a test grid with configurable dimensions
        
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
    def create_meshgrid(grid_manager: GridManager, indexing: str = 'ij') -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create a meshgrid from grid manager
        
        Args:
            grid_manager: Grid management object
            indexing: Meshgrid indexing style
        
        Returns:
            Tuple of meshgrid coordinates
        """
        x, y, _ = grid_manager.get_coordinates()
        return jnp.meshgrid(x, y, indexing=indexing)
    
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
    ) -> Tuple[float, plt.Figure, bool]:
        """
        Test the accuracy of spatial derivatives with enhanced debugging
        
        Args:
            discretization: Spatial discretization scheme
            test_func: Function to differentiate
            derivative_func: Analytical derivative function
            direction: Differentiation direction
        
        Returns:
            Tuple of (relative error, visualization figure, pass/fail flag)
        """
        # Create grid manager with exact grid configuration
        grid_manager = cls.create_test_grid_manager()
        
        # Create meshgrid with 'ij' indexing
        X, Y = cls.create_meshgrid(grid_manager, indexing='ij')
        
        # Verify grid properties
        print("\nGrid Diagnostic Information:")
        print(f"X grid shape: {X.shape}")
        print(f"Y grid shape: {Y.shape}")
        print(f"X grid range: [{X.min()}, {X.max()}]")
        print(f"Y grid range: [{Y.min()}, {Y.max()}]")
        
        # Compute field using correct grid
        field = test_func(X, Y)
        
        # Verify field properties
        print("\nField Diagnostic Information:")
        print(f"Field shape: {field.shape}")
        print(f"Field min: {field.min()}")
        print(f"Field max: {field.max()}")
        
        # Compute numerical derivatives
        numerical_deriv, _ = discretization.discretize(field, direction)
        
        # Compute analytical derivatives
        analytical_deriv = derivative_func(X, Y)
        
        # Additional checks
        print("\nDerivative Diagnostic Information:")
        print(f"Numerical derivative shape: {numerical_deriv.shape}")
        print(f"Analytical derivative shape: {analytical_deriv.shape}")
        print(f"Numerical derivative min: {numerical_deriv.min()}")
        print(f"Numerical derivative max: {numerical_deriv.max()}")
        print(f"Analytical derivative min: {analytical_deriv.min()}")
        print(f"Analytical derivative max: {analytical_deriv.max()}")
        
        # Robust error calculation
        abs_diff = jnp.abs(numerical_deriv - analytical_deriv)
        max_diff = jnp.max(abs_diff)
        mean_diff = jnp.mean(abs_diff)
        rms_diff = jnp.sqrt(jnp.mean(abs_diff**2))
        
        # Normalize by the magnitude of the analytical derivative
        norm_factor = jnp.max(jnp.abs(analytical_deriv)) + 1e-10
        relative_error = max_diff / norm_factor
        
        print("\nError Diagnostic Information:")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"RMS difference: {rms_diff}")
        print(f"Normalized maximum difference: {relative_error}")
        
        # More stringent error criteria
        passed = (
            relative_error < 1e-3 and  # Relative error
            mean_diff / norm_factor < 1e-3 and  # Mean relative error
            rms_diff / norm_factor < 1e-3  # RMS relative error
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
        
        plt.suptitle(f'Derivative Test - {direction.upper()}')
        plt.tight_layout()
        
        return relative_error, fig, passed
    
    @staticmethod
    def test_func1(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Test function 1: sin(πx)sin(πy)"""
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    
    @staticmethod
    def dx_test_func1(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Analytical x-derivative of test function 1"""
        return jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
    
    @staticmethod
    def dy_test_func1(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Analytical y-derivative of test function 1"""
        return jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
    
    @staticmethod
    def test_func2(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Test function 2: exp(x)cos(y)"""
        return jnp.exp(x) * jnp.cos(y)
    
    @staticmethod
    def dx_test_func2(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Analytical x-derivative of test function 2"""
        return jnp.exp(x) * jnp.cos(y)
    
    @staticmethod
    def dy_test_func2(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Analytical y-derivative of test function 2"""
        return -jnp.exp(x) * jnp.sin(y)
    
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
            
            # Run tests for different test functions and directions
            test_cases = [
                ("Func1 X-Derivative", cls.test_func1, cls.dx_test_func1, 'x'),
                ("Func1 Y-Derivative", cls.test_func1, cls.dy_test_func1, 'y'),
                ("Func2 X-Derivative", cls.test_func2, cls.dx_test_func2, 'x'),
                ("Func2 Y-Derivative", cls.test_func2, cls.dy_test_func2, 'y')
            ]
            
            overall_test_results[scheme_name] = {}
            
            for name, func, deriv_func, direction in test_cases:
                print(f"\n--- Running Test: {scheme_name} - {name} ---")
                
                # Run test
                error, fig, passed = cls.test_derivative_accuracy(
                    discretization, func, deriv_func, direction
                )
                
                # Save figure
                scheme_safe_name = scheme_name.lower().replace(" ", "_")
                fig_filename = f'test_results/spatial_discretization/{scheme_safe_name}_{name.lower().replace(" ", "_")}.png'
                fig.savefig(fig_filename)
                plt.close(fig)
                
                # Store result
                overall_test_results[scheme_name][name] = {
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