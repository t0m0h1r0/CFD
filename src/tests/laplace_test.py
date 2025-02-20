import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict, Any
from jax.typing import ArrayLike

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition

class LaplacianVisualization:
    """Laplacian計算結果の可視化を担当するクラス"""
    
    @staticmethod
    def normalize_data(data: ArrayLike) -> np.ndarray:
        """
        データを可視化に適したスケールに正規化
        
        Args:
            data: 入力データ
        
        Returns:
            正規化されたデータ
        """
        data_np = np.array(data)
        return (data_np - np.min(data_np)) / (np.max(data_np) - np.min(data_np) + 1e-10)
    
    @classmethod
    def create_comparative_visualization(
        cls, 
        computed_lap: ArrayLike, 
        exact_lap: ArrayLike, 
        grid_size: int, 
        test_name: str
    ):
        """
        ラプラシアン計算結果の比較可視化
        
        Args:
            computed_lap: 計算されたラプラシアン
            exact_lap: 解析的ラプラシアン
            grid_size: グリッドサイズ
            test_name: テスト名
        """
        # 中心スライスの選択
        mid_idx = grid_size // 2
        
        # データの準備
        computed_slice = computed_lap[mid_idx, :, :]
        exact_slice = exact_lap[mid_idx, :, :]
        error_slice = np.abs(computed_lap - exact_lap)[mid_idx, :, :]
        
        # 座標の準備
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # 可視化
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Laplacian Analysis: {test_name} (Grid Size: {grid_size})', fontsize=16)
        
        # 計算されたラプラシアン
        im1 = axs[0].pcolormesh(X, Y, computed_slice, cmap='viridis', shading='auto')
        axs[0].set_title('Computed Laplacian')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axs[0], label='Value')
        
        # 解析的ラプラシアン
        im2 = axs[1].pcolormesh(X, Y, exact_slice, cmap='viridis', shading='auto')
        axs[1].set_title('Exact Laplacian')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axs[1], label='Value')
        
        # 絶対誤差
        im3 = axs[2].pcolormesh(X, Y, error_slice, cmap='plasma', shading='auto')
        axs[2].set_title('Absolute Error')
        axs[2].set_xlabel('X')
        axs[2].set_ylabel('Y')
        plt.colorbar(im3, ax=axs[2], label='|Computed - Exact|')
        
        # レイアウト調整と保存
        plt.tight_layout()
        
        # 結果保存用のディレクトリ作成
        os.makedirs('test_results/laplacian', exist_ok=True)
        plt.savefig(
            f'test_results/laplacian/{test_name.lower().replace(" ", "_")}_grid{grid_size}_visualization.png', 
            dpi=300
        )
        plt.close()

class LaplacianSolverTestSuite:
    """
    CCD Laplacian Solver Test Suite
    
    This test suite verifies:
    1. Laplacian calculation accuracy for analytical functions
    2. Behavior under different boundary conditions
    3. High-order accuracy characteristics
    """
    
    @staticmethod
    def create_test_grid_manager(
        nx: int = 64, 
        ny: int = 64, 
        nz: int = 64
    ) -> GridManager:
        """
        Create a uniform test grid
        
        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            nz: Number of grid points in z direction
        
        Returns:
            GridManager instance
        """
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(nx, ny, nz),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)
    
    @staticmethod
    def default_boundary_conditions() -> dict:
        """
        Set default test boundary conditions
        
        Returns:
            Dictionary of boundary conditions
        """
        return {
            'left': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='left'),
            'right': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='right'),
            'bottom': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='bottom'),
            'top': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='top'),
            'front': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='front'),
            'back': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='back')
        }
    
    @classmethod
    def create_3d_grid(cls, grid_size: int):
        """Create 3D grid"""
        x = jnp.linspace(0, 1, grid_size)
        y = jnp.linspace(0, 1, grid_size)
        z = jnp.linspace(0, 1, grid_size)
        return jnp.meshgrid(x, y, z, indexing='ij')
    
    @classmethod
    def compute_relative_error(
        cls, 
        computed: ArrayLike, 
        exact: ArrayLike
    ) -> float:
        """
        Compute relative error
        
        Args:
            computed: Computed values
            exact: Analytical values
        
        Returns:
            Relative error
        """
        # 相対誤差の計算（安定化のため、分母に微小値を追加）
        error = jnp.linalg.norm(computed - exact)
        norm = jnp.linalg.norm(exact)
        return float(error / (norm + 1e-10))
    
    @classmethod
    def test_laplacian_accuracy(
        cls, 
        test_name: str, 
        test_function: Callable, 
        exact_laplacian: Callable,
        grid_sizes: list = [32, 64, 128]
    ) -> dict:
        """
        Test Laplacian calculation accuracy
        
        Args:
            test_name: Name of the test
            test_function: Test function
            exact_laplacian: Analytical Laplacian function
            grid_sizes: List of grid sizes to test
        
        Returns:
            Dictionary of test results
        """
        # Store test results
        test_results = {
            'name': test_name,
            'grid_convergence': []
        }
        
        # Test accuracy for different grid sizes
        for grid_size in grid_sizes:
            # Create grid
            X, Y, Z = cls.create_3d_grid(grid_size)
            
            # Compute function and analytical Laplacian
            field = test_function(X, Y, Z)
            exact_lap = exact_laplacian(X, Y, Z)
            
            # Create grid manager
            grid_manager = cls.create_test_grid_manager(grid_size, grid_size, grid_size)
            
            # Create Laplacian solver
            laplacian_solver = CCDLaplacianSolver(
                grid_manager=grid_manager,
                boundary_conditions=cls.default_boundary_conditions(),
                order=8  # 8th-order accuracy
            )
            
            # Compute Laplacian
            computed_lap = laplacian_solver.compute_laplacian(field)
            
            # Compute relative error
            relative_error = cls.compute_relative_error(computed_lap, exact_lap)
            
            # Store grid convergence information
            test_results['grid_convergence'].append({
                'grid_size': grid_size,
                'relative_error': relative_error
            })
            
            # Visualize results
            LaplacianVisualization.create_comparative_visualization(
                computed_lap, exact_lap, grid_size, test_name
            )
        
        return test_results
    
    @classmethod
    def run_laplacian_tests(cls):
        """
        Run all Laplacian-related tests
        
        Returns:
            Dictionary of all test results
        """
        # Test function cases
        test_cases = [
            # Test Case 1: Quadratic Function Laplacian
            {
                'name': 'Quadratic Function Laplacian',
                'function': lambda x, y, z: x**2 + y**2 + z**2,
                'laplacian': lambda x, y, z: 2 * jnp.ones_like(x) + 2 * jnp.ones_like(y) + 2 * jnp.ones_like(x)
            },
            # Test Case 2: Sine Function Laplacian
            {
                'name': 'Sine Function Laplacian',
                'function': lambda x, y, z: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z),
                'laplacian': lambda x, y, z: -3 * (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
            },
            # Test Case 3: Exponential Function Laplacian
            {
                'name': 'Exponential Function Laplacian',
                'function': lambda x, y, z: jnp.exp(x + y + z),
                'laplacian': lambda x, y, z: 3 * jnp.exp(x + y + z)
            },
        ]
        
        # Store test results
        all_results = {}
        
        # Run each test case
        for case in test_cases:
            test_result = cls.test_laplacian_accuracy(
                test_name=case['name'],
                test_function=case['function'],
                exact_laplacian=case['laplacian']
            )
            all_results[case['name']] = test_result
        
        # Print test results
        print("Laplacian Solver Test Results:")
        for name, result in all_results.items():
            print(f"\n{name}:")
            for grid_result in result['grid_convergence']:
                print(f"  Grid Size: {grid_result['grid_size']}")
                print(f"    Relative Error: {grid_result['relative_error']:.6e}")
        
        return all_results

# Execute tests when script is run directly
if __name__ == '__main__':
    LaplacianSolverTestSuite.run_laplacian_tests()