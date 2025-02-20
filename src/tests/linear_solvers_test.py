# src/tests/linear_solvers_test.py

import os
import unittest
from typing import Callable, Dict, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.core.spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
from src.core.linear_solvers.poisson_cg import ConjugateGradientSolver as PoissonSolver, CGSolverConfig as PoissonSolverConfig
#from src.core.linear_solvers.gauss_seidel import PoissonSolver, PoissonSolverConfig
from src.core.common.grid import GridManager, GridConfig
from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.common.types import GridType, BoundaryCondition, BCType

@dataclass
class PoissonTestCase:
    """ポアソン方程式のテストケース"""
    name: str
    solution: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    rhs: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    domain: Tuple[float, float, float] = (1.0, 1.0, 1.0)

class PoissonSolverTest(unittest.TestCase):
    """ポアソンソルバーのテストスイート"""
    
    @classmethod
    def setUpClass(cls):
        """テスト環境のセットアップ"""
        os.makedirs('test_results/poisson_solver', exist_ok=True)
        
        # テストケースの定義
        cls.test_cases = [
            PoissonTestCase(
                name="Simple Harmonic",
                solution=lambda x, y, z: jnp.sin(jnp.pi * x) * 
                                       jnp.sin(jnp.pi * y) * 
                                       jnp.sin(jnp.pi * z),
                rhs=lambda x, y, z: -3 * (jnp.pi**2) * 
                                   jnp.sin(jnp.pi * x) * 
                                   jnp.sin(jnp.pi * y) * 
                                   jnp.sin(jnp.pi * z)
            ),
            PoissonTestCase(
                name="Polynomial",
                solution=lambda x, y, z: x**2 + y**2 + z**2,
                rhs=lambda x, y, z: -6 * jnp.ones_like(x)
            ),
            PoissonTestCase(
                name="Exponential",
                solution=lambda x, y, z: jnp.exp(x + y + z),
                rhs=lambda x, y, z: 3 * jnp.exp(x + y + z)
            )
        ]

    def setUp(self):
        """各テストケース用のソルバー初期化"""
        self.grid_size = 64  # グリッドサイズ
        self.grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(self.grid_size, self.grid_size, self.grid_size),
            grid_type=GridType.UNIFORM
        )
        self.grid_manager = GridManager(self.grid_config)
        
        # 境界条件の設定
        self.boundary_conditions = {
            'left': BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location='left'
            ),
            'right': BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location='right'
            ),
            'bottom': BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location='bottom'
            ),
            'top': BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location='top'
            ),
            'front': BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location='front'
            ),
            'back': BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location='back'
            )
        }
        
        # ラプラシアンソルバーの初期化
        self.laplacian = CCDLaplacianSolver(
            grid_manager=self.grid_manager,
            boundary_conditions=self.boundary_conditions
        )
        
        # ポアソンソルバーの初期化
        self.solver = PoissonSolver(
            config=PoissonSolverConfig(
                max_iterations=10000,  # 最大反復回数を増加
                tolerance=1e-6,        # 収束閾値
                record_history=True,
                relaxation_factor=1.2,
                adaptive_tolerance=True
            ),
            grid_manager=self.grid_manager
        )

    def test_convergence(self):
        """収束テスト"""
        for case in self.test_cases:
            print(f"\nTesting: {case.name}")
            
            # グリッドの生成
            x = jnp.linspace(0, 1, self.grid_size)
            y = jnp.linspace(0, 1, self.grid_size)
            z = jnp.linspace(0, 1, self.grid_size)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            
            # 解析解と右辺の計算
            exact = case.solution(X, Y, Z)
            rhs = case.rhs(X, Y, Z)
            
            # デバッグ出力
            print(f"  RHS Magnitude: {jnp.linalg.norm(rhs)}")
            print(f"  RHS Min: {jnp.min(rhs)}, RHS Max: {jnp.max(rhs)}")
            
            # 数値解の計算
            numerical, history = self.solver.solve(
                self.laplacian,
                rhs,
                jnp.zeros_like(rhs)
            )
            
            # 診断情報の取得
            diagnostics = self.solver.compute_diagnostics(
                numerical, self.laplacian, rhs
            )
            
            # 誤差の計算
            error = jnp.linalg.norm(numerical - exact) / jnp.linalg.norm(exact)
            
            # 結果の可視化
            self._plot_results(
                case.name,
                numerical,
                exact,
                error,
                history,
                diagnostics
            )
            
            # 診断情報の出力
            print("\nDiagnostics:")
            for key, value in diagnostics.items():
                print(f"  {key}: {value}")
            
            # アサーション
            self.assertTrue(
                history['converged'],
                f"{case.name}: Failed to converge"
            )
            self.assertLess(
                error,
                1e-2,  # 許容誤差
                f"{case.name}: Error too large"
            )

    def _plot_results(
        self,
        case_name: str,
        numerical: jnp.ndarray,
        exact: jnp.ndarray,
        error: float,
        history: Dict,
        diagnostics: Dict
    ):
        """結果の可視化"""
        # 中央断面の抽出
        mid_slice = self.grid_size // 2
        num_slice = numerical[:, :, mid_slice]
        exact_slice = exact[:, :, mid_slice]
        diff_slice = jnp.abs(num_slice - exact_slice)
        
        # プロット
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 数値解
        im1 = axes[0, 0].imshow(num_slice, cmap='viridis')
        axes[0, 0].set_title('Numerical Solution')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 厳密解
        im2 = axes[0, 1].imshow(exact_slice, cmap='viridis')
        axes[0, 1].set_title('Exact Solution')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 誤差分布
        im3 = axes[1, 0].imshow(diff_slice, cmap='plasma')
        axes[1, 0].set_title(f'Absolute Error (Max: {error:.2e})')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 収束履歴
        if history.get('residual_history'):
            axes[1, 1].semilogy(history['residual_history'])
            axes[1, 1].set_title('Convergence History')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Residual')
            axes[1, 1].grid(True)
        
        # 診断情報の追加
        plt.suptitle(
            f'{case_name}\n'
            f'Iterations: {history["iterations"]}, '
            f'Final Residual: {history["final_residual"]:.2e}\n'
            f'Relative Error: {error:.2e}, '
            f'Solution Magnitude: {diagnostics["solution_magnitude"]:.2e}'
        )
        
        # 保存
        plt.tight_layout()
        plt.savefig(
            f'test_results/poisson_solver/{case_name.lower().replace(" ", "_")}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def test_adaptive_tolerance(self):
        """アダプティブ許容誤差のテスト"""
        config = PoissonSolverConfig(
            max_iterations=1000,
            tolerance=1e-4,
            adaptive_tolerance=True,
            relaxation_factor=1.5
        )
        solver = PoissonSolver(
            config=config, 
            grid_manager=self.grid_manager
        )
        
        # グリッドの生成
        x = jnp.linspace(0, 1, self.grid_size)
        y = jnp.linspace(0, 1, self.grid_size)
        z = jnp.linspace(0, 1, self.grid_size)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # テストケースの選択
        case = self.test_cases[0]  # Simple Harmonicを選択
        exact = case.solution(X, Y, Z)
        rhs = case.rhs(X, Y, Z)
        
        # 解の計算
        numerical, history = solver.solve(
            self.laplacian,
            rhs,
            jnp.zeros_like(rhs)
        )
        
        # アダプティブ許容誤差の調整のテスト
        initial_tolerance = solver.config.tolerance
        adjusted_tolerance = solver.adaptive_tolerance_adjustment(
            initial_tolerance, 
            history
        )
        
        print("\nAdaptive Tolerance Test:")
        print(f"  Initial Tolerance: {initial_tolerance}")
        print(f"  Adjusted Tolerance: {adjusted_tolerance}")
        print(f"  Iterations: {history['iterations']}")
        
        # アサーション
        self.assertTrue(
            adjusted_tolerance <= initial_tolerance,
            "Adjusted tolerance should not be larger than initial tolerance"
        )
        self.assertTrue(
            history['converged'],
            "Solver should converge with adaptive tolerance"
        )

if __name__ == '__main__':
    unittest.main()