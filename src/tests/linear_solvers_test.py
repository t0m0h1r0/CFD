# src/core/linear_solvers/tests/test_gauss_seidel.py

import os
import unittest
from typing import Callable, Dict, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.core.spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
from src.core.linear_solvers.gauss_seidel import GaussSeidelSolver, LinearSolverConfig
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BoundaryCondition, BCType

@dataclass
class PoissonTestCase:
    """ポアソン方程式のテストケース"""
    name: str
    solution: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    rhs: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    domain: Tuple[float, float, float] = (1.0, 1.0, 1.0)

class GaussSeidelTest(unittest.TestCase):
    """ガウス=サイデル法のテストスイート"""
    
    @classmethod
    def setUpClass(cls):
        """テスト環境のセットアップ"""
        os.makedirs('test_results/gauss_seidel', exist_ok=True)
        
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
            )
        ]

# src/tests/linear_solvers_test.py

    def setUp(self):
        """各テストケース用のソルバー初期化"""
        self.grid_size = 32
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
        
        # ガウス=サイデルソルバーの初期化
        # テストコードの該当部分
        self.solver = GaussSeidelSolver(
            config=LinearSolverConfig(
                max_iterations=1000,
                tolerance=1e-6,
                record_history=True
            ),
            discretization=self.laplacian,  # ここで適切なdiscretizationを渡す
            omega=1.0
        )

    def test_convergence(self):
        """収束テスト"""
        for case in self.test_cases:
            # グリッドの生成
            x = jnp.linspace(0, 1, self.grid_size)
            y = jnp.linspace(0, 1, self.grid_size)
            z = jnp.linspace(0, 1, self.grid_size)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            
            # 解析解と右辺の計算
            exact = case.solution(X, Y, Z)
            rhs = case.rhs(X, Y, Z)
            
            # 数値解の計算
            numerical, history = self.solver.solve(
                self.laplacian,
                rhs,
                jnp.zeros_like(rhs)
            )
            
            # 誤差の計算
            error = jnp.linalg.norm(numerical - exact) / jnp.linalg.norm(exact)
            
            # 結果の可視化
            self._plot_results(
                case.name,
                numerical,
                exact,
                error,
                history
            )
            
            # アサーション
            self.assertTrue(
                history['converged'],
                f"{case.name}: Failed to converge"
            )
            self.assertLess(
                error,
                1e-3,
                f"{case.name}: Error too large"
            )

    def _plot_results(
        self,
        case_name: str,
        numerical: jnp.ndarray,
        exact: jnp.ndarray,
        error: float,
        history: Dict
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
        im1 = axes[0, 0].imshow(num_slice)
        axes[0, 0].set_title('Numerical Solution')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 厳密解
        im2 = axes[0, 1].imshow(exact_slice)
        axes[0, 1].set_title('Exact Solution')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 誤差分布
        im3 = axes[1, 0].imshow(diff_slice)
        axes[1, 0].set_title(f'Absolute Error (Max: {error:.2e})')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 収束履歴
        if history['residual_history']:
            axes[1, 1].semilogy(history['residual_history'])
            axes[1, 1].set_title('Convergence History')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Residual')
            axes[1, 1].grid(True)
        
        plt.suptitle(
            f'{case_name}\n'
            f'Iterations: {history["iterations"]}, '
            f'Final Residual: {history["final_residual"]:.2e}'
        )
        
        # 保存
        plt.savefig(
            f'test_results/gauss_seidel/{case_name.lower().replace(" ", "_")}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

if __name__ == '__main__':
    unittest.main()