import os
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition
from src.core.linear_solvers.sor_solver import PoissonSORSolver

class PoissonSORSolverTestSuite(unittest.TestCase):
    """
    ポアソンSORソルバのテストスイート
    
    テスト内容:
    1. 単純な解析解との比較
    2. 異なる境界条件下での解の検証
    3. パラメータ最適化の検証
    4. エラー評価
    """
    
    @classmethod
    def setUpClass(cls):
        """テスト共通の設定"""
        # テスト結果保存用ディレクトリの作成
        os.makedirs('test_results/poisson_sor', exist_ok=True)
    
    def _create_grid_manager(self, grid_size=64):
        """均一グリッドの作成"""
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(grid_size, grid_size, grid_size),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)
    
    def _create_default_boundary_conditions(self):
        """デフォルトのディリクレ境界条件"""
        return {
            'left': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='left'),
            'right': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='right'),
            'bottom': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='bottom'),
            'top': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='top'),
            'front': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='front'),
            'back': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='back')
        }
    
    def test_simple_source_term(self):
        """単純な解析解との比較テスト"""
        grid_manager = self._create_grid_manager()
        boundary_conditions = self._create_default_boundary_conditions()
        
        # 解析解の設定: p = sin(πx)sin(πy)sin(πz)に対応するソース項
        x, y, z = grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # ソース項の計算: -∇²p = -3π²sin(πx)sin(πy)sin(πz)
        f = -3 * (jnp.pi**2) * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.sin(jnp.pi * Z)
        
        # 解析解
        p_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.sin(jnp.pi * Z)
        
        # SORソルバの初期化と解法
        solver = PoissonSORSolver(
            grid_manager=grid_manager,
            boundary_conditions=boundary_conditions,
            omega=1.5,
            max_iterations=1000,
            tolerance=1e-6,
            verbose=False
        )
        
        p_solved, history = solver.solve(f)
        
        # 相対誤差の計算
        relative_error = jnp.linalg.norm(p_solved - p_exact) / jnp.linalg.norm(p_exact)
        
        # 結果の可視化
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title('解析解')
        plt.pcolormesh(X[0], Y[0], p_exact[0], shading='auto')
        plt.colorbar()
        
        plt.subplot(132)
        plt.title('数値解')
        plt.pcolormesh(X[0], Y[0], p_solved[0], shading='auto')
        plt.colorbar()
        
        plt.subplot(133)
        plt.title('絶対誤差')
        plt.pcolormesh(X[0], Y[0], jnp.abs(p_solved - p_exact)[0], shading='auto')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('test_results/poisson_sor/simple_source_term.png')
        plt.close()
        
        # 検証
        print(f"相対誤差: {relative_error}")
        self.assertTrue(relative_error < 1e-3, 
                        f"相対誤差が許容範囲を超えています: {relative_error}")
        self.assertTrue(history['converged'], "収束しませんでした")
    
    def test_omega_optimization(self):
        """緩和パラメータの最適化テスト"""
        grid_manager = self._create_grid_manager(32)
        boundary_conditions = self._create_default_boundary_conditions()
        
        # ソース項の生成
        x, y, z = grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        f = jnp.exp(-(X-0.5)**2 - (Y-0.5)**2 - (Z-0.5)**2)
        
        solver = PoissonSORSolver(
            grid_manager=grid_manager,
            boundary_conditions=boundary_conditions,
            omega=1.5,
            max_iterations=500,
            tolerance=1e-5,
            verbose=False
        )
        
        # 最適な緩和パラメータの探索
        optimal_omega = solver.optimize_omega(f)
        
        # 最適化された緩和パラメータでの解法
        solver.omega = optimal_omega
        p_solved, history = solver.solve(f)
        
        # 結果の可視化と検証
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title(f'最適なω = {optimal_omega:.4f}')
        plt.plot(history['residual_history'], label='収束曲線')
        plt.xlabel('反復回数')
        plt.ylabel('残差')
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(132)
        plt.title('解の中心スライス')
        plt.pcolormesh(p_solved[p_solved.shape[0]//2], shading='auto')
        plt.colorbar()
        
        plt.subplot(133)
        plt.title('ソース項の中心スライス')
        plt.pcolormesh(f[f.shape[0]//2], shading='auto')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('test_results/poisson_sor/omega_optimization.png')
        plt.close()
        
        # 検証
        print(f"最適な緩和パラメータ: {optimal_omega}")
        self.assertTrue(1.0 < optimal_omega < 2.0, 
                        f"最適なωが不正: {optimal_omega}")
        self.assertTrue(history['converged'], "収束しませんでした")
    
    def test_different_boundary_conditions(self):
        """異なる境界条件の検証"""
        grid_manager = self._create_grid_manager(32)
        
        # テスト用の境界条件セット
        test_cases = [
            {
                'name': 'Dirichlet境界条件',
                'boundary_conditions': {
                    'left': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='left'),
                    'right': BoundaryCondition(type=BCType.DIRICHLET, value=1.0, location='right'),
                    'bottom': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='bottom'),
                    'top': BoundaryCondition(type=BCType.DIRICHLET, value=1.0, location='top'),
                    'front': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='front'),
                    'back': BoundaryCondition(type=BCType.DIRICHLET, value=1.0, location='back')
                }
            },
            {
                'name': '混合境界条件（ディリクレとノイマン）',
                'boundary_conditions': {
                    'left': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='left'),
                    'right': BoundaryCondition(type=BCType.NEUMANN, value=1.0, location='right'),
                    'bottom': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='bottom'),
                    'top': BoundaryCondition(type=BCType.NEUMANN, value=1.0, location='top'),
                    'front': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='front'),
                    'back': BoundaryCondition(type=BCType.NEUMANN, value=1.0, location='back')
                }
            }
        ]
        
        for test_case in test_cases:
            # ソース項の生成
            x, y, z = grid_manager.get_coordinates()
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            
            # ソース項: 複雑な関数を使用
            f = jnp.exp(-(X-0.5)**2 - (Y-0.5)**2 - (Z-0.5)**2)
            
            # ソルバの初期化
            solver = PoissonSORSolver(
                grid_manager=grid_manager,
                boundary_conditions=test_case['boundary_conditions'],
                omega=1.5,
                max_iterations=1000,
                tolerance=1e-6,
                verbose=False
            )
            
            # 解法
            p_solved, history = solver.solve(f)
            
            # 結果の可視化
            plt.figure(figsize=(15, 5))
            plt.suptitle(test_case['name'])
            
            plt.subplot(131)
            plt.title('解の中心スライス')
            plt.pcolormesh(p_solved[p_solved.shape[0]//2], shading='auto')
            plt.colorbar()
            
            plt.subplot(132)
            plt.title('ソース項の中心スライス')
            plt.pcolormesh(f[f.shape[0]//2], shading='auto')
            plt.colorbar()
            
            plt.subplot(133)
            plt.title('残差履歴')
            plt.plot(history['residual_history'])
            plt.xlabel('反復回数')
            plt.ylabel('残差')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(f'test_results/poisson_sor/{test_case["name"].replace(" ", "_")}.png')
            plt.close()
            
            # 検証
            print(f"{test_case['name']}:")
            print(f"  反復回数: {history['iterations']}")
            print(f"  最終残差: {history['residual_history'][-1]}")
            
            self.assertTrue(history['converged'], 
                            f"{test_case['name']}で収束しませんでした")
            self.assertLess(history['residual_history'][-1], 1e-5, 
                            f"{test_case['name']}での残差が大きすぎます")
    
    @classmethod
    def tearDownClass(cls):
        """テスト後処理"""
        print("ポアソンSORソルバテストスイートが完了しました。")

def run_tests():
    """テストの実行"""
    unittest.main(argv=[''], exit=False)

if __name__ == '__main__':
    run_tests()