import os
import time
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition
from src.core.linear_solvers.sor_solver import PoissonSORSolver

class ConvergenceMonitor:
    """収束状況を追跡・可視化するクラス"""
    def __init__(self, test_name: str, output_dir: str = 'test_results/poisson_sor'):
        self.test_name = test_name
        self.output_dir = output_dir
        self.iterations = []
        self.residuals = []
        self.solutions = []
        self.computation_times = []
    
    def update(self, iteration: int, residual: float, solution: jnp.ndarray, computation_time: float):
        """反復の進捗を記録"""
        self.iterations.append(iteration)
        self.residuals.append(residual)
        self.solutions.append(solution)
        self.computation_times.append(computation_time)
    
    def visualize_convergence(
        self, 
        source_term: jnp.ndarray, 
        omega: float, 
        boundary_conditions: Dict[str, BoundaryCondition]
    ):
        """収束過程を可視化"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'{self.test_name} - 収束解析', fontsize=16)
        
        # 残差履歴
        plt.subplot(2, 3, 1)
        plt.title('残差履歴')
        plt.plot(self.iterations, self.residuals, marker='o')
        plt.xlabel('反復回数')
        plt.ylabel('残差')
        plt.yscale('log')
        plt.grid(True)
        
        # 収束速度
        plt.subplot(2, 3, 2)
        plt.title('収束速度')
        diff_residuals = np.diff(self.residuals)
        plt.plot(self.iterations[1:], np.abs(diff_residuals), marker='o')
        plt.xlabel('反復回数')
        plt.ylabel('残差変化量')
        plt.yscale('log')
        plt.grid(True)
        
        # 計算時間
        plt.subplot(2, 3, 3)
        plt.title('反復ごとの計算時間')
        plt.plot(self.iterations, self.computation_times, marker='o')
        plt.xlabel('反復回数')
        plt.ylabel('計算時間 (秒)')
        plt.grid(True)
        
        # 解の進化
        selected_iterations = [
            0,  # 初期状態
            len(self.solutions) // 2,  # 中間
            -1  # 最終状態
        ]
        
        for i, iter_idx in enumerate(selected_iterations):
            plt.subplot(2, 3, 4+i)
            plt.title(f'解の進化 (反復 {self.iterations[iter_idx]})')
            plt.pcolormesh(self.solutions[iter_idx][self.solutions[iter_idx].shape[0]//2], shading='auto')
            plt.colorbar(label='値')
        
        # パラメータと境界条件の表示
        boundary_text = '\n'.join([
            f'{k}: {v.type}, value={v.value}' 
            for k, v in boundary_conditions.items()
        ])
        plt.figtext(0.02, 0.02, 
                    f'ω = {omega:.4f}\n境界条件:\n{boundary_text}', 
                    verticalalignment='bottom', 
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{self.test_name}_convergence.png', dpi=300)
        plt.close()
    
    def print_summary(self):
        """収束プロセスの要約を出力"""
        print(f"\n{self.test_name} - 収束解析:")
        print(f"  最終反復回数: {self.iterations[-1]}")
        print(f"  初期残差: {self.residuals[0]:.6e}")
        print(f"  最終残差: {self.residuals[-1]:.6e}")
        print(f"  残差低減率: {self.residuals[0] / self.residuals[-1]:.2f}")
        print(f"  総計算時間: {sum(self.computation_times):.4f}秒")
        print(f"  平均反復時間: {np.mean(self.computation_times):.6f}秒")

class PoissonSORSolverTestSuite(unittest.TestCase):
    """
    ポアソンSORソルバのテストスイート
    
    テスト内容:
    1. 単純な解析解との比較
    2. 異なる境界条件下での解の検証
    3. パラメータ最適化の検証
    4. エラー評価
    """
    
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
        # 収束モニターの初期化
        monitor = ConvergenceMonitor('SimpleSourceTerm解析')
        
        grid_manager = self._create_grid_manager()
        boundary_conditions = self._create_default_boundary_conditions()
        
        # 解析解の設定: p = sin(πx)sin(πy)sin(πz)に対応するソース項
        x, y, z = grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # ソース項の計算: -∇²p = -3π²sin(πx)sin(πy)sin(πz)
        f = -3 * (jnp.pi**2) * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.sin(jnp.pi * Z)
        
        # 解析解
        p_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.sin(jnp.pi * Z)
        
        # カスタムSORソルバの実装（収束モニタリング付き）
        class MonitoredSORSolver(PoissonSORSolver):
            def solve(self, f, initial_guess=None):
                # トラッキング用の拡張
                p = initial_guess if initial_guess is not None else jnp.zeros_like(f)
                
                # 反復のトラッキングと時間計測
                start_time = time.time()
                history = {
                    'iterations': 0,
                    'residual_history': [],
                    'converged': False
                }
                
                # 反復終了条件の定義
                def convergence_condition(state):
                    p, f, iteration = state
                    
                    # ラプラシアンの計算と残差評価
                    laplacian = self._compute_laplacian(p, f)
                    residual = float(jnp.linalg.norm(laplacian))
                    
                    # 時間計測
                    current_time = time.time() - start_time
                    
                    # モニターへの記録
                    monitor.update(iteration, residual, p, current_time)
                    history['residual_history'].append(residual)
                    
                    # 収束判定
                    return jnp.logical_and(
                        residual >= self.tolerance,
                        iteration < self.max_iterations
                    )
                
                # while_loopによる反復
                final_state = jax.lax.while_loop(
                    convergence_condition, 
                    self._single_sor_iteration, 
                    (p, f, 0)
                )
                
                # 最終状態の展開
                p_solved, _, iterations = final_state
                
                # ラプラシアンと残差の計算
                laplacian = self._compute_laplacian(p_solved, f)
                final_residual = float(jnp.linalg.norm(laplacian))
                
                # 収束情報の構築
                history['iterations'] = int(iterations)
                history['final_residual'] = final_residual
                history['converged'] = final_residual < self.tolerance
                
                return p_solved, history
        
        # SORソルバの初期化と解法
        solver = MonitoredSORSolver(
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
        
        # 収束プロセスの可視化と要約
        monitor.visualize_convergence(
            source_term=f, 
            omega=1.5, 
            boundary_conditions=boundary_conditions
        )
        monitor.print_summary()
        
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
            print(f"  最終残差: {history['final_residual']}")
            
            self.assertTrue(history['converged'], 
                            f"{test_case['name']}で収束しませんでした")
            self.assertLess(history['final_residual'], 1e-5, 
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