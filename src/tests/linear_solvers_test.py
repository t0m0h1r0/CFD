import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict, Optional

from src.core.linear_solvers.iterative.sor import SORSolver
from src.core.linear_solvers.iterative.cg import ConjugateGradientSolver
from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BoundaryCondition, BCType

class LinearSolversTestSuite:
    """線形ソルバーのテストスイート"""
    
    @classmethod
    def _create_default_grid_manager(
        cls, 
        matrix_size: int
    ) -> GridManager:
        """
        デフォルトのグリッドマネージャーを作成
        
        Args:
            matrix_size: マトリクスのサイズ
        
        Returns:
            GridManagerインスタンス
        """
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(matrix_size, matrix_size, 1),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)
    
    @classmethod
    def create_test_matrix(
        cls, 
        n: int, 
        condition_number: float = 10.0, 
        symmetric: bool = True,
        key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        テスト用の対称正定値行列を生成
        
        Args:
            n: 行列サイズ
            condition_number: 条件数
            symmetric: 対称行列かどうか
            key: JAXのランダムキー
        
        Returns:
            生成された行列
        """
        # ランダムキーの生成
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # ランダムな対称行列の生成
        A_rand = jax.random.normal(key, (n, n))
        A = A_rand @ A_rand.T
        
        # 固有値の計算
        evals, evecs = jnp.linalg.eigh(A)
        
        # 固有値の調整
        min_eval = 1.0
        max_eval = condition_number
        modified_evals = jnp.linspace(min_eval, max_eval, n)
        
        # 修正された固有値で行列を再構築
        return evecs @ jnp.diag(modified_evals) @ evecs.T
    
    @classmethod
    def solve_linear_system(
        cls,
        solver_class,
        matrix_size: int = 100,
        condition_number: float = 10.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        preconditioner: Optional[Callable] = None,
        key: Optional[jax.Array] = None,
        grid_manager: Optional[GridManager] = None
    ) -> Tuple[Dict, plt.Figure]:
        """
        線形システムの解法をテスト
        
        Args:
            solver_class: テストする線形ソルバークラス
            matrix_size: テスト行列のサイズ
            condition_number: 行列の条件数
            max_iterations: 最大反復回数
            tolerance: 収束許容誤差
            preconditioner: 前処理関数
            key: JAXのランダムキー
            grid_manager: グリッドマネージャー（オプション）
        
        Returns:
            テスト結果と収束履歴の図
        """
        # ランダムキーの生成
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # グリッドマネージャーの作成（未指定の場合）
        if grid_manager is None:
            grid_manager = cls._create_default_grid_manager(matrix_size)
        
        # テスト行列と右辺ベクトルの生成
        key, subkey1, subkey2 = jax.random.split(key, 3)
        A = cls.create_test_matrix(matrix_size, condition_number, key=subkey1)
        x_true = jax.random.normal(subkey2, (matrix_size,))
        b = A @ x_true
        
        # ソルバーの生成
        solver = solver_class(
            discretization=CombinedCompactDifference(grid_manager=grid_manager),
            max_iterations=max_iterations, 
            tolerance=tolerance
        )
        
        # 前処理が指定されている場合は適用
        if preconditioner:
            solver.preconditioner = preconditioner(A)
        
        # 初期推定解
        x0 = jnp.zeros_like(b)
        
        # システムを解く
        x_solved, history = solver.solve(A, b, x0)
        
        # エラー指標の計算
        residual = jnp.linalg.norm(b - A @ x_solved)
        relative_error = residual / jnp.linalg.norm(b)
        solution_error = jnp.linalg.norm(x_true - x_solved)
        
        # 収束履歴のプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'residual_history' in history and history['residual_history'] is not None:
            ax.semilogy(history['residual_history'], '-o')
            ax.set_xlabel('反復回数')
            ax.set_ylabel('残差ノルム')
            ax.set_title(f'{solver_class.__name__} の収束')
            ax.grid(True)
        
        # 結果の辞書
        results = {
            'matrix_size': matrix_size,
            'condition_number': condition_number,
            'residual': float(residual),
            'relative_error': float(relative_error),
            'solution_error': float(solution_error),
            'iterations': history.get('iterations', 0),
            'converged': history.get('converged', False)
        }
        
        return results, fig
    
    @classmethod
    def run_tests(cls):
        """
        線形ソルバーの包括的なテスト
        
        Returns:
            全テスト結果の辞書
        """
        # 出力ディレクトリの作成
        os.makedirs('test_results/linear_solvers', exist_ok=True)
        
        # テストするソルバー
        solvers = [
            SORSolver,
            ConjugateGradientSolver
        ]
        
        # テスト設定
        test_configs = [
            {'matrix_size': 50, 'condition_number': 10.0},
            {'matrix_size': 100, 'condition_number': 100.0},
            {'matrix_size': 200, 'condition_number': 1000.0}
        ]
        
        # テスト結果の保存
        all_results = {}
        
        # テストの実行
        for solver in solvers:
            solver_results = []
            
            for config in test_configs:
                # デフォルトのグリッドマネージャーを作成
                grid_manager = cls._create_default_grid_manager(config['matrix_size'])
                
                # 追加の前処理オプション
                preconditioners = [None]
                
                # CGソルバーの場合は追加の前処理を考慮
                if solver == ConjugateGradientSolver:
                    preconditioners.append(
                        ConjugateGradientSolver.diagonal_preconditioner
                    )
                    preconditioners.append(
                        ConjugateGradientSolver.symmetric_sor_preconditioner
                    )
                
                # 各前処理オプションでテスト
                for preconditioner in preconditioners:
                    # テストの実行
                    result, fig = cls.solve_linear_system(
                        solver, 
                        matrix_size=config['matrix_size'], 
                        condition_number=config['condition_number'],
                        preconditioner=preconditioner,
                        grid_manager=grid_manager
                    )
                    
                    # 図の保存
                    fig_filename = (f'test_results/linear_solvers/'
                                   f'{solver.__name__.lower()}_'
                                   f'{config["matrix_size"]}x{config["matrix_size"]}_'
                                   f'cond{config["condition_number"]}_'
                                   f'{"preconditioned" if preconditioner else "standard"}.png')
                    plt.savefig(fig_filename)
                    plt.close(fig)
                    
                    # 結果の保存
                    result['preconditioner'] = (
                        preconditioner.__name__ if preconditioner else 'None'
                    )
                    solver_results.append(result)
            
            # ソルバーごとの結果を保存
            all_results[solver.__name__] = solver_results
        
        # 結果の出力
        print("線形ソルバーテスト結果:")
        for solver_name, results in all_results.items():
            print(f"\n{solver_name}:")
            for result in results:
                print(f"  マトリクスサイズ: {result['matrix_size']}")
                print(f"    条件数: {result['condition_number']}")
                print(f"    前処理: {result['preconditioner']}")
                print(f"    残差: {result['residual']:.6e}")
                print(f"    相対誤差: {result['relative_error']:.6e}")
                print(f"    解誤差: {result['solution_error']:.6e}")
                print(f"    反復回数: {result['iterations']}")
                print(f"    収束: {result['converged']}")
        
        return all_results

# スクリプトが直接実行された場合にテストを実行
if __name__ == '__main__':
    LinearSolversTestSuite.run_tests()