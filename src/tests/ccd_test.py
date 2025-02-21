from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition

@dataclass
class DerivativeTestCase:
    """微分計算のテストケース"""
    name: str
    function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    first_derivative: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    second_derivative: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

class DerivativeVisualization:
    """計算結果の可視化を担当"""
    
    @staticmethod
    def normalize_data(data: jnp.ndarray) -> np.ndarray:
        """データの正規化"""
        data_np = np.array(data)
        return (data_np - np.min(data_np)) / (np.max(data_np) - np.min(data_np) + 1e-10)
    
    @classmethod
    def create_comparative_visualization(
        cls,
        computed_first: jnp.ndarray,
        computed_second: jnp.ndarray,
        exact_first: jnp.ndarray,
        exact_second: jnp.ndarray,
        grid_size: int,
        test_name: str,
        direction: str
    ):
        """微分計算結果の比較可視化"""
        # 中心スライスの選択
        mid_idx = grid_size // 2
        
        # 相対誤差の計算
        error_first = jnp.abs(computed_first - exact_first) / (jnp.abs(exact_first) + 1e-10)
        error_second = jnp.abs(computed_second - exact_second) / (jnp.abs(exact_second) + 1e-10)
        
        # 可視化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Derivative Analysis: {test_name} ({direction}-direction)', fontsize=16)
        
        # 一階微分の可視化
        im1 = axes[0,0].plot(computed_first[mid_idx,:], label='Computed')
        axes[0,0].plot(exact_first[mid_idx,:], '--', label='Exact')
        axes[0,0].set_title('First Derivative')
        axes[0,0].legend()
        
        # 二階微分の可視化
        im2 = axes[1,0].plot(computed_second[mid_idx,:], label='Computed')
        axes[1,0].plot(exact_second[mid_idx,:], '--', label='Exact')
        axes[1,0].set_title('Second Derivative')
        axes[1,0].legend()
        
        # 誤差分布の可視化
        im3 = axes[0,1].pcolormesh(error_first, cmap='plasma')
        axes[0,1].set_title('First Derivative Error')
        plt.colorbar(im3, ax=axes[0,1])
        
        im4 = axes[1,1].pcolormesh(error_second, cmap='plasma')
        axes[1,1].set_title('Second Derivative Error')
        plt.colorbar(im4, ax=axes[1,1])
        
        # 相対誤差のヒストグラム
        axes[0,2].hist(error_first.flatten(), bins=50)
        axes[0,2].set_title('First Derivative Error Distribution')
        axes[0,2].set_yscale('log')
        
        axes[1,2].hist(error_second.flatten(), bins=50)
        axes[1,2].set_title('Second Derivative Error Distribution')
        axes[1,2].set_yscale('log')
        
        # レイアウト調整と保存
        plt.tight_layout()
        plt.savefig(
            f'test_results/ccd/{test_name.lower().replace(" ", "_")}_{direction}_derivatives.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

class CCDTest:
    """CCDスキームのテストスイート"""
    
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.setup_test_environment()
        
    def setup_test_environment(self):
        """テスト環境のセットアップ"""
        # 出力ディレクトリの作成
        os.makedirs('test_results/ccd', exist_ok=True)
        
        # グリッド設定
        self.grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(self.grid_size, self.grid_size, 1),
            grid_type=GridType.UNIFORM
        )
        self.grid_manager = GridManager(self.grid_config)
        
        # 境界条件の設定
        self.boundary_conditions = {
            'left': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='left'),
            'right': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='right'),
            'bottom': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='bottom'),
            'top': BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='top')
        }
        
        # CCDスキームの初期化
        self.ccd = CombinedCompactDifference(
            grid_manager=self.grid_manager,
            boundary_conditions=self.boundary_conditions,
            order=6
        )
        
        # テストケースの定義
        self.test_cases = [
            DerivativeTestCase(
                name="Sine Function",
                function=lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y),
                first_derivative=lambda x, y: 2 * jnp.pi * jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y),
                second_derivative=lambda x, y: -4 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
            ),
            DerivativeTestCase(
                name="Gaussian",
                function=lambda x, y: jnp.exp(-(x-0.5)**2/0.1 - (y-0.5)**2/0.1),
                first_derivative=lambda x, y: -2*(x-0.5)/0.1 * jnp.exp(-(x-0.5)**2/0.1 - (y-0.5)**2/0.1),
                second_derivative=lambda x, y: (-2/0.1 + 4*(x-0.5)**2/0.1**2) * jnp.exp(-(x-0.5)**2/0.1 - (y-0.5)**2/0.1)
            ),
            DerivativeTestCase(
                name="Polynomial",
                function=lambda x, y: x**3 * y**2,
                first_derivative=lambda x, y: 3 * x**2 * y**2,
                second_derivative=lambda x, y: 6 * x * y**2
            )
        ]
    
    def run_derivative_tests(self) -> Dict[str, Dict[str, float]]:
        """微分計算テストの実行"""
        results = {}
        
        # グリッドの生成
        x = jnp.linspace(0, 1, self.grid_size)
        y = jnp.linspace(0, 1, self.grid_size)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        for case in self.test_cases:
            case_results = {}
            print(f"\nTesting: {case.name}")
            
            # テスト関数と厳密解の計算
            field = case.function(X, Y)
            exact_first_x = case.first_derivative(X, Y)
            exact_second_x = case.second_derivative(X, Y)
            
            # x方向の数値微分
            computed_first_x, computed_second_x = self.ccd.discretize(field, 'x')
            
            # 相対誤差の計算
            error_first_x = jnp.linalg.norm(computed_first_x - exact_first_x) / jnp.linalg.norm(exact_first_x)
            error_second_x = jnp.linalg.norm(computed_second_x - exact_second_x) / jnp.linalg.norm(exact_second_x)
            
            # 結果の記録
            case_results.update({
                'first_derivative_error_x': float(error_first_x),
                'second_derivative_error_x': float(error_second_x)
            })
            
            # 結果の可視化
            DerivativeVisualization.create_comparative_visualization(
                computed_first_x,
                computed_second_x,
                exact_first_x,
                exact_second_x,
                self.grid_size,
                case.name,
                'x'
            )
            
            # y方向でも同様のテストを実行
            exact_first_y = case.first_derivative(Y, X)  # 引数を入れ替えて y 方向の微分
            exact_second_y = case.second_derivative(Y, X)
            
            computed_first_y, computed_second_y = self.ccd.discretize(field, 'y')
            
            error_first_y = jnp.linalg.norm(computed_first_y - exact_first_y) / jnp.linalg.norm(exact_first_y)
            error_second_y = jnp.linalg.norm(computed_second_y - exact_second_y) / jnp.linalg.norm(exact_second_y)
            
            case_results.update({
                'first_derivative_error_y': float(error_first_y),
                'second_derivative_error_y': float(error_second_y)
            })
            
            DerivativeVisualization.create_comparative_visualization(
                computed_first_y,
                computed_second_y,
                exact_first_y,
                exact_second_y,
                self.grid_size,
                case.name,
                'y'
            )
            
            results[case.name] = case_results
            
            # 結果の出力
            print(f"  X-direction:")
            print(f"    First derivative error: {error_first_x:.2e}")
            print(f"    Second derivative error: {error_second_x:.2e}")
            print(f"  Y-direction:")
            print(f"    First derivative error: {error_first_y:.2e}")
            print(f"    Second derivative error: {error_second_y:.2e}")
        
        return results

def main():
    """メインテスト実行関数"""
    # 異なるグリッドサイズでテストを実行
    grid_sizes = [32, 64, 128]
    all_results = {}
    
    for grid_size in grid_sizes:
        print(f"\nRunning tests with grid size: {grid_size}")
        test_suite = CCDTest(grid_size=grid_size)
        results = test_suite.run_derivative_tests()
        all_results[grid_size] = results
    
    # 収束性の解析
    print("\nConvergence Analysis:")
    for case_name in all_results[grid_sizes[0]].keys():
        print(f"\n{case_name}:")
        for error_type in ['first_derivative_error_x', 'second_derivative_error_x',
                          'first_derivative_error_y', 'second_derivative_error_y']:
            errors = [all_results[n][case_name][error_type] for n in grid_sizes]
            convergence_rates = [np.log2(errors[i]/errors[i+1]) 
                               for i in range(len(errors)-1)]
            
            print(f"  {error_type}:")
            print(f"    Errors: {', '.join(f'{e:.2e}' for e in errors)}")
            print(f"    Convergence rates: {', '.join(f'{r:.2f}' for r in convergence_rates)}")

if __name__ == '__main__':
    main()