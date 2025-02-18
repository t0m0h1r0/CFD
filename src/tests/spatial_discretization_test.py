import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Callable, Tuple

from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.cd import CentralDifferenceDiscretization
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition

class SpatialDiscretizationTestSuite:
    """空間離散化スキームのテストスイート"""
    
    @staticmethod
    def create_test_grid_manager(nx: int = 64, ny: int = 64) -> GridManager:
        """均一な試験用グリッドの作成"""
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(nx, ny, 1),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)
    
    @staticmethod
    def boundary_conditions() -> dict[str, BoundaryCondition]:
        """テスト用の境界条件の定義"""
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
        空間微分の精度テスト
        
        Args:
            discretization: 空間離散化スキーム
            test_func: 微分する関数
            derivative_func: 解析的微分関数
            direction: 微分方向
        
        Returns:
            相対誤差と誤差可視化図のタプル
        """
        # グリッドの作成
        grid_manager = cls.create_test_grid_manager()
        x, y, _ = grid_manager.get_coordinates()
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        # フィールドの計算
        field = test_func(X, Y)
        
        # 数値微分の計算
        numerical_deriv, _ = discretization.discretize(field, direction)
        
        # 解析的微分の計算
        analytical_deriv = derivative_func(X, Y)
        
        # 相対誤差の計算
        # 解析的微分がゼロの場合の処理を追加
        analytical_norm = jnp.linalg.norm(analytical_deriv)
        if analytical_norm == 0:
            error = jnp.linalg.norm(numerical_deriv)
        else:
            error = jnp.linalg.norm(numerical_deriv - analytical_deriv) / analytical_norm
        
        # 可視化
        plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォントの設定
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = ax1.pcolormesh(X, Y, numerical_deriv, shading='auto')
        ax1.set_title('数値微分')
        fig.colorbar(im1, ax=ax1)
        
        im2 = ax2.pcolormesh(X, Y, analytical_deriv, shading='auto')
        ax2.set_title('解析的微分')
        fig.colorbar(im2, ax=ax2)
        
        error_map = jnp.abs(numerical_deriv - analytical_deriv)
        im3 = ax3.pcolormesh(X, Y, error_map, shading='auto')
        ax3.set_title('絶対誤差')
        fig.colorbar(im3, ax=ax3)
        
        plt.suptitle(f'微分テスト - {direction.upper()}')
        plt.tight_layout()
        
        return float(error), fig
    
    @classmethod
    def run_tests(cls):
        """包括的な空間離散化テストの実行"""
        # 出力ディレクトリの作成
        os.makedirs('test_results/spatial_discretization', exist_ok=True)
        
        # テスト関数の定義
        def test_func1(x, y):
            """テスト関数1: sin(πx)sin(πy)"""
            return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
        
        def dx_test_func1(x, y):
            """テスト関数1の解析的x微分"""
            return jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)
        
        def dy_test_func1(x, y):
            """テスト関数1の解析的y微分"""
            return jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
        
        def test_func2(x, y):
            """テスト関数2: exp(x)cos(y)"""
            return jnp.exp(x) * jnp.cos(y)
        
        def dx_test_func2(x, y):
            """テスト関数2の解析的x微分"""
            return jnp.exp(x) * jnp.cos(y)
        
        def dy_test_func2(x, y):
            """テスト関数2の解析的y微分"""
            return -jnp.exp(x) * jnp.sin(y)
        
        # 離散化スキームの作成
        grid_manager = cls.create_test_grid_manager()
        discretization = CentralDifferenceDiscretization(
            grid_manager=grid_manager,
            boundary_conditions=cls.boundary_conditions()
        )
        
        # テストケースの定義
        test_cases = [
            ("Func1 X微分", test_func1, dx_test_func1, 'x'),
            ("Func1 Y微分", test_func1, dy_test_func1, 'y'),
            ("Func2 X微分", test_func2, dx_test_func2, 'x'),
            ("Func2 Y微分", test_func2, dy_test_func2, 'y')
        ]
        
        # テスト結果の保存
        test_results = {}
        
        # テストの実行
        for name, func, deriv_func, direction in test_cases:
            # テストの実行
            error, fig = cls.test_derivative_accuracy(
                discretization, func, deriv_func, direction
            )
            
            # 図の保存
            fig.savefig(f'test_results/spatial_discretization/{name.lower().replace(" ", "_")}.png')
            plt.close(fig)
            
            # 結果の記録
            test_results[name] = {
                'error': error,
                'passed': error < 1e-3  # 許容誤差を調整
            }
        
        # 結果の出力
        print("空間離散化テスト結果:")
        for name, result in test_results.items():
            status = "合格" if result['passed'] else "不合格"
            print(f"{name}: {status} (相対誤差: {result['error']:.6f})")
        
        return test_results

# スクリプトが直接実行された場合にテストを実行
if __name__ == '__main__':
    SpatialDiscretizationTestSuite.run_tests()