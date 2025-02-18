import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass

from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BoundaryCondition, BCType

class SpatialDiscretizationTestSuite:
    """空間離散化スキームのテストスイート"""
    
    @staticmethod
    def create_test_grid_manager(nx: int = 64, ny: int = 64, nz: int = 64) -> GridManager:
        """均一なテスト格子を生成"""
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(nx, ny, nz),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)
    
    @staticmethod
    def create_test_functions() -> dict:
        """テスト関数とその厳密な微分を定義"""
        functions = {}
        
        # テスト関数1: sin(πx)sin(πy)sin(πz)
        def f1(x, y, z):
            return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
            
        def df1_dx(x, y, z):
            return jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
            
        def d2f1_dx2(x, y, z):
            return -(jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
            
        functions['periodic'] = {
            'f': f1,
            'df_dx': df1_dx,
            'd2f_dx2': d2f1_dx2,
            'description': '三重正弦波 (周期境界条件テスト用)'
        }
        
        # テスト関数2: exp(-((x-0.5)²+(y-0.5)²+(z-0.5)²))
        def f2(x, y, z):
            return jnp.exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2))
            
        def df2_dx(x, y, z):
            return -2*(x-0.5) * f2(x, y, z)
            
        def d2f2_dx2(x, y, z):
            return (-2 + 4*(x-0.5)**2) * f2(x, y, z)
            
        functions['gaussian'] = {
            'f': f2,
            'df_dx': df2_dx,
            'd2f_dx2': d2f2_dx2,
            'description': 'ガウス関数 (Dirichlet境界条件テスト用)'
        }
        
        # テスト関数3: tanh(5(x-0.5))
        def f3(x, y, z):
            return jnp.tanh(5*(x-0.5))
            
        def df3_dx(x, y, z):
            return 5/(jnp.cosh(5*(x-0.5))**2)
            
        def d2f3_dx2(x, y, z):
            return -10*jnp.sinh(5*(x-0.5))/(jnp.cosh(5*(x-0.5))**3)
            
        functions['steep'] = {
            'f': f3,
            'df_dx': df3_dx,
            'd2f_dx2': d2f3_dx2,
            'description': '急勾配関数 (数値的安定性テスト用)'
        }
        
        return functions
    
    @classmethod
    def test_accuracy(cls,
                     discretization: SpatialDiscretizationBase,
                     test_function: dict,
                     direction: str = 'x') -> dict:
        """
        離散化スキームの精度をテスト
        
        Args:
            discretization: テスト対象の離散化スキーム
            test_function: テスト関数の辞書
            direction: テストする方向
            
        Returns:
            テスト結果の辞書
        """
        # 格子点での関数値を計算
        x, y, z = discretization.grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        field = test_function['f'](X, Y, Z)
        exact_deriv1 = test_function['df_dx'](X, Y, Z)
        exact_deriv2 = test_function['d2f_dx2'](X, Y, Z)
        
        # 数値微分を計算
        numerical_deriv1, numerical_deriv2 = discretization.discretize(field, direction)
        
        # 誤差を計算
        error1 = jnp.linalg.norm(numerical_deriv1 - exact_deriv1) / jnp.linalg.norm(exact_deriv1)
        error2 = jnp.linalg.norm(numerical_deriv2 - exact_deriv2) / jnp.linalg.norm(exact_deriv2)
        
        # 誤差の次数を推定
        order1, order2 = discretization.estimate_error_order(
            field, direction, exact_deriv1, exact_deriv2
        )
        
        return {
            'error_deriv1': float(error1),
            'error_deriv2': float(error2),
            'order_deriv1': order1,
            'order_deriv2': order2
        }
    
    @classmethod
    def test_symmetry(cls,
                     discretization: SpatialDiscretizationBase,
                     test_function: dict) -> dict:
        """
        離散化スキームの対称性をテスト
        
        Args:
            discretization: テスト対象の離散化スキーム
            test_function: テスト関数の辞書
            
        Returns:
            テスト結果の辞書
        """
        x, y, z = discretization.grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        field = test_function['f'](X, Y, Z)
        
        symmetry_results = {}
        for direction in ['x', 'y', 'z']:
            symmetry_results[direction] = discretization.check_symmetry(
                field, direction
            )
            
        return symmetry_results
    
    @classmethod
    def test_stability(cls,
                      discretization: SpatialDiscretizationBase,
                      test_function: dict) -> dict:
        """
        離散化スキームの安定性をテスト
        
        Args:
            discretization: テスト対象の離散化スキーム
            test_function: テスト関数の辞書
            
        Returns:
            テスト結果の辞書
        """
        x, y, z = discretization.grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        field = test_function['f'](X, Y, Z)
        
        # ノイズレベルを変えながらテスト
        noise_levels = [0.01, 0.05, 0.1]
        noise_results = {}
        
        for noise_level in noise_levels:
            # ノイズ付加
            noise = noise_level * jnp.random.normal(size=field.shape)
            noisy_field = field + noise
            
            # 微分を計算
            deriv1, deriv2 = discretization.discretize(noisy_field, 'x')
            
            # 真値との比較
            exact_deriv1 = test_function['df_dx'](X, Y, Z)
            exact_deriv2 = test_function['d2f_dx2'](X, Y, Z)
            
            error1 = jnp.linalg.norm(deriv1 - exact_deriv1) / jnp.linalg.norm(exact_deriv1)
            error2 = jnp.linalg.norm(deriv2 - exact_deriv2) / jnp.linalg.norm(exact_deriv2)
            
            noise_results[noise_level] = {
                'error_deriv1': float(error1),
                'error_deriv2': float(error2)
            }
            
        return noise_results
    
    @classmethod
    def visualize_results(cls,
                         discretization: SpatialDiscretizationBase,
                         test_function: dict,
                         results: dict,
                         output_dir: str) -> None:
        """
        テスト結果を可視化
        
        Args:
            discretization: テスト対象の離散化スキーム
            test_function: テスト関数の辞書
            results: テスト結果の辞書
            output_dir: 出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 精度の可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'accuracy' in results:
            accuracy = results['accuracy']
            grid_sizes = accuracy['grid_sizes']
            errors1 = accuracy['errors_deriv1']
            errors2 = accuracy['errors_deriv2']
            
            ax1.loglog(grid_sizes, errors1, 'o-', label='First derivative')
            ax1.loglog(grid_sizes, errors2, 's-', label='Second derivative')
            ax1.grid(True)
            ax1.set_xlabel('Grid size')
            ax1.set_ylabel('Relative error')
            ax1.legend()
            ax1.set_title('Convergence Test')
            
            # 理論的な収束率との比較
            ref_line1 = errors1[0] * (grid_sizes/grid_sizes[0])**(-discretization.order)
            ref_line2 = errors2[0] * (grid_sizes/grid_sizes[0])**(-discretization.order)
            ax1.loglog(grid_sizes, ref_line1, 'k--', alpha=0.5, label=f'{discretization.order}th order')
            ax1.loglog(grid_sizes, ref_line2, 'k--', alpha=0.5)
        
        # 2. 安定性の可視化
        if 'stability' in results:
            stability = results['stability']
            noise_levels = list(stability.keys())
            errors1 = [stability[nl]['error_deriv1'] for nl in noise_levels]
            errors2 = [stability[nl]['error_deriv2'] for nl in noise_levels]
            
            ax2.semilogy(noise_levels, errors1, 'o-', label='First derivative')
            ax2.semilogy(noise_levels, errors2, 's-', label='Second derivative')
            ax2.grid(True)
            ax2.set_xlabel('Noise level')
            ax2.set_ylabel('Relative error')
            ax2.legend()
            ax2.set_title('Stability Test')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_stability.png'), dpi=300)
        plt.close()
        
        # 3. 微分の可視化（2D断面）
        x, y, z = discretization.grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        field = test_function['f'](X, Y, Z)
        
        # 中央断面でのプロット
        mid_z = len(z) // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 元の場
        im0 = axes[0,0].pcolormesh(X[:,:,mid_z], Y[:,:,mid_z], field[:,:,mid_z], shading='auto')
        axes[0,0].set_title('Original Field')
        plt.colorbar(im0, ax=axes[0,0])
        
        # 数値微分
        deriv1, deriv2 = discretization.discretize(field, 'x')
        im1 = axes[0,1].pcolormesh(X[:,:,mid_z], Y[:,:,mid_z], deriv1[:,:,mid_z], shading='auto')
        axes[0,1].set_title('First Derivative (Numerical)')
        plt.colorbar(im1, ax=axes[0,1])
        
        # 厳密解との比較
        exact_deriv1 = test_function['df_dx'](X, Y, Z)
        error1 = deriv1 - exact_deriv1
        im2 = axes[0,2].pcolormesh(X[:,:,mid_z], Y[:,:,mid_z], error1[:,:,mid_z], shading='auto')
        axes[0,2].set_title('First Derivative Error')
        plt.colorbar(im2, ax=axes[0,2])
        
        # 二階微分
        im3 = axes[1,0].pcolormesh(X[:,:,mid_z], Y[:,:,mid_z], deriv2[:,:,mid_z], shading='auto')
        axes[1,0].set_title('Second Derivative (Numerical)')
        plt.colorbar(im3, ax=axes[1,0])
        
        # 厳密解との比較
        exact_deriv2 = test_function['d2f_dx2'](X, Y, Z)
        error2 = deriv2 - exact_deriv2
        im4 = axes[1,1].pcolormesh(X[:,:,mid_z], Y[:,:,mid_z], error2[:,:,mid_z], shading='auto')
        axes[1,1].set_title('Second Derivative Error')
        plt.colorbar(im4, ax=axes[1,1])
        
        # 相対誤差の分布
        relative_error = jnp.abs(error2) / (jnp.abs(exact_deriv2) + 1e-10)
        im5 = axes[1,2].pcolormesh(X[:,:,mid_z], Y[:,:,mid_z], relative_error[:,:,mid_z], 
                                  shading='auto', norm=plt.LogNorm())
        axes[1,2].set_title('Relative Error (Second Derivative)')
        plt.colorbar(im5, ax=axes[1,2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'derivatives_visualization.png'), dpi=300)
        plt.close()
    
    @classmethod
    def run_comprehensive_tests(cls) -> dict:
        """
        包括的なテストスイートを実行
        
        Returns:
            全テスト結果の辞書
        """
        # 出力ディレクトリの作成
        output_dir = 'test_results/spatial_discretization'
        os.makedirs(output_dir, exist_ok=True)
        
        # グリッドマネージャとスキームの生成
        grid_manager = cls.create_test_grid_manager()
        discretization = CombinedCompactDifference(grid_manager)
        
        # テスト関数の取得
        test_functions = cls.create_test_functions()
        
        # 結果を格納する辞書
        all_results = {}
        
        # 各テスト関数に対してテストを実行
        for name, test_function in test_functions.items():
            print(f"\nTesting with {name} function...")
            
            # 精度テスト
            accuracy_results = cls.test_accuracy(discretization, test_function)
            print(f"Accuracy test completed:")
            print(f"  First derivative error: {accuracy_results['error_deriv1']:.2e}")
            print(f"  Second derivative error: {accuracy_results['error_deriv2']:.2e}")
            
            # 対称性テスト
            symmetry_results = cls.test_symmetry(discretization, test_function)
            print("Symmetry test completed:")
            for direction, passed in symmetry_results.items():
                print(f"  {direction}-direction: {'PASSED' if passed else 'FAILED'}")
            
            # 安定性テスト
            stability_results = cls.test_stability(discretization, test_function)
            print("Stability test completed:")
            for noise_level, errors in stability_results.items():
                print(f"  Noise level {noise_level}:")
                print(f"    First derivative error: {errors['error_deriv1']:.2e}")
                print(f"    Second derivative error: {errors['error_deriv2']:.2e}")
            
            # 結果の格納
            function_results = {
                'accuracy': accuracy_results,
                'symmetry': symmetry_results,
                'stability': stability_results
            }
            
            all_results[name] = function_results
            
            # 結果の可視化
            cls.visualize_results(
                discretization,
                test_function,
                function_results,
                os.path.join(output_dir, name)
            )
        
        return all_results

if __name__ == '__main__':
    results = SpatialDiscretizationTestSuite.run_comprehensive_tests()