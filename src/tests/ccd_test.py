import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional, Tuple

# 以下のインポートは、プロジェクト内の実装に合わせた相対パスとなっています。
from src.core.spatial_discretization.operators.ccd import CCDCompactDifference
from src.core.common.grid import GridManager, GridConfig, GridType
from src.core.common.types import BCType, BoundaryCondition

class CCDTestFunction:
    """
    CCDCompactDifference のテスト関数群。
    各関数は解析的な微分解（1階～3階）も合わせて提供します。
    """
    @staticmethod
    def harmonic_function(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(jnp.pi * x)

    @staticmethod
    def harmonic_first_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.pi * jnp.cos(jnp.pi * x)

    @staticmethod
    def harmonic_second_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return -jnp.pi**2 * jnp.sin(jnp.pi * x)

    @staticmethod
    def harmonic_third_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return -jnp.pi**3 * jnp.cos(jnp.pi * x)

    @staticmethod
    def exponential_function(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x)

    @staticmethod
    def exponential_first_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x)

    @staticmethod
    def exponential_second_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x)

    @staticmethod
    def exponential_third_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x)

    @staticmethod
    def polynomial_function(x: jnp.ndarray) -> jnp.ndarray:
        return x**3 + x**2 + x

    @staticmethod
    def polynomial_first_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return 3*x**2 + 2*x + 1

    @staticmethod
    def polynomial_second_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return 6*x + 2

    @staticmethod
    def polynomial_third_derivative(x: jnp.ndarray) -> jnp.ndarray:
        return 6 * jnp.ones_like(x)


class CCDTestSuite:
    """
    CCDCompactDifference の精度と収束特性を評価するテストスイート。
    
    ・GridManager は 1 次元（x軸）のグリッドを作成しますが、GridConfig は
      他の軸も含む（例: (dim, 1.0, 1.0)）形で初期化しています。
    ・各テストケースでは、テスト関数とその解析的微分解を用い、グリッド解像度を変化させた場合の
      相対誤差から収束次数を算出します。
    """
    
    @classmethod
    def create_grid_manager(cls, points: int = 64, dimension: float = 1.0) -> GridManager:
        """
        1D テスト用の GridManager を生成。
        GridConfig では x 軸に対して points 個、その他の軸は 1 点としています。
        """
        grid_config = GridConfig(
            dimensions=(dimension, 1.0, 1.0),
            points=(points, 1, 1),
            grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)

    @classmethod
    def create_boundary_conditions(cls, bc_type: BCType = BCType.DIRICHLET) -> Dict[str, BoundaryCondition]:
        """
        境界条件の辞書を生成。
        CCDCompactDifference では、x軸の 'left' と 'right' を利用しますが、
        他軸も設定可能なよう全体を返します。
        """
        return {
            'left': BoundaryCondition(type=bc_type, value=0.0, location='left'),
            'right': BoundaryCondition(type=bc_type, value=0.0, location='right'),
            'bottom': BoundaryCondition(type=bc_type, value=0.0, location='bottom'),
            'top': BoundaryCondition(type=bc_type, value=0.0, location='top'),
            'front': BoundaryCondition(type=bc_type, value=0.0, location='front'),
            'back': BoundaryCondition(type=bc_type, value=0.0, location='back')
        }

    @classmethod
    def evaluate_derivative_accuracy(
        cls,
        function: Callable,
        derivative_func: Callable,
        derivative_order: int = 1,
        points_range: list = [32, 64, 128, 256],
        bc_type: BCType = BCType.DIRICHLET
    ) -> Dict[int, float]:
        """
        指定関数とその解析微分関数に対して、CCDCompactDifference の相対誤差を評価する。
        
        CCDCompactDifference の compute_derivatives() により (D1, D2, D3) を返すため、
        derivative_order に応じて該当の微分結果と解析解を比較します。
        """
        accuracy_results = {}
        for points in points_range:
            grid_manager = cls.create_grid_manager(points)
            bc = cls.create_boundary_conditions(bc_type)
            # x軸方向のソルバを生成
            ccd_solver = CCDCompactDifference(
                grid_manager=grid_manager,
                direction='x',
                boundary_conditions=bc
            )
            # 1D のグリッド生成
            x = jnp.linspace(0, 1, points)
            field = function(x)
            exact = derivative_func(x)
            # CCDCompactDifference では compute_derivatives() で (D1, D2, D3) を返す
            derivatives = ccd_solver.compute_derivatives(field)
            computed = derivatives[derivative_order - 1]
            rel_error = jnp.linalg.norm(computed - exact) / (jnp.linalg.norm(exact) + 1e-10)
            accuracy_results[points] = float(rel_error)
        return accuracy_results

    @classmethod
    def compute_convergence_rate(cls, accuracy_results: Dict[int, float]) -> float:
        """
        精度評価結果から収束次数を算出する。
        グリッドポイント数と相対誤差の対数プロットに線形回帰を行い、傾きを収束次数とします。
        """
        pts = jnp.array(list(accuracy_results.keys()), dtype=jnp.float32)
        errs = jnp.array(list(accuracy_results.values()), dtype=jnp.float32)
        log_pts = jnp.log(pts)
        log_errs = jnp.log(errs)
        slope, _ = jnp.polyfit(log_pts, log_errs, 1)
        return float(-slope)

    @classmethod
    def plot_convergence(
        cls,
        accuracy_results: Dict[int, float],
        title: str,
        derivative_order: int,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        収束特性（グリッド解像度 vs 相対誤差）のプロットを作成する。
        """
        plt.figure(figsize=(10, 6))
        pts = list(accuracy_results.keys())
        errs = list(accuracy_results.values())
        plt.loglog(pts, errs, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('Number of grid points', fontsize=12)
        plt.ylabel('Relative error', fontsize=12)
        plt.title(f'{title} (Derivative order: {derivative_order})', fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        conv_rate = cls.compute_convergence_rate(accuracy_results)
        plt.text(0.05, 0.95, f'Convergence rate: {conv_rate:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top')
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return plt.gcf()

    @classmethod
    def run_comprehensive_tests(cls, derivative_orders: Optional[Tuple[int, ...]] = None):
        """
        包括的なテストを実行し、各関数・微分階数ごとに相対誤差と収束次数を計算、プロットを出力する。
        """
        os.makedirs("test_results/ccd_solver", exist_ok=True)
        if derivative_orders is None:
            derivative_orders = (1, 2, 3)
        test_cases = [
            {
                'name': 'Harmonic Function',
                'function': CCDTestFunction.harmonic_function,
                'derivatives': [
                    CCDTestFunction.harmonic_first_derivative,
                    CCDTestFunction.harmonic_second_derivative,
                    CCDTestFunction.harmonic_third_derivative
                ]
            },
            {
                'name': 'Exponential Function',
                'function': CCDTestFunction.exponential_function,
                'derivatives': [
                    CCDTestFunction.exponential_first_derivative,
                    CCDTestFunction.exponential_second_derivative,
                    CCDTestFunction.exponential_third_derivative
                ]
            },
            {
                'name': 'Polynomial Function',
                'function': CCDTestFunction.polynomial_function,
                'derivatives': [
                    CCDTestFunction.polynomial_first_derivative,
                    CCDTestFunction.polynomial_second_derivative,
                    CCDTestFunction.polynomial_third_derivative
                ]
            }
        ]
        all_results = {}
        for case in test_cases:
            case_results = {}
            for order in derivative_orders:
                acc_results = cls.evaluate_derivative_accuracy(
                    function=case['function'],
                    derivative_func=case['derivatives'][order - 1],
                    derivative_order=order
                )
                conv_rate = cls.compute_convergence_rate(acc_results)
                case_results[order] = {
                    'accuracy_results': acc_results,
                    'convergence_rate': conv_rate
                }
                cls.plot_convergence(
                    accuracy_results=acc_results,
                    title=case['name'],
                    derivative_order=order,
                    output_path=f"test_results/ccd_solver/{case['name'].lower().replace(' ', '_')}_{order}_derivative.png"
                )
            all_results[case['name']] = case_results
        return all_results


if __name__ == '__main__':
    results = CCDTestSuite.run_comprehensive_tests()
    for func_name, func_results in results.items():
        print(f"\n{func_name}:")
        for order, order_results in func_results.items():
            print(f"  Order {order} derivative:")
            print(f"    Convergence rate: {order_results['convergence_rate']:.4f}")
            print("    Grid points and relative errors:")
            for pts, err in order_results['accuracy_results'].items():
                print(f"      {pts}: {err:.6e}")
