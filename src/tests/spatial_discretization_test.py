import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict

from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition


class SpatialDiscretizationTestSuite:
    """空間離散化スキームのテストスイート"""

    @staticmethod
    def create_test_grid_manager(
        nx: int = 64, ny: int = 64, nz: int = 64
    ) -> GridManager:
        """一様テストグリッドの生成"""
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0), points=(nx, ny, nz), grid_type=GridType.UNIFORM
        )
        return GridManager(grid_config)

    @staticmethod
    def boundary_conditions() -> Dict[str, BoundaryCondition]:
        """テスト用境界条件の定義"""
        return {
            "left": BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location="left"
            ),
            "right": BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location="right"
            ),
            "bottom": BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location="bottom"
            ),
            "top": BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location="top"),
            "front": BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location="front"
            ),
            "back": BoundaryCondition(
                type=BCType.DIRICHLET, value=0.0, location="back"
            ),
        }

    @classmethod
    def test_derivative_accuracy(
        cls,
        discretization: SpatialDiscretizationBase,
        test_func: Callable,
        derivative_func: Callable,
        direction: str,
    ) -> Tuple[float, plt.Figure]:
        """
        空間微分の精度テスト

        Args:
            discretization: 空間離散化スキーム
            test_func: 被微分関数
            derivative_func: 解析的な微分関数
            direction: 微分方向

        Returns:
            相対誤差と可視化図のタプル
        """
        # グリッド生成
        grid_manager = cls.create_test_grid_manager()
        x, y, z = grid_manager.get_coordinates()
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

        # フィールドの計算
        field = test_func(X, Y, Z)

        # 数値微分の計算
        numerical_deriv, _ = discretization.discretize(field, direction)

        # 解析的微分の計算
        analytical_deriv = derivative_func(X, Y, Z)

        # 相対誤差の計算
        error = jnp.linalg.norm(numerical_deriv - analytical_deriv) / jnp.linalg.norm(
            analytical_deriv
        )

        # 可視化（中央断面）
        mid_idx = field.shape[2] // 2  # Z方向の中央断面

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        im1 = ax1.pcolormesh(
            X[:, :, mid_idx],
            Y[:, :, mid_idx],
            numerical_deriv[:, :, mid_idx],
            shading="auto",
        )
        ax1.set_title(f"Numerical {direction.upper()} Derivative")
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.pcolormesh(
            X[:, :, mid_idx],
            Y[:, :, mid_idx],
            analytical_deriv[:, :, mid_idx],
            shading="auto",
        )
        ax2.set_title(f"Analytical {direction.upper()} Derivative")
        fig.colorbar(im2, ax=ax2)

        error_map = jnp.abs(numerical_deriv - analytical_deriv)[:, :, mid_idx]
        im3 = ax3.pcolormesh(
            X[:, :, mid_idx], Y[:, :, mid_idx], error_map, shading="auto"
        )
        ax3.set_title(f"Absolute Error ({direction.upper()} Derivative)")
        fig.colorbar(im3, ax=ax3)

        plt.suptitle(f"Derivative Test - {direction.upper()}")
        plt.tight_layout()

        return error, fig

    @classmethod
    def run_tests(cls):
        """包括的な空間離散化テストの実行"""
        # 出力ディレクトリの作成
        os.makedirs("test_results/spatial_discretization", exist_ok=True)

        # テスト関数群
        def test_func1(x, y, z):
            """テスト関数1: sin(πx)sin(πy)sin(πz)"""
            return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)

        def dx_test_func1(x, y, z):
            """テスト関数1のx方向微分"""
            return (
                jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
            )

        def dy_test_func1(x, y, z):
            """テスト関数1のy方向微分"""
            return (
                jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.sin(jnp.pi * z)
            )

        def dz_test_func1(x, y, z):
            """テスト関数1のz方向微分"""
            return (
                jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.cos(jnp.pi * z)
            )

        def test_func2(x, y, z):
            """テスト関数2: exp(x)cos(y)sin(z)"""
            return jnp.exp(x) * jnp.cos(y) * jnp.sin(z)

        def dx_test_func2(x, y, z):
            """テスト関数2のx方向微分"""
            return jnp.exp(x) * jnp.cos(y) * jnp.sin(z)

        def dy_test_func2(x, y, z):
            """テスト関数2のy方向微分"""
            return -jnp.exp(x) * jnp.sin(y) * jnp.sin(z)

        def dz_test_func2(x, y, z):
            """テスト関数2のz方向微分"""
            return jnp.exp(x) * jnp.cos(y) * jnp.cos(z)

        # 離散化スキームの生成
        grid_manager = cls.create_test_grid_manager()
        discretization = CombinedCompactDifference(
            grid_manager=grid_manager, boundary_conditions=cls.boundary_conditions()
        )

        # テストケースの定義
        test_cases = [
            ("Func1 X-Derivative", test_func1, dx_test_func1, "x"),
            ("Func1 Y-Derivative", test_func1, dy_test_func1, "y"),
            ("Func1 Z-Derivative", test_func1, dz_test_func1, "z"),
            ("Func2 X-Derivative", test_func2, dx_test_func2, "x"),
            ("Func2 Y-Derivative", test_func2, dy_test_func2, "y"),
            ("Func2 Z-Derivative", test_func2, dz_test_func2, "z"),
        ]

        # テスト結果の格納
        test_results = {}

        for name, func, deriv_func, direction in test_cases:
            # テスト実行
            error, fig = cls.test_derivative_accuracy(
                discretization, func, deriv_func, direction
            )

            # 図の保存
            fig.savefig(
                f"test_results/spatial_discretization/{name.lower().replace(' ', '_')}.png"
            )
            plt.close(fig)

            # 結果の保存
            test_results[name] = {
                "error": float(error),
                "passed": error < 1e-4,  # 許容誤差の閾値
            }

        # 結果の出力
        print("Spatial Discretization Test Results:")
        for name, result in test_results.items():
            status = "PASSED" if result["passed"] else "FAILED"
            print(f"{name}: {status} (Error: {result['error']:.6f})")

        return test_results


# スクリプト実行時のテスト実行
if __name__ == "__main__":
    SpatialDiscretizationTestSuite.run_tests()
