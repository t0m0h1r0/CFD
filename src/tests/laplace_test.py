from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

from src.core.spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BCType, BoundaryCondition


class LaplacianVisualization:
    """Laplacian計算結果の可視化を担当するクラス"""

    @staticmethod
    def normalize_data(data: jnp.ndarray) -> np.ndarray:
        """
        データを可視化に適したスケールに正規化

        Args:
            data: 入力データ

        Returns:
            正規化されたデータ
        """
        data_np = np.array(data)
        return (data_np - np.min(data_np)) / (np.max(data_np) - np.min(data_np) + 1e-10)

    @classmethod
    def create_comparative_visualization(
        cls,
        computed_lap: jnp.ndarray,
        exact_lap: jnp.ndarray,
        grid_size: int,
        test_name: str,
    ):
        """
        ラプラシアン計算結果の比較可視化

        Args:
            computed_lap: 計算されたラプラシアン
            exact_lap: 解析的ラプラシアン
            grid_size: グリッドサイズ
            test_name: テスト名
        """
        # 中心スライスの選択
        mid_idx = grid_size // 2

        # データの準備
        computed_slice = computed_lap[mid_idx, :, :]
        exact_slice = exact_lap[mid_idx, :, :]
        error_slice = np.abs(computed_lap - exact_lap)[mid_idx, :, :]

        # 座標の準備
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)

        # 可視化
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Laplacian Analysis: {test_name} (Grid Size: {grid_size})", fontsize=16
        )

        # 計算されたラプラシアン
        im1 = axs[0].pcolormesh(X, Y, computed_slice, cmap="viridis", shading="auto")
        axs[0].set_title("Computed Laplacian")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        plt.colorbar(im1, ax=axs[0], label="Value")

        # 解析的ラプラシアン
        im2 = axs[1].pcolormesh(X, Y, exact_slice, cmap="viridis", shading="auto")
        axs[1].set_title("Exact Laplacian")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        plt.colorbar(im2, ax=axs[1], label="Value")

        # 絶対誤差
        im3 = axs[2].pcolormesh(X, Y, error_slice, cmap="plasma", shading="auto")
        axs[2].set_title("Absolute Error")
        axs[2].set_xlabel("X")
        axs[2].set_ylabel("Y")
        plt.colorbar(im3, ax=axs[2], label="|Computed - Exact|")

        # レイアウト調整と保存
        plt.tight_layout()

        # 結果保存用のディレクトリ作成
        os.makedirs("test_results/laplacian", exist_ok=True)
        plt.savefig(
            f"test_results/laplacian/{test_name.lower().replace(' ', '_')}_grid{grid_size}_visualization.png",
            dpi=300,
        )
        plt.close()


@dataclass
class TestFunction:
    """解析関数とその微分の抽象化"""

    name: str
    function: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    laplacian: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


class ErrorMetric(ABC):
    """誤差計算の抽象基底クラス"""

    @abstractmethod
    def compute(self, computed: jnp.ndarray, exact: jnp.ndarray) -> float:
        """誤差計算の抽象メソッド"""
        pass


class RelativeRMSEErrorMetric(ErrorMetric):
    """相対二乗平均平方根誤差の計算"""

    def compute(self, computed: jnp.ndarray, exact: jnp.ndarray) -> float:
        squared_error = jnp.mean((computed - exact) ** 2)
        exact_squared_mean = jnp.mean(exact**2)

        if exact_squared_mean == 0:
            return float("inf")

        return float(jnp.sqrt(squared_error) / jnp.sqrt(exact_squared_mean))


class LaplaceTestConfiguration:
    """ラプラシアンテストの設定管理"""

    def __init__(
        self,
        grid_sizes: List[int] = [32, 64, 128],
        boundary_conditions: Dict[str, BoundaryCondition] = None,
        order: int = 8,
    ):
        self.grid_sizes = grid_sizes
        self.boundary_conditions = (
            boundary_conditions or self._default_boundary_conditions()
        )
        self.order = order

    def _default_boundary_conditions(self) -> Dict[str, BoundaryCondition]:
        """デフォルトの境界条件を生成"""
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


class LaplacianTestSuite:
    """ラプラシアンソルバーのテストスイート"""

    def __init__(
        self,
        test_functions: List[TestFunction],
        config: LaplaceTestConfiguration,
        error_metric: ErrorMetric = RelativeRMSEErrorMetric(),
    ):
        self.test_functions = test_functions
        self.config = config
        self.error_metric = error_metric

    def _create_grid_manager(self, grid_size: int) -> GridManager:
        """グリッドマネージャーを生成"""
        grid_config = GridConfig(
            dimensions=(1.0, 1.0, 1.0),
            points=(grid_size, grid_size, grid_size),
            grid_type=GridType.UNIFORM,
        )
        return GridManager(grid_config)

    def _create_3d_grid(
        self, grid_size: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """3Dグリッドを生成"""
        x = jnp.linspace(0, 1, grid_size)
        y = jnp.linspace(0, 1, grid_size)
        z = jnp.linspace(0, 1, grid_size)
        return jnp.meshgrid(x, y, z, indexing="ij")

    def run_tests(self) -> Dict[str, Dict[int, float]]:
        """テストの実行"""
        results = {}

        for test_func in self.test_functions:
            grid_results = {}

            for grid_size in self.config.grid_sizes:
                # グリッドとソルバーの準備
                grid_manager = self._create_grid_manager(grid_size)
                laplacian_solver = CCDLaplacianSolver(
                    grid_manager=grid_manager,
                    boundary_conditions=self.config.boundary_conditions,
                    order=self.config.order,
                )

                # グリッドと解析解の準備
                X, Y, Z = self._create_3d_grid(grid_size)
                field = test_func.function(X, Y, Z)
                exact_lap = test_func.laplacian(X, Y, Z)

                # ラプラシアンの計算
                computed_lap = laplacian_solver.compute_laplacian(field)

                # 誤差計算
                relative_error = self.error_metric.compute(computed_lap, exact_lap)
                grid_results[grid_size] = relative_error

                LaplacianVisualization.create_comparative_visualization(
                    computed_lap, exact_lap, grid_size, test_func.name
                )
            results[test_func.name] = grid_results

        return results

    def print_results(self, results: Dict[str, Dict[int, float]]):
        """テスト結果の出力"""
        print("Laplacian Solver Test Results:")
        for func_name, grid_results in results.items():
            print(f"\n{func_name}:")
            for grid_size, error in grid_results.items():
                print(f"  Grid Size: {grid_size}")
                print(f"    Relative Error: {error:.6e}")


def default_test_functions() -> List[TestFunction]:
    """デフォルトのテスト関数群"""
    return [
        TestFunction(
            name="Quadratic Function Laplacian",
            function=lambda x, y, z: x**2 + y**2 + z**2,
            laplacian=lambda x, y, z: 2 * jnp.ones_like(x)
            + 2 * jnp.ones_like(y)
            + 2 * jnp.ones_like(x),
        ),
        TestFunction(
            name="Sine Function Laplacian",
            function=lambda x, y, z: jnp.sin(jnp.pi * x)
            * jnp.sin(jnp.pi * y)
            * jnp.sin(jnp.pi * z),
            laplacian=lambda x, y, z: -3
            * (jnp.pi**2)
            * jnp.sin(jnp.pi * x)
            * jnp.sin(jnp.pi * y)
            * jnp.sin(jnp.pi * z),
        ),
        TestFunction(
            name="Exponential Function Laplacian",
            function=lambda x, y, z: jnp.exp(x + y + z),
            laplacian=lambda x, y, z: 3 * jnp.exp(x + y + z),
        ),
    ]


def main():
    """メイン実行関数"""
    config = LaplaceTestConfiguration()
    test_suite = LaplacianTestSuite(
        test_functions=default_test_functions(), config=config
    )

    results = test_suite.run_tests()
    test_suite.print_results(results)


if __name__ == "__main__":
    main()
