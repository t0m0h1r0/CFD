"""
2次元CCD法のテストモジュール

簡単なテスト関数を使用して2次元CCD法の精度を検証します。
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Dict, Callable, Tuple, List, Any

from grid2d_config import Grid2DConfig
from ccd2d_solver import CCD2DSolver


class TestFunction2D:
    """2次元テスト関数を定義するクラス"""

    def __init__(
        self,
        name: str,
        f: Callable[[float, float], float],
        f_x: Callable[[float, float], float],
        f_y: Callable[[float, float], float],
        f_xx: Callable[[float, float], float],
        f_yy: Callable[[float, float], float],
        f_xy: Callable[[float, float], float],
    ):
        """
        2次元テスト関数の初期化

        Args:
            name: 関数の名前
            f: 関数値
            f_x: x方向の1階偏導関数
            f_y: y方向の1階偏導関数
            f_xx: x方向の2階偏導関数
            f_yy: y方向の2階偏導関数
            f_xy: 混合偏導関数
        """
        self.name = name
        self.f = f
        self.f_x = f_x
        self.f_y = f_y
        self.f_xx = f_xx
        self.f_yy = f_yy
        self.f_xy = f_xy


def create_test_functions() -> List[TestFunction2D]:
    """テスト関数のリストを作成して返す"""
    test_functions = []

    # テスト関数1: 2次元ガウス関数
    def gaussian(x, y):
        return cp.exp(-(x**2 + y**2))

    def gaussian_x(x, y):
        return -2 * x * cp.exp(-(x**2 + y**2))

    def gaussian_y(x, y):
        return -2 * y * cp.exp(-(x**2 + y**2))

    def gaussian_xx(x, y):
        return (-2 + 4 * x**2) * cp.exp(-(x**2 + y**2))

    def gaussian_yy(x, y):
        return (-2 + 4 * y**2) * cp.exp(-(x**2 + y**2))

    def gaussian_xy(x, y):
        return 4 * x * y * cp.exp(-(x**2 + y**2))

    test_functions.append(
        TestFunction2D(
            "Gaussian",
            gaussian,
            gaussian_x,
            gaussian_y,
            gaussian_xx,
            gaussian_yy,
            gaussian_xy,
        )
    )

    # テスト関数2: 2次元サイン関数
    def sin_function(x, y):
        return cp.sin(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_x(x, y):
        return cp.pi * cp.cos(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_y(x, y):
        return cp.pi * cp.sin(cp.pi * x) * cp.cos(cp.pi * y)

    def sin_function_xx(x, y):
        return -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_yy(x, y):
        return -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_xy(x, y):
        return (cp.pi**2) * cp.cos(cp.pi * x) * cp.cos(cp.pi * y)

    test_functions.append(
        TestFunction2D(
            "Sine",
            sin_function,
            sin_function_x,
            sin_function_y,
            sin_function_xx,
            sin_function_yy,
            sin_function_xy,
        )
    )

    # テスト関数3: 多項式関数
    def poly_function(x, y):
        return x**3 * y**2 + x * y**3

    def poly_function_x(x, y):
        return 3 * x**2 * y**2 + y**3

    def poly_function_y(x, y):
        return 2 * x**3 * y + 3 * x * y**2

    def poly_function_xx(x, y):
        return 6 * x * y**2

    def poly_function_yy(x, y):
        return 2 * x**3 + 6 * x * y

    def poly_function_xy(x, y):
        return 6 * x**2 * y + 3 * y**2

    test_functions.append(
        TestFunction2D(
            "Polynomial",
            poly_function,
            poly_function_x,
            poly_function_y,
            poly_function_xx,
            poly_function_yy,
            poly_function_xy,
        )
    )

    return test_functions


def evaluate_function_on_grid(
    func: Callable[[float, float], float], x_grid: cp.ndarray, y_grid: cp.ndarray
) -> cp.ndarray:
    """
    関数を格子点上で評価する

    Args:
        func: 評価する関数 (x, y) -> 値
        x_grid: x座標のメッシュグリッド
        y_grid: y座標のメッシュグリッド

    Returns:
        評価結果の2次元配列
    """
    nx, ny = x_grid.shape[0], y_grid.shape[0]
    result = cp.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            result[i, j] = func(x_grid[i], y_grid[j])

    return result


def compute_error(
    numerical: cp.ndarray, analytical: cp.ndarray, norm_type: str = "L2"
) -> float:
    """
    数値解と解析解の誤差を計算する

    Args:
        numerical: 数値解
        analytical: 解析解
        norm_type: 誤差のノルムタイプ ("L1", "L2", "Linf")

    Returns:
        誤差値
    """
    if norm_type == "L1":
        return float(cp.mean(cp.abs(numerical - analytical)))
    elif norm_type == "L2":
        return float(cp.sqrt(cp.mean((numerical - analytical) ** 2)))
    elif norm_type == "Linf":
        return float(cp.max(cp.abs(numerical - analytical)))
    else:
        raise ValueError(f"未知のノルムタイプ: {norm_type}")


def plot_results(
    x_grid: cp.ndarray,
    y_grid: cp.ndarray,
    numerical: Dict[str, cp.ndarray],
    analytical: Dict[str, cp.ndarray],
    function_name: str,
    save_path: str = None,
):
    """
    数値解と解析解の3Dプロットを作成

    Args:
        x_grid: x座標のメッシュグリッド
        y_grid: y座標のメッシュグリッド
        numerical: 数値解の辞書
        analytical: 解析解の辞書
        function_name: 関数の名前
        save_path: 保存先のパス（Noneの場合は保存しない）
    """
    # CuPyからNumPyに変換
    x_np = cp.asnumpy(x_grid)
    y_np = cp.asnumpy(y_grid)
    X, Y = np.meshgrid(x_np, y_np, indexing='ij')

    # プロット設定
    component_names = {
        "f": "Function",
        "f_x": "∂f/∂x",
        "f_y": "∂f/∂y",
        "f_xx": "∂²f/∂x²",
        "f_yy": "∂²f/∂y²",
        "f_xy": "∂²f/∂x∂y",
    }

    # 2×3のサブプロットを作成
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"2D CCD Results for {function_name} Function", fontsize=16)

    for i, comp in enumerate(list(component_names.keys())[:6]):  # 最大6成分
        if comp in numerical and comp in analytical:
            # 数値解と解析解をNumPy配列に変換
            Z_num = cp.asnumpy(numerical[comp])
            Z_ana = cp.asnumpy(analytical[comp])
            
            # 誤差計算
            error = compute_error(numerical[comp], analytical[comp], "L2")
            
            # サブプロット作成
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
            
            # ワイヤーフレームプロット
            ax.plot_wireframe(X, Y, Z_num, color='red', alpha=0.5, label='Numerical')
            ax.plot_wireframe(X, Y, Z_ana, color='blue', alpha=0.5, label='Analytical')
            
            ax.set_title(f"{component_names[comp]} (L2 Error: {error:.2e})")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def run_test(
    nx: int = 20,
    ny: int = 20,
    domain: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    use_iterative: bool = True,
    solver_kwargs: Dict[str, Any] = None,
    visualize: bool = True,
):
    """
    2次元CCD法のテストを実行する

    Args:
        nx: x方向のグリッド点数
        ny: y方向のグリッド点数
        domain: 計算領域 (x_min, x_max, y_min, y_max)
        use_iterative: 反復法を使用するかどうか
        solver_kwargs: ソルバーのパラメータ
        visualize: 結果を可視化するかどうか
    """
    # デフォルトのソルバーパラメータ
    if solver_kwargs is None:
        solver_kwargs = {"solver_type": "gmres", "tol": 1e-10, "maxiter": 1000}

    # 計算領域とグリッド設定
    x_min, x_max, y_min, y_max = domain
    hx = (x_max - x_min) / (nx - 1)
    hy = (y_max - y_min) / (ny - 1)

    # グリッド作成
    x_grid = cp.linspace(x_min, x_max, nx)
    y_grid = cp.linspace(y_min, y_max, ny)

    # グリッド設定
    grid_config = Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        x_deriv_order=2,  # 2次までの導関数を考慮
        y_deriv_order=2,
        mixed_deriv_order=1,
    )

    # テスト関数の取得
    test_functions = create_test_functions()

    # 各テスト関数に対してテストを実行
    for test_func in test_functions:
        print(f"\n--- テスト関数: {test_func.name} ---")

        # 関数値を計算
        f_values = evaluate_function_on_grid(test_func.f, x_grid, y_grid)

        # 解析解の計算
        analytical_results = {
            "f": f_values,
            "f_x": evaluate_function_on_grid(test_func.f_x, x_grid, y_grid),
            "f_y": evaluate_function_on_grid(test_func.f_y, x_grid, y_grid),
            "f_xx": evaluate_function_on_grid(test_func.f_xx, x_grid, y_grid),
            "f_yy": evaluate_function_on_grid(test_func.f_yy, x_grid, y_grid),
            "f_xy": evaluate_function_on_grid(test_func.f_xy, x_grid, y_grid),
        }

        # CCDソルバーの初期化
        solver = CCD2DSolver(
            grid_config,
            use_iterative=use_iterative,
            solver_type=solver_kwargs.get("solver_type", "gmres"),
            solver_kwargs={
                k: v for k, v in solver_kwargs.items() if k != "solver_type"
            },
        )

        # 計測開始
        start_time = time.time()

        # ソルバーを実行
        numerical_results = solver.solve(f_values)

        # 計測終了
        elapsed_time = time.time() - start_time

        # 誤差の計算と表示
        print(f"計算時間: {elapsed_time:.4f} 秒")
        print("L2誤差:")
        for comp in ["f_x", "f_y", "f_xx", "f_yy", "f_xy"]:
            if comp in numerical_results and comp in analytical_results:
                error = compute_error(
                    numerical_results[comp], analytical_results[comp], "L2"
                )
                print(f"  {comp}: {error:.2e}")

        # 結果の可視化
        if visualize:
            plot_results(
                x_grid,
                y_grid,
                numerical_results,
                analytical_results,
                test_func.name,
                save_path=f"results_{test_func.name.lower()}.png",
            )


if __name__ == "__main__":
    # テストの実行
    run_test(
        nx=30,
        ny=30,
        domain=(-1.0, 1.0, -1.0, 1.0),
        use_iterative=True,
        solver_kwargs={"solver_type": "gmres", "tol": 1e-10, "maxiter": 1000},
        visualize=True,
    )