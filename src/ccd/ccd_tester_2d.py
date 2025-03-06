"""
シンプル化された2次元CCD法テスターモジュール

2次元CCDソルバー実装のテスト機能を提供します。
"""

import cupy as cp
import time
import os
from typing import Tuple, List, Type, Dict, Any, Optional

from grid_config_2d import GridConfig2D
from composite_solver_2d import CCDCompositeSolver2D
from test_functions_2d import TestFunction2DExplicit, TestFunction2DFactory
from visualization_2d import visualize_derivative_results_2d


class CCDMethodTester2D:
    """2次元CCD法のテストを実行するクラス"""

    def __init__(
        self,
        solver_class: Type[CCDCompositeSolver2D],
        grid_config: GridConfig2D,
        xy_range: Tuple[Tuple[float, float], Tuple[float, float]],
        solver_kwargs: Optional[Dict[str, Any]] = None,
        test_functions: Optional[List[TestFunction2DExplicit]] = None,
        coeffs: Optional[List[float]] = None,
    ):
        """
        Args:
            solver_class: テスト対象の2次元CCDソルバークラス
            grid_config: 2次元グリッド設定
            xy_range: (x軸の範囲, y軸の範囲) のタプル
            solver_kwargs: ソルバーの初期化パラメータ
            test_functions: テスト関数のリスト (Noneの場合は標準関数セットを使用)
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用 (f = psi)
        """
        # 元のグリッド設定を保存
        self.original_grid_config = grid_config
        self.xy_range = xy_range
        self.coeffs = coeffs if coeffs is not None else [1.0, 0.0, 0.0, 0.0]

        # xy範囲の保存
        self.x_range, self.y_range = xy_range

        # ソルバーパラメータを保存
        self.solver_kwargs = solver_kwargs or {}
        self.solver_class = solver_class

        # テスト関数の設定
        self.test_functions = (
            test_functions or TestFunction2DFactory.create_standard_functions()
        )

        # 境界条件が更新されたグリッド設定を作成
        boundary_grid_config = self._create_boundary_grid_config(self.test_functions[0])

        # ソルバーの初期化
        solver_kwargs_copy = self.solver_kwargs.copy()
        if "coeffs" in solver_kwargs_copy:
            del solver_kwargs_copy["coeffs"]

        self.solver = solver_class(boundary_grid_config, **solver_kwargs_copy)
        self.solver_name = solver_class.__name__

    def _create_boundary_grid_config(self, test_func: TestFunction2DExplicit) -> GridConfig2D:
        """
        テスト関数から境界条件を含むグリッド設定を作成

        Args:
            test_func: 2次元テスト関数

        Returns:
            境界条件が設定された2次元GridConfig
        """
        # 元のグリッド設定をコピー
        nx, ny = self.original_grid_config.nx_points, self.original_grid_config.ny_points
        hx, hy = self.original_grid_config.hx, self.original_grid_config.hy
        
        # グリッド点を計算
        x_start, x_end = self.x_range
        y_start, y_end = self.y_range
        
        # 境界値を計算
        x = cp.linspace(x_start, x_end, nx)
        y = cp.linspace(y_start, y_end, ny)
        
        # 境界での関数値を計算（ディリクレ条件用）
        left_boundary = cp.array([test_func.f(x_start, yj) for yj in y])
        right_boundary = cp.array([test_func.f(x_end, yj) for yj in y])
        bottom_boundary = cp.array([test_func.f(xi, y_start) for xi in x])
        top_boundary = cp.array([test_func.f(xi, y_end) for xi in x])
        
        # 境界での勾配値を計算（ノイマン条件用）
        left_neumann = cp.array([test_func.df_dx(x_start, yj) for yj in y])
        right_neumann = cp.array([test_func.df_dx(x_end, yj) for yj in y])
        bottom_neumann = cp.array([test_func.df_dy(xi, y_start) for xi in x])
        top_neumann = cp.array([test_func.df_dy(xi, y_end) for xi in x])
        
        # 境界値の設定
        boundary_values = {
            "left": left_boundary.tolist(),
            "right": right_boundary.tolist(),
            "bottom": bottom_boundary.tolist(),
            "top": top_boundary.tolist(),
            "left_neumann": left_neumann.tolist(),
            "right_neumann": right_neumann.tolist(),
            "bottom_neumann": bottom_neumann.tolist(),
            "top_neumann": top_neumann.tolist(),
        }
        
        # 新しいグリッド設定を作成
        return GridConfig2D(
            nx_points=nx,
            ny_points=ny,
            hx=hx,
            hy=hy,
            boundary_values=boundary_values,
            coeffs=self.coeffs,
        )

    def compute_errors(
        self, test_func: TestFunction2DExplicit
    ) -> Tuple[float, float, float, float, float, float]:
        """
        各偏導関数の誤差を計算

        Args:
            test_func: 2次元テスト関数

        Returns:
            (f_x誤差, f_y誤差, f_xx誤差, f_xy誤差, f_yy誤差, 計算時間)のタプル
        """
        # テスト関数に合わせた境界条件でグリッド設定を更新
        boundary_grid_config = self._create_boundary_grid_config(test_func)

        nx, ny = boundary_grid_config.nx_points, boundary_grid_config.ny_points
        hx, hy = boundary_grid_config.hx, boundary_grid_config.hy
        x_start, y_start = self.x_range[0], self.y_range[0]

        # グリッド点でのx,y座標を計算
        x = cp.linspace(x_start, self.x_range[1], nx)
        y = cp.linspace(y_start, self.y_range[1], ny)
        X, Y = cp.meshgrid(x, y)

        # 解析解の計算
        f_values = cp.zeros((nx, ny))
        f_x_exact = cp.zeros((nx, ny))
        f_y_exact = cp.zeros((nx, ny))
        f_xx_exact = cp.zeros((nx, ny))
        f_xy_exact = cp.zeros((nx, ny))
        f_yy_exact = cp.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                f_values[i, j] = test_func.f(x[i], y[j])
                f_x_exact[i, j] = test_func.df_dx(x[i], y[j])
                f_y_exact[i, j] = test_func.df_dy(x[i], y[j])
                f_xx_exact[i, j] = test_func.d2f_dx2(x[i], y[j])
                f_xy_exact[i, j] = test_func.d2f_dxdy(x[i], y[j])
                f_yy_exact[i, j] = test_func.d2f_dy2(x[i], y[j])

        # ソルバーの初期化
        solver_kwargs_copy = self.solver_kwargs.copy()
        if "coeffs" in solver_kwargs_copy:
            del solver_kwargs_copy["coeffs"]

        self.solver = self.solver_class(boundary_grid_config, **solver_kwargs_copy)

        # 係数に基づいて入力関数値を計算
        a, b, c, d = boundary_grid_config.coeffs
        input_values = (
            a * f_values
            + b * f_x_exact
            + c * f_xx_exact
            + d * f_yy_exact
        )

        # 計測開始
        start_time = time.time()

        # 数値解の計算
        f, f_x, f_y, f_xx, f_xy, f_yy = self.solver.solve(input_values)

        # 計測終了
        elapsed_time = time.time() - start_time

        # 誤差の計算 (L2ノルム)
        error_fx = cp.sqrt(cp.mean((f_x - f_x_exact) ** 2))
        error_fy = cp.sqrt(cp.mean((f_y - f_y_exact) ** 2))
        error_fxx = cp.sqrt(cp.mean((f_xx - f_xx_exact) ** 2))
        error_fxy = cp.sqrt(cp.mean((f_xy - f_xy_exact) ** 2))
        error_fyy = cp.sqrt(cp.mean((f_yy - f_yy_exact) ** 2))

        return float(error_fx), float(error_fy), float(error_fxx), float(error_fxy), float(error_fyy), elapsed_time

    def run_tests(
        self, prefix: str = "", visualize: bool = True
    ) -> Dict[str, Tuple[List[float], float]]:
        """
        すべてのテスト関数に対してテストを実行

        Args:
            prefix: 出力ファイルの接頭辞
            visualize: 可視化を行うかどうか

        Returns:
            テスト結果の辞書 {関数名: ([f_x誤差, f_y誤差, f_xx誤差, f_xy誤差, f_yy誤差], 計算時間)}
        """
        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)

        results = {}
        total_errors = [0.0, 0.0, 0.0, 0.0, 0.0]
        total_time = 0.0

        # 結果テーブルのヘッダーを表示
        coeff_str = f" (coeffs={self.coeffs})"
        print(f"2D Error Analysis Results for {self.solver_name}{coeff_str}:")
        print("-" * 90)
        print(
            f"{'Function':<15} {'∂f/∂x':<12} {'∂f/∂y':<12} {'∂²f/∂x²':<12} {'∂²f/∂x∂y':<12} {'∂²f/∂y²':<12} {'Time (s)':<12}"
        )
        print("-" * 90)

        for test_func in self.test_functions:
            # 誤差と時間を計算
            errors = self.compute_errors(test_func)
            results[test_func.name] = (errors[:5], errors[5])

            # 結果を表示
            print(
                f"{test_func.name:<15} {errors[0]:<12.2e} {errors[1]:<12.2e} {errors[2]:<12.2e} "
                f"{errors[3]:<12.2e} {errors[4]:<12.2e} {errors[5]:<12.4f}"
            )

            # 誤差と時間を累積
            for i in range(5):
                total_errors[i] += errors[i]
            total_time += errors[5]

            # 可視化（オプション）
            if visualize:
                self._visualize_test_results(test_func, prefix)

        # 平均誤差と時間を表示
        avg_errors = [e / len(self.test_functions) for e in total_errors]
        avg_time = total_time / len(self.test_functions)

        print("-" * 90)
        print(
            f"{'Average':<15} {avg_errors[0]:<12.2e} {avg_errors[1]:<12.2e} {avg_errors[2]:<12.2e} "
            f"{avg_errors[3]:<12.2e} {avg_errors[4]:<12.2e} {avg_time:<12.4f}"
        )
        print("-" * 90)

        # 境界条件の情報を表示
        print("\n境界条件: ディリクレ + ノイマン (テスト関数の境界値を使用)")
        print("各テスト関数ごとに境界値を動的に設定")

        return results

    def _visualize_test_results(self, test_func: TestFunction2DExplicit, prefix: str):
        """テスト関数の結果を可視化"""
        # 境界条件を含むグリッド設定を作成
        boundary_grid_config = self._create_boundary_grid_config(test_func)

        nx, ny = boundary_grid_config.nx_points, boundary_grid_config.ny_points
        hx, hy = boundary_grid_config.hx, boundary_grid_config.hy
        
        # グリッド点を計算
        x = cp.linspace(self.x_range[0], self.x_range[1], nx)
        y = cp.linspace(self.y_range[0], self.y_range[1], ny)
        
        # 解析解の計算
        f_values = cp.zeros((nx, ny))
        f_x_exact = cp.zeros((nx, ny))
        f_y_exact = cp.zeros((nx, ny))
        f_xx_exact = cp.zeros((nx, ny))
        f_xy_exact = cp.zeros((nx, ny))
        f_yy_exact = cp.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                f_values[i, j] = test_func.f(x[i], y[j])
                f_x_exact[i, j] = test_func.df_dx(x[i], y[j])
                f_y_exact[i, j] = test_func.df_dy(x[i], y[j])
                f_xx_exact[i, j] = test_func.d2f_dx2(x[i], y[j])
                f_xy_exact[i, j] = test_func.d2f_dxdy(x[i], y[j])
                f_yy_exact[i, j] = test_func.d2f_dy2(x[i], y[j])

        # 係数に基づいて入力関数値を計算
        a, b, c, d = boundary_grid_config.coeffs
        input_values = (
            a * f_values
            + b * f_x_exact
            + c * f_xx_exact
            + d * f_yy_exact
        )

        # 数値解の計算
        f, f_x, f_y, f_xx, f_xy, f_yy = self.solver.solve(input_values)

        # 解析解のタプル
        analytical_derivatives = (f_values, f_x_exact, f_y_exact, f_xx_exact, f_xy_exact, f_yy_exact)
        
        # 数値解のタプル
        numerical_derivatives = (f, f_x, f_y, f_xx, f_xy, f_yy)

        # モード名を取得
        mode_name = self._get_mode_name()

        # 可視化保存パス
        save_path = f"results/{prefix}{test_func.name.lower()}_2d_results.png"

        # 結果を可視化
        visualize_derivative_results_2d(
            test_func=test_func,
            f_values=input_values,
            numerical_derivatives=numerical_derivatives,
            analytical_derivatives=analytical_derivatives,
            grid_config=boundary_grid_config,
            xy_range=(self.x_range, self.y_range),
            solver_name=f"{self.solver_name} ({mode_name})",
            save_path=save_path,
        )

    def _get_mode_name(self) -> str:
        """係数に基づく微分モード名を取得"""
        mode_names = {
            (1, 0, 0, 0): "PSI",
            (0, 1, 0, 0): "PSI_X",
            (0, 0, 1, 0): "PSI_XX",
            (0, 0, 0, 1): "PSI_YY",
            (1, 1, 0, 0): "PSI+PSI_X",
            (1, 0, 1, 0): "PSI+PSI_XX",
            (1, 0, 0, 1): "PSI+PSI_YY",
            (1, 1, 1, 1): "PSI+PSI_X+PSI_XX+PSI_YY",
        }

        coeffs_tuple = tuple(self.coeffs)
        return mode_names.get(coeffs_tuple, f"coeffs={self.coeffs}")