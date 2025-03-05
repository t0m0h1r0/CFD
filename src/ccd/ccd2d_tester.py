"""
2次元CCDテスターモジュール

2次元CCDソルバーの精度と性能をテストするためのクラスを提供します。
"""

import os
import time
import cupy as cp
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Type

from grid2d_config import Grid2DConfig
from ccd2d_solver import CCD2DSolver
from test2d_functions import Test2DFunction, Test2DFunctionFactory
from visualization2d_utils import visualize_all_results


class CCD2DMethodTester:
    """2次元CCD法のテストを実行するクラス"""

    def __init__(
        self,
        solver_class: Type[CCD2DSolver],
        grid_config: Grid2DConfig,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        solver_kwargs: Optional[Dict[str, Any]] = None,
        test_functions: Optional[List[Test2DFunction]] = None,
        coeffs: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            solver_class: テスト対象の2次元CCDソルバークラス
            grid_config: 2次元グリッド設定
            x_range: x軸の範囲 (開始位置, 終了位置)
            y_range: y軸の範囲 (開始位置, 終了位置)
            solver_kwargs: ソルバーの初期化パラメータ
            test_functions: テスト関数のリスト (Noneの場合は標準関数セットを使用)
            coeffs: 方程式の係数辞書
        """
        # 元のグリッド設定を保存
        self.original_grid_config = grid_config
        self.x_range = x_range
        self.y_range = y_range
        
        # グリッド設定から間隔を計算
        self.hx = (x_range[1] - x_range[0]) / (grid_config.nx - 1)
        self.hy = (y_range[1] - y_range[0]) / (grid_config.ny - 1)
        
        # グリッド情報の更新
        grid_config.hx = self.hx
        grid_config.hy = self.hy
        
        # 係数の設定
        self.coeffs = coeffs
        if coeffs is not None:
            grid_config.coeffs = coeffs

        # ソルバーパラメータを保存
        self.solver_kwargs = solver_kwargs or {}
        self.solver_class = solver_class
        self.solver_name = solver_class.__name__

        # テスト関数の設定
        self.test_functions = (
            test_functions or Test2DFunctionFactory.create_standard_functions()
        )

        # 最初のテスト関数を使って、境界条件の値を設定
        first_test_func = self.test_functions[0]

        # 境界条件を含むグリッド設定を作成
        boundary_grid_config = self._create_boundary_grid_config(first_test_func)

        # ソルバーの初期化
        solver_kwargs_copy = self.solver_kwargs.copy()
        if "coeffs" in solver_kwargs_copy:
            # solver_kwargsにcoeffsが指定されている場合は削除（grid_configに設定済み）
            del solver_kwargs_copy["coeffs"]

        self.solver = solver_class(boundary_grid_config, **solver_kwargs_copy)

    def _create_boundary_grid_config(self, test_func: Test2DFunction) -> Grid2DConfig:
        """
        テスト関数から境界条件を含むグリッド設定を作成

        Args:
            test_func: テスト関数

        Returns:
            境界条件が設定されたGrid2DConfig
        """
        nx, ny = self.original_grid_config.nx, self.original_grid_config.ny
        
        # x方向のメッシュグリッド
        x_start, x_end = self.x_range
        x = cp.linspace(x_start, x_end, nx)
        
        # y方向のメッシュグリッド
        y_start, y_end = self.y_range
        y = cp.linspace(y_start, y_end, ny)
        
        # メッシュグリッドを作成
        X, Y = cp.meshgrid(x, y)
        
        # ディリクレ境界条件の値（関数値）
        # 左端 (x=x_start, y=all)
        left_x = cp.zeros(ny)
        for j in range(ny):
            left_x[j] = test_func.f(x_start, y[j])
            
        # 右端 (x=x_end, y=all)
        right_x = cp.zeros(ny)
        for j in range(ny):
            right_x[j] = test_func.f(x_end, y[j])
            
        # 下端 (x=all, y=y_start)
        bottom_y = cp.zeros(nx)
        for i in range(nx):
            bottom_y[i] = test_func.f(x[i], y_start)
            
        # 上端 (x=all, y=y_end)
        top_y = cp.zeros(nx)
        for i in range(nx):
            top_y[i] = test_func.f(x[i], y_end)
        
        # ノイマン境界条件の値（微分値）
        # 左端 (x=x_start, y=all)
        left_dx = cp.zeros(ny)
        for j in range(ny):
            left_dx[j] = test_func.f_x(x_start, y[j])
            
        # 右端 (x=x_end, y=all)
        right_dx = cp.zeros(ny)
        for j in range(ny):
            right_dx[j] = test_func.f_x(x_end, y[j])
            
        # 下端 (x=all, y=y_start)
        bottom_dy = cp.zeros(nx)
        for i in range(nx):
            bottom_dy[i] = test_func.f_y(x[i], y_start)
            
        # 上端 (x=all, y=y_end)
        top_dy = cp.zeros(nx)
        for i in range(nx):
            top_dy[i] = test_func.f_y(x[i], y_end)
            
        # 簡単のため、各辺の平均値を使用
        dirichlet_values_x = [float(cp.mean(left_x)), float(cp.mean(right_x))]
        dirichlet_values_y = [float(cp.mean(bottom_y)), float(cp.mean(top_y))]
        neumann_values_x = [float(cp.mean(left_dx)), float(cp.mean(right_dx))]
        neumann_values_y = [float(cp.mean(bottom_dy)), float(cp.mean(top_dy))]

        # 新しいグリッド設定を作成
        return Grid2DConfig(
            nx=nx,
            ny=ny,
            hx=self.hx,
            hy=self.hy,
            dirichlet_values_x=dirichlet_values_x,
            dirichlet_values_y=dirichlet_values_y,
            neumann_values_x=neumann_values_x,
            neumann_values_y=neumann_values_y,
            coeffs=self.coeffs or self.original_grid_config.coeffs,
            x_deriv_order=self.original_grid_config.x_deriv_order,
            y_deriv_order=self.original_grid_config.y_deriv_order,
            enable_boundary_correction=self.original_grid_config.enable_boundary_correction,
        )

    def compute_errors(
        self, test_func: Test2DFunction
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        """
        各導関数の誤差を計算

        Args:
            test_func: テスト関数

        Returns:
            (誤差メトリクス辞書, 計算時間)
        """
        # テスト関数に合わせた境界条件でグリッド設定を更新
        boundary_grid_config = self._create_boundary_grid_config(test_func)

        nx, ny = boundary_grid_config.nx, boundary_grid_config.ny

        # グリッド点でのx, y座標を生成
        x_points = cp.linspace(self.x_range[0], self.x_range[1], nx)
        y_points = cp.linspace(self.y_range[0], self.y_range[1], ny)
        X, Y = cp.meshgrid(x_points, y_points, indexing='ij')
        
        # テスト関数と導関数をグリッド上で評価
        analytical_results = Test2DFunctionFactory.evaluate_function_on_grid(
            test_func, X, Y
        )
        
        # 入力関数値
        f_values = analytical_results["f"]

        # ソルバーの初期化
        solver_kwargs_copy = self.solver_kwargs.copy()
        if "coeffs" in solver_kwargs_copy:
            del solver_kwargs_copy["coeffs"]
            
        self.solver = self.solver_class(boundary_grid_config, **solver_kwargs_copy)

        # 計測開始
        start_time = time.time()

        # 数値解の計算
        numerical_results = self.solver.solve(f_values)

        # 計測終了
        elapsed_time = time.time() - start_time

        # 誤差メトリクスの計算
        error_metrics = Test2DFunctionFactory.calculate_error_metrics(
            numerical_results, analytical_results
        )

        return error_metrics, elapsed_time

    def run_tests(
        self, prefix: str = "", visualize: bool = True, output_dir: str = "results_2d"
    ) -> Dict[str, Tuple[Dict[str, Dict[str, float]], float]]:
        """
        すべてのテスト関数に対してテストを実行

        Args:
            prefix: 出力ファイルの接頭辞
            visualize: 可視化を行うかどうか
            output_dir: 出力ディレクトリ

        Returns:
            テスト結果の辞書 {関数名: (誤差メトリクス, 計算時間)}
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)

        results = {}
        total_time = 0.0

        # 結果テーブルのヘッダーを表示
        coeff_str = f" (coeffs={self.coeffs})"
        print(f"Error Analysis Results for {self.solver_name}{coeff_str}:")
        print("-" * 100)
        print(
            f"{'Function':<15} {'f_x (L2)':<12} {'f_y (L2)':<12} {'f_xx (L2)':<12} {'f_yy (L2)':<12} {'f_xy (L2)':<12} {'Time (s)':<12}"
        )
        print("-" * 100)

        for test_func in self.test_functions:
            # 誤差と時間を計算
            error_metrics, elapsed_time = self.compute_errors(test_func)
            results[test_func.name] = (error_metrics, elapsed_time)

            # 結果を表示
            l2_fx = error_metrics.get("f_x", {}).get("l2", float('nan'))
            l2_fy = error_metrics.get("f_y", {}).get("l2", float('nan'))
            l2_fxx = error_metrics.get("f_xx", {}).get("l2", float('nan'))
            l2_fyy = error_metrics.get("f_yy", {}).get("l2", float('nan'))
            l2_fxy = error_metrics.get("f_xy", {}).get("l2", float('nan'))
            
            print(
                f"{test_func.name:<15} {l2_fx:<12.2e} {l2_fy:<12.2e} {l2_fxx:<12.2e} {l2_fyy:<12.2e} {l2_fxy:<12.2e} {elapsed_time:<12.4f}"
            )

            # 時間を累積
            total_time += elapsed_time

            # 可視化が有効な場合、詳細な可視化を実行
            if visualize:
                # 境界条件を含むグリッド設定を作成
                boundary_grid_config = self._create_boundary_grid_config(test_func)
                
                # グリッド点でのx, y座標を生成
                nx, ny = boundary_grid_config.nx, boundary_grid_config.ny
                x_points = cp.linspace(self.x_range[0], self.x_range[1], nx)
                y_points = cp.linspace(self.y_range[0], self.y_range[1], ny)
                X, Y = cp.meshgrid(x_points, y_points, indexing='ij')
                
                # 解析解を計算
                analytical_results = Test2DFunctionFactory.evaluate_function_on_grid(
                    test_func, X, Y
                )
                
                # 入力関数値
                f_values = analytical_results["f"]
                
                # 数値解を計算
                numerical_results = self.solver.solve(f_values)
                
                # 可視化
                visualize_all_results(
                    numerical_results,
                    analytical_results,
                    boundary_grid_config,
                    test_func_name=test_func.name,
                    output_dir=output_dir,
                    prefix=f"{prefix}{test_func.name.lower()}_"
                )

        # 平均時間を表示
        avg_time = total_time / len(self.test_functions)
        print("-" * 100)
        print(f"{'Average Time:':<15} {'':<60} {avg_time:<12.4f}")
        print("-" * 100)

        return results

    def convergence_study(
        self, 
        test_func: Test2DFunction,
        grid_sizes: List[int],
        is_square_grid: bool = True,
        output_dir: str = "convergence_2d"
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        収束性の調査を実行

        Args:
            test_func: テスト関数
            grid_sizes: テストするグリッドサイズのリスト
            is_square_grid: 正方形グリッドを使用するかどうか
            output_dir: 出力ディレクトリ

        Returns:
            収束性の結果辞書
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 結果保存用の辞書
        convergence_results = {
            "grid_sizes": grid_sizes,
            "grid_spacing": [],
            "errors": {
                "f_x": {"l2": [], "max": []},
                "f_y": {"l2": [], "max": []},
                "f_xx": {"l2": [], "max": []},
                "f_yy": {"l2": [], "max": []},
                "f_xy": {"l2": [], "max": []}
            },
            "time": []
        }
        
        print(f"\n収束性調査: {test_func.name} 関数")
        print("-" * 100)
        print(f"{'Grid Size':<15} {'h':<12} {'f_x (L2)':<12} {'f_y (L2)':<12} {'f_xx (L2)':<12} {'f_yy (L2)':<12} {'Time (s)':<12}")
        print("-" * 100)
        
        for size in grid_sizes:
            # グリッドサイズの設定
            if is_square_grid:
                nx = ny = size
            else:
                nx = size
                ny = size // 2  # 例: 縦横比を2:1に設定
                
            # グリッド間隔
            hx = (self.x_range[1] - self.x_range[0]) / (nx - 1)
            hy = (self.y_range[1] - self.y_range[0]) / (ny - 1)
            h = max(hx, hy)  # 代表的なグリッド間隔
            
            # グリッド設定の更新
            grid_config = Grid2DConfig(
                nx=nx,
                ny=ny,
                hx=hx,
                hy=hy,
                x_deriv_order=self.original_grid_config.x_deriv_order,
                y_deriv_order=self.original_grid_config.y_deriv_order,
                coeffs=self.coeffs or self.original_grid_config.coeffs,
            )
            
            # テスターの初期化
            tester = CCD2DMethodTester(
                self.solver_class,
                grid_config,
                x_range=self.x_range,
                y_range=self.y_range,
                solver_kwargs=self.solver_kwargs,
                test_functions=[test_func],
                coeffs=self.coeffs,
            )
            
            # 誤差と時間を計算
            error_metrics, elapsed_time = tester.compute_errors(test_func)
            
            # 結果の保存
            convergence_results["grid_spacing"].append(h)
            convergence_results["time"].append(elapsed_time)
            
            # 誤差の保存
            for deriv_key in convergence_results["errors"].keys():
                if deriv_key in error_metrics:
                    for metric_key in convergence_results["errors"][deriv_key].keys():
                        if metric_key in error_metrics[deriv_key]:
                            convergence_results["errors"][deriv_key][metric_key].append(
                                error_metrics[deriv_key][metric_key]
                            )
            
            # 結果の表示
            l2_fx = error_metrics.get("f_x", {}).get("l2", float('nan'))
            l2_fy = error_metrics.get("f_y", {}).get("l2", float('nan'))
            l2_fxx = error_metrics.get("f_xx", {}).get("l2", float('nan'))
            l2_fyy = error_metrics.get("f_yy", {}).get("l2", float('nan'))
            
            print(
                f"{nx}x{ny:<15} {h:<12.3e} {l2_fx:<12.2e} {l2_fy:<12.2e} {l2_fxx:<12.2e} {l2_fyy:<12.2e} {elapsed_time:<12.4f}"
            )
        
        # 各メトリクスの収束率を計算
        print("\n収束率:")
        for deriv_key, metrics in convergence_results["errors"].items():
            for metric_key, values in metrics.items():
                if len(values) >= 2:
                    # 最小二乗法で傾きを計算（log-logスケール）
                    h_values = convergence_results["grid_spacing"]
                    if len(values) == len(h_values):
                        log_h = cp.log10(cp.array(h_values))
                        log_err = cp.log10(cp.array(values))
                        
                        # 異常値を排除
                        valid_indices = cp.isfinite(log_err)
                        if cp.any(valid_indices):
                            log_h_valid = log_h[valid_indices]
                            log_err_valid = log_err[valid_indices]
                            
                            # 最小二乗法で傾きを計算
                            A = cp.vstack([log_h_valid, cp.ones_like(log_h_valid)]).T
                            slope, _ = cp.linalg.lstsq(A, log_err_valid, rcond=-1)[0]
                            
                            print(f"{deriv_key} - {metric_key}: 傾き = {slope:.2f} (理論上の収束率: {deriv_key.count('x') + deriv_key.count('y')})")
        
        # 結果をJSONファイルに保存
        results_file = os.path.join(output_dir, f"convergence_{test_func.name.lower()}.json")
        with open(results_file, 'w') as f:
            # NumPy/CuPy配列をリストに変換してから保存
            json_compatible = {}
            for key, val in convergence_results.items():
                if isinstance(val, dict):
                    json_compatible[key] = {}
                    for subkey, subval in val.items():
                        if isinstance(subval, dict):
                            json_compatible[key][subkey] = {}
                            for metric, metric_val in subval.items():
                                json_compatible[key][subkey][metric] = [float(v) for v in metric_val]
                        else:
                            json_compatible[key][subkey] = [float(v) for v in subval]
                else:
                    json_compatible[key] = [float(v) for v in val] if isinstance(val, (list, cp.ndarray)) else val
            
            json.dump(json_compatible, f, indent=2)
        
        return convergence_results
