"""
2次元ソルバー比較モジュール

異なる2次元CCDソルバー間での性能比較を行う機能を提供します。
"""

from typing import List, Tuple, Dict, Optional
import json
import os

from grid_config_2d import GridConfig2D
from test_functions_2d import TestFunction2DExplicit, TestFunction2DFactory
from ccd_tester_2d import CCDMethodTester2D
from visualization_2d import visualize_error_comparison_2d


class SolverComparator2D:
    """複数の2次元ソルバーを比較するクラス"""

    def __init__(
        self,
        solvers_list: List[Tuple[str, CCDMethodTester2D]],
        grid_config: GridConfig2D,
        xy_range: Tuple[Tuple[float, float], Tuple[float, float]],
        test_functions: Optional[List[TestFunction2DExplicit]] = None,
        coeffs: Optional[List[float]] = None,
    ):
        """
        Args:
            solvers_list: [("名前", テスターインスタンス), ...] の形式のリスト
            grid_config: 2次元グリッド設定
            xy_range: ((x_min, x_max), (y_min, y_max)) の形式の座標範囲
            test_functions: テスト関数のリスト (Noneの場合は標準関数セットを使用)
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用 (f = psi)
        """
        self.solvers_list = solvers_list
        self.grid_config = grid_config
        self.xy_range = xy_range
        self.test_functions = (
            test_functions or TestFunction2DFactory.create_standard_functions()
        )
        self.coeffs = coeffs

        # 各テスターのテスト関数をこのクラスで指定したものに統一
        for _, tester in solvers_list:
            tester.test_functions = self.test_functions
            if self.coeffs is not None:
                tester.coeffs = self.coeffs

    def _print_comparison_tables(self, results: Dict, timings: Dict):
        """比較表を出力"""
        # 係数情報の文字列
        coeff_str = "" if self.coeffs is None else f" (coeffs={self.coeffs})"

        # 各テスト関数ごとに比較表を出力
        print(f"\n===== 2D Error Comparison{coeff_str} =====")
        for func_name in next(iter(results.values())).keys():
            print(f"\n{func_name} Function:")
            print(
                f"{'Solver':<15} {'∂f/∂x':<12} {'∂f/∂y':<12} {'∂²f/∂x²':<12} "
                f"{'∂²f/∂x∂y':<12} {'∂²f/∂y²':<12} {'Time (s)':<12}"
            )
            print("-" * 90)

            for name in results.keys():
                errors = results[name][func_name]
                time = timings[name][func_name]
                print(
                    f"{name:<15} {errors[0]:<12.2e} {errors[1]:<12.2e} {errors[2]:<12.2e} "
                    f"{errors[3]:<12.2e} {errors[4]:<12.2e} {time:<12.4f}"
                )

        # 全体の平均誤差と時間を計算
        print(f"\n===== Overall 2D Performance{coeff_str} =====")
        print(
            f"{'Solver':<15} {'Avg ∂f/∂x':<12} {'Avg ∂f/∂y':<12} {'Avg ∂²f/∂x²':<12} "
            f"{'Avg ∂²f/∂x∂y':<12} {'Avg ∂²f/∂y²':<12} {'Avg Time (s)':<12}"
        )
        print("-" * 90)

        for name in results.keys():
            avg_errors = [0.0, 0.0, 0.0, 0.0, 0.0]
            avg_time = 0.0

            for func_name in results[name].keys():
                for i in range(5):
                    avg_errors[i] += results[name][func_name][i]
                avg_time += timings[name][func_name]

            # 平均を計算
            func_count = len(results[name])
            avg_errors = [e / func_count for e in avg_errors]
            avg_time /= func_count

            print(
                f"{name:<15} {avg_errors[0]:<12.2e} {avg_errors[1]:<12.2e} {avg_errors[2]:<12.2e} "
                f"{avg_errors[3]:<12.2e} {avg_errors[4]:<12.2e} {avg_time:<12.4f}"
            )

    def run_comparison(
        self, save_results: bool = True, visualize: bool = True, prefix: str = ""
    ):
        """
        全てのソルバーに対して比較テストを実行

        Args:
            save_results: 結果をJSONファイルに保存するかどうか
            visualize: 比較結果を可視化するかどうか
            prefix: 出力ファイル名の接頭辞
        """
        results = {}  # ソルバー名 -> {関数名 -> [fx誤差, fy誤差, fxx誤差, fxy誤差, fyy誤差]} の辞書
        timings = {}  # ソルバー名 -> {関数名 -> 計算時間} の辞書

        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)

        # 係数情報の文字列
        coeff_str = "" if self.coeffs is None else f" (coeffs={self.coeffs})"

        # 各ソルバーに対してテストを実行
        for name, tester in self.solvers_list:
            print(f"\n===== Testing {name} 2D solver{coeff_str} =====")

            # テストを実行
            test_prefix = f"{name.lower()}_{prefix}" if prefix else f"{name.lower()}_"
            test_results = tester.run_tests(prefix=f"2d_{test_prefix}", visualize=visualize)

            # 結果を格納
            solver_results = {}
            solver_timings = {}

            for func_name, (errors, time_val) in test_results.items():
                solver_results[func_name] = errors
                solver_timings[func_name] = time_val

            results[name] = solver_results
            timings[name] = solver_timings

        # 各ソルバー間の比較表を作成
        self._print_comparison_tables(results, timings)

        # 可視化
        if visualize:
            for func in self.test_functions:
                visualize_error_comparison_2d(
                    results,
                    timings,
                    func.name,
                    save_path=f"results/{prefix}2d_comparison_{func.name.lower()}.png",
                )

        # 結果の保存
        if save_results:
            comparison_data = {
                "grid_points_x": self.grid_config.nx_points,
                "grid_points_y": self.grid_config.ny_points,
                "grid_spacing_x": self.grid_config.hx,
                "grid_spacing_y": self.grid_config.hy,
                "x_range": self.xy_range[0],
                "y_range": self.xy_range[1],
                "coefficients": self.coeffs,
                "solvers": [name for name, _ in self.solvers_list],
                "results": results,
                "timings": timings,
            }

            os.makedirs("results", exist_ok=True)
            # ファイル名に接頭辞と係数情報を含める
            coeff_suffix = (
                ""
                if self.coeffs is None
                else f"_coeffs_{'-'.join(map(str, self.coeffs))}"
            )
            with open(
                f"results/{prefix}2d_comparison_results{coeff_suffix}.json", "w"
            ) as f:
                json.dump(comparison_data, f, indent=2)

        return results, timings