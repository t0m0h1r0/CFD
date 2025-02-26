"""
ソルバー比較モジュール

異なるCCDソルバー間での性能比較を行う機能を提供します。
"""

import jax.numpy as jnp
from typing import List, Tuple, Dict, Any, Optional
import json
import os

from ccd_core import GridConfig
from test_functions import TestFunction, TestFunctionFactory
from ccd_tester import CCDMethodTester
from visualization import visualize_error_comparison


class SolverComparator:
    """複数のソルバーを比較するクラス"""
    
    def __init__(
        self, 
        solvers_list: List[Tuple[str, CCDMethodTester]],
        grid_config: GridConfig,
        x_range: Tuple[float, float],
        test_functions: Optional[List[TestFunction]] = None
    ):
        """
        Args:
            solvers_list: [("名前", テスターインスタンス), ...] の形式のリスト
            grid_config: グリッド設定
            x_range: x軸の範囲 (開始位置, 終了位置)
            test_functions: テスト関数のリスト (Noneの場合は標準関数セットを使用)
        """
        self.solvers_list = solvers_list
        self.grid_config = grid_config
        self.x_range = x_range
        self.test_functions = test_functions or TestFunctionFactory.create_standard_functions()
        
        # 各テスターのテスト関数をこのクラスで指定したものに統一
        for _, tester in solvers_list:
            tester.test_functions = self.test_functions
        
        # テスターを名前をキーとして辞書に保存
        self.testers = {}
        for name, tester in solvers_list:
            self.testers[name] = tester
    
    def run_comparison(self, save_results: bool = True, visualize: bool = True):
        """
        全てのソルバーに対して比較テストを実行
        
        Args:
            save_results: 結果をJSONファイルに保存するかどうか
            visualize: 比較結果を可視化するかどうか
            
        Returns:
            比較結果の辞書
        """
        results = {}  # ソルバー名 -> {関数名 -> [1階誤差, 2階誤差, 3階誤差]} の辞書
        timings = {}  # ソルバー名 -> {関数名 -> 計算時間} の辞書
        
        # 出力ディレクトリを作成
        os.makedirs("results", exist_ok=True)
        
        for name, tester in self.solvers_list:
            print(f"\n===== Testing {name} solver =====")
            
            # テストを実行
            test_results = tester.run_tests(prefix=f"{name.lower()}_", visualize=visualize)
            
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
                visualize_error_comparison(
                    results, 
                    timings, 
                    func.name, 
                    save_path=f"results/comparison_{func.name.lower()}.png"
                )
        
        # 結果の保存
        if save_results:
            comparison_data = {
                "grid_points": self.grid_config.n_points,
                "grid_spacing": self.grid_config.h,
                "x_range": self.x_range,
                "solvers": [name for name, _ in self.solvers_list],
                "results": results,
                "timings": timings
            }
            
            os.makedirs("results", exist_ok=True)
            with open("results/comparison_results.json", "w") as f:
                json.dump(comparison_data, f, indent=2)
        
        return results, timings
    
    def _print_comparison_tables(self, results: Dict, timings: Dict):
        """比較表を出力"""
        # 各テスト関数ごとに比較表を出力
        print("\n===== Error Comparison =====")
        for func_name in next(iter(results.values())).keys():
            print(f"\n{func_name} Function:")
            print(f"{'Solver':<15} {'1st Der.':<12} {'2nd Der.':<12} {'3rd Der.':<12} {'Time (s)':<12}")
            print("-" * 63)
            
            for name in results.keys():
                errors = results[name][func_name]
                time = timings[name][func_name]
                print(f"{name:<15} {errors[0]:<12.2e} {errors[1]:<12.2e} {errors[2]:<12.2e} {time:<12.4f}")
        
        # 全体の平均誤差と時間を計算
        print("\n===== Overall Performance =====")
        print(f"{'Solver':<15} {'Avg 1st Der.':<12} {'Avg 2nd Der.':<12} {'Avg 3rd Der.':<12} {'Avg Time (s)':<12}")
        print("-" * 63)
        
        for name in results.keys():
            avg_errors = [0.0, 0.0, 0.0]
            avg_time = 0.0
            
            for func_name in results[name].keys():
                for i in range(3):
                    avg_errors[i] += results[name][func_name][i]
                avg_time += timings[name][func_name]
            
            # 平均を計算
            func_count = len(results[name])
            avg_errors = [e / func_count for e in avg_errors]
            avg_time /= func_count
            
            print(f"{name:<15} {avg_errors[0]:<12.2e} {avg_errors[1]:<12.2e} {avg_errors[2]:<12.2e} {avg_time:<12.4f}")