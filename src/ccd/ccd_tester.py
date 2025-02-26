"""
CCD法のテスト実行モジュール

数値導関数計算の精度と性能をテストする機能を提供します。
"""

import jax.numpy as jnp
import time
from typing import Tuple, List, Type, Dict, Any, Optional

from ccd_core import GridConfig
from ccd_solver import CCDSolver
from test_functions import TestFunction, TestFunctionFactory
from visualization import visualize_derivative_results


class CCDMethodTester:
    """CCD法のテストを実行するクラス"""

    def __init__(
        self, 
        solver_class: Type[CCDSolver], 
        grid_config: GridConfig, 
        x_range: Tuple[float, float], 
        solver_kwargs: Optional[Dict[str, Any]] = None,
        test_functions: Optional[List[TestFunction]] = None
    ):
        """
        Args:
            solver_class: テスト対象のCCDソルバークラス
            grid_config: グリッド設定
            x_range: x軸の範囲 (開始位置, 終了位置)
            solver_kwargs: ソルバーの初期化パラメータ
            test_functions: テスト関数のリスト (Noneの場合は標準関数セットを使用)
        """
        self.grid_config = grid_config
        self.x_range = x_range
        solver_kwargs = solver_kwargs or {}
        
        # ソルバーの初期化
        self.solver = solver_class(grid_config, **solver_kwargs)
        self.solver_class = solver_class
        self.solver_name = solver_class.__name__
        
        # テスト関数の設定
        self.test_functions = test_functions or TestFunctionFactory.create_standard_functions()

    def compute_errors(self, test_func: TestFunction) -> Tuple[float, float, float, float]:
        """
        各導関数の誤差を計算
        
        Args:
            test_func: テスト関数
            
        Returns:
            (f'の誤差, f''の誤差, f'''の誤差, 計算時間)
        """
        n = self.grid_config.n_points
        h = self.grid_config.h
        x_start = self.x_range[0]

        # グリッド点での関数値を計算
        x_points = jnp.array([x_start + i * h for i in range(n)])
        f_values = jnp.array([test_func.f(x) for x in x_points])

        # 計測開始
        start_time = time.time()
        
        # 数値解の計算
        numerical_derivatives = self.solver.solve(f_values)
        
        # 計測終了
        elapsed_time = time.time() - start_time

        # 解析解の計算
        analytical_derivatives = jnp.zeros(3 * n)
        for i in range(n):
            x = x_points[i]
            analytical_derivatives = analytical_derivatives.at[3 * i].set(
                test_func.df(x)
            )
            analytical_derivatives = analytical_derivatives.at[3 * i + 1].set(
                test_func.d2f(x)
            )
            analytical_derivatives = analytical_derivatives.at[3 * i + 2].set(
                test_func.d3f(x)
            )

        # 誤差の計算 (L2ノルム)
        errors = []
        for i in range(3):
            numerical = numerical_derivatives[i::3]
            analytical = analytical_derivatives[i::3]
            error = jnp.sqrt(jnp.mean((numerical - analytical) ** 2))
            errors.append(float(error))

        return (*errors, elapsed_time)

    def run_tests(self, prefix: str = "", visualize: bool = True) -> Dict[str, Tuple[List[float], float]]:
        """
        すべてのテスト関数に対してテストを実行
        
        Args:
            prefix: 出力ファイルの接頭辞
            visualize: 可視化を行うかどうか
            
        Returns:
            テスト結果の辞書 {関数名: ([1階誤差, 2階誤差, 3階誤差], 計算時間)}
        """
        results = {}
        
        print(f"Error Analysis Results for {self.solver_name}:")
        print("-" * 75)
        print(f"{'Function':<15} {'1st Der.':<12} {'2nd Der.':<12} {'3rd Der.':<12} {'Time (s)':<12}")
        print("-" * 75)
        
        total_errors = [0.0, 0.0, 0.0]
        total_time = 0.0
        
        for test_func in self.test_functions:
            errors = self.compute_errors(test_func)
            results[test_func.name] = (errors[:3], errors[3])
            
            print(
                f"{test_func.name:<15} {errors[0]:<12.2e} {errors[1]:<12.2e} {errors[2]:<12.2e} {errors[3]:<12.4f}"
            )
            
            # 誤差と時間を累積
            for i in range(3):
                total_errors[i] += errors[i]
            total_time += errors[3]
            
            # 結果の可視化
            if visualize:
                # グリッド点での計算
                n = self.grid_config.n_points
                h = self.grid_config.h
                x_start = self.x_range[0]
                x_points = jnp.array([x_start + i * h for i in range(n)])
                f_values = jnp.array([test_func.f(x) for x in x_points])
                numerical_derivatives = self.solver.solve(f_values)
                
                visualize_derivative_results(
                    test_func=test_func,
                    numerical_derivatives=numerical_derivatives,
                    grid_config=self.grid_config,
                    x_range=self.x_range,
                    solver_name=self.solver_name,
                    save_path=f"{prefix}{test_func.name.lower()}_results.png"
                )
        
        # 平均誤差と時間
        avg_errors = [e / len(self.test_functions) for e in total_errors]
        avg_time = total_time / len(self.test_functions)
        
        print("-" * 75)
        print(f"{'Average':<15} {avg_errors[0]:<12.2e} {avg_errors[1]:<12.2e} {avg_errors[2]:<12.2e} {avg_time:<12.4f}")
        print("-" * 75)
        
        return results