import jax.numpy as jnp
import time
import os
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
        test_functions: Optional[List[TestFunction]] = None,
        coeffs: Optional[List[float]] = None
    ):
        """
        Args:
            solver_class: テスト対象のCCDソルバークラス
            grid_config: グリッド設定
            x_range: x軸の範囲 (開始位置, 終了位置)
            solver_kwargs: ソルバーの初期化パラメータ
            test_functions: テスト関数のリスト (Noneの場合は標準関数セットを使用)
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用 (f = psi)
        """
        self.grid_config = grid_config
        self.x_range = x_range
        solver_kwargs = solver_kwargs or {}
        
        # 係数の設定
        self.coeffs = coeffs
        
        # solver_kwargsに係数を追加（存在する場合）
        if self.coeffs is not None:
            solver_kwargs['coeffs'] = self.coeffs
        
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
            (psi'の誤差, psi''の誤差, psi'''の誤差, 計算時間)
        """
        n = self.grid_config.n_points
        h = self.grid_config.h
        x_start = self.x_range[0]

        # グリッド点での関数値を計算
        x_points = jnp.array([x_start + i * h for i in range(n)])
        f_values = jnp.array([test_func.f(x) for x in x_points])

        # 計測開始
        start_time = time.time()
        
        # 数値解の計算（新しいAPIに対応）
        psi, psi_prime, psi_second, psi_third = self.solver.solve(f_values)
        
        # 計測終了
        elapsed_time = time.time() - start_time

        # 解析解の計算 (係数を考慮)
        if self.coeffs is not None:
            a, b, c, d = self.coeffs
            # f = a*psi + b*psi' + c*psi'' + d*psi''' の関係なので、
            # 解析解からpsiを求める必要がある
            # ここではシンプルなケースとして、解析解のpsiをそのまま使用
            analytical_psi = jnp.array([test_func.f(x) for x in x_points])
            analytical_df = jnp.array([test_func.df(x) for x in x_points])
            analytical_d2f = jnp.array([test_func.d2f(x) for x in x_points])
            analytical_d3f = jnp.array([test_func.d3f(x) for x in x_points])
        else:
            # デフォルトケース: f = psi
            analytical_psi = jnp.array([test_func.f(x) for x in x_points])
            analytical_df = jnp.array([test_func.df(x) for x in x_points])
            analytical_d2f = jnp.array([test_func.d2f(x) for x in x_points])
            analytical_d3f = jnp.array([test_func.d3f(x) for x in x_points])

        # 誤差の計算 (L2ノルム)
        error_psi = jnp.sqrt(jnp.mean((psi - analytical_psi) ** 2))
        error_df = jnp.sqrt(jnp.mean((psi_prime - analytical_df) ** 2))
        error_d2f = jnp.sqrt(jnp.mean((psi_second - analytical_d2f) ** 2))
        error_d3f = jnp.sqrt(jnp.mean((psi_third - analytical_d3f) ** 2))

        return float(error_df), float(error_d2f), float(error_d3f), elapsed_time

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
        
        coeff_str = "" if self.coeffs is None else f" (coeffs={self.coeffs})"
        print(f"Error Analysis Results for {self.solver_name}{coeff_str}:")
        print("-" * 75)
        print(f"{'Function':<15} {'1st Der.':<12} {'2nd Der.':<12} {'3rd Der.':<12} {'Time (s)':<12}")
        print("-" * 75)
        
        total_errors = [0.0, 0.0, 0.0]
        total_time = 0.0
        
        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)
        
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
                
                # 新しいAPIに対応
                psi, psi_prime, psi_second, psi_third = self.solver.solve(f_values)
                
                # visualize_derivative_results関数への入力も変更する必要があります
                visualize_derivative_results(
                    test_func=test_func,
                    f_values=f_values,  # 元の関数値
                    numerical_derivatives=(psi, psi_prime, psi_second, psi_third),  # タプルで渡す
                    grid_config=self.grid_config,
                    x_range=self.x_range,
                    solver_name=f"{self.solver_name}{coeff_str}",
                    save_path=f"results/{prefix}{test_func.name.lower()}_results.png"
                )
        
        # 平均誤差と時間
        avg_errors = [e / len(self.test_functions) for e in total_errors]
        avg_time = total_time / len(self.test_functions)
        
        print("-" * 75)
        print(f"{'Average':<15} {avg_errors[0]:<12.2e} {avg_errors[1]:<12.2e} {avg_errors[2]:<12.2e} {avg_time:<12.4f}")
        print("-" * 75)
        
        return results