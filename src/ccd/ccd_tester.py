"""
シンプル化されたCCD法テスターモジュール

CCDソルバー実装のテスト機能を提供します。
"""

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
        solver_class: Type[CCDSolver],  # type: ignore
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
        # 元のグリッド設定を保存（境界条件なし）
        self.original_grid_config = grid_config
        self.x_range = x_range
        self.coeffs = coeffs if coeffs is not None else [1.0, 0.0, 0.0, 0.0]
        
        # 境界値を計算するためのヘルパー変数を設定
        self.x_start = x_range[0]
        self.x_end = x_range[1]
        
        # ソルバーパラメータを保存
        self.solver_kwargs = solver_kwargs or {}
        self.solver_class = solver_class
        
        # テスト関数の設定
        self.test_functions = test_functions or TestFunctionFactory.create_standard_functions()
        
        # 最初のテスト関数を使って、境界条件の値を設定
        first_test_func = self.test_functions[0]
        
        # ディリクレ境界条件とノイマン境界条件の両方に対応
        boundary_grid_config = self._create_boundary_grid_config(first_test_func)
        
        # ソルバーの初期化
        solver_kwargs_copy = self.solver_kwargs.copy()
        if 'coeffs' not in solver_kwargs_copy and self.coeffs is not None:
            solver_kwargs_copy['coeffs'] = self.coeffs
            
        self.solver = solver_class(boundary_grid_config, **solver_kwargs_copy)
        self.solver_name = solver_class.__name__

    def _create_boundary_grid_config(self, test_func: TestFunction) -> GridConfig:
        """
        テスト関数から境界条件を含むグリッド設定を作成
        
        Args:
            test_func: テスト関数
            
        Returns:
            境界条件が設定されたGridConfig
        """
        # ディリクレ境界条件の値（関数値）
        dirichlet_left = test_func.f(self.x_start)
        dirichlet_right = test_func.f(self.x_end)
        
        # ノイマン境界条件の値（微分値）
        neumann_left = test_func.df(self.x_start)
        neumann_right = test_func.df(self.x_end)
        
        # 新しいグリッド設定を作成
        return GridConfig(
            n_points=self.original_grid_config.n_points,
            h=self.original_grid_config.h,
            dirichlet_values=[dirichlet_left, dirichlet_right],
            neumann_values=[neumann_left, neumann_right]
        )

    def compute_errors(self, test_func: TestFunction) -> Tuple[float, float, float, float]:
        """
        各導関数の誤差を計算
        
        Args:
            test_func: テスト関数
            
        Returns:
            (psi'の誤差, psi''の誤差, psi'''の誤差, 計算時間)
        """
        # テスト関数に合わせた境界条件でグリッド設定を更新
        boundary_grid_config = self._create_boundary_grid_config(test_func)
        
        n = boundary_grid_config.n_points
        h = boundary_grid_config.h
        x_start = self.x_range[0]

        # グリッド点でのx座標と解析解を計算
        x_points = jnp.array([x_start + i * h for i in range(n)])
        analytical_psi = jnp.array([test_func.f(x) for x in x_points])
        analytical_df = jnp.array([test_func.df(x) for x in x_points])
        analytical_d2f = jnp.array([test_func.d2f(x) for x in x_points])
        analytical_d3f = jnp.array([test_func.d3f(x) for x in x_points])
        
        # ソルバーのグリッド設定も更新（新しいソルバーを作成）
        solver_kwargs_copy = self.solver_kwargs.copy()
        if 'coeffs' not in solver_kwargs_copy and self.coeffs is not None:
            solver_kwargs_copy['coeffs'] = self.coeffs
            
        self.solver = self.solver_class(boundary_grid_config, **solver_kwargs_copy)
        
        # 係数に基づいて入力関数値を計算
        a, b, c, d = self.coeffs
        f_values = (a * analytical_psi + b * analytical_df + 
                   c * analytical_d2f + d * analytical_d3f)

        # 計測開始
        start_time = time.time()
        
        # 数値解の計算
        psi, psi_prime, psi_second, psi_third = self.solver.solve(f_values)
        
        # 計測終了
        elapsed_time = time.time() - start_time

        # 誤差の計算 (L2ノルム)
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
        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)
        
        results = {}
        total_errors = [0.0, 0.0, 0.0]
        total_time = 0.0
        
        # 結果テーブルのヘッダーを表示
        coeff_str = f" (coeffs={self.coeffs})"
        print(f"Error Analysis Results for {self.solver_name}{coeff_str}:")
        print("-" * 75)
        print(f"{'Function':<15} {'1st Der.':<12} {'2nd Der.':<12} {'3rd Der.':<12} {'Time (s)':<12}")
        print("-" * 75)
        
        for test_func in self.test_functions:
            # 誤差と時間を計算
            errors = self.compute_errors(test_func)
            results[test_func.name] = (errors[:3], errors[3])
            
            # 結果を表示
            print(f"{test_func.name:<15} {errors[0]:<12.2e} {errors[1]:<12.2e} {errors[2]:<12.2e} {errors[3]:<12.4f}")
            
            # 誤差と時間を累積
            for i in range(3):
                total_errors[i] += errors[i]
            total_time += errors[3]
            
            # 可視化が有効な場合、結果をプロット
            if visualize:
                # 境界条件を含むグリッド設定を作成
                boundary_grid_config = self._create_boundary_grid_config(test_func)
                
                # グリッド点での計算用データを準備
                n = boundary_grid_config.n_points
                h = boundary_grid_config.h
                x_start = self.x_range[0]
                x_points = jnp.array([x_start + i * h for i in range(n)])
                
                # 解析解と入力関数値を計算
                analytical_psi = jnp.array([test_func.f(x) for x in x_points])
                analytical_df = jnp.array([test_func.df(x) for x in x_points])
                analytical_d2f = jnp.array([test_func.d2f(x) for x in x_points])
                analytical_d3f = jnp.array([test_func.d3f(x) for x in x_points])
                
                # 係数に基づいて入力関数値を計算
                a, b, c, d = self.coeffs
                f_values = (a * analytical_psi + b * analytical_df + 
                           c * analytical_d2f + d * analytical_d3f)
                
                # 数値解の計算（ソルバーは既にcompute_errors内で更新済み）
                psi, psi_prime, psi_second, psi_third = self.solver.solve(f_values)
                
                # 微分モードの名前を取得
                mode_name = self._get_mode_name()
                
                # 解析解を準備
                analytical_derivatives = (analytical_psi, analytical_df, analytical_d2f, analytical_d3f)
                
                visualize_derivative_results(
                    test_func=test_func,
                    f_values=f_values,
                    numerical_derivatives=(psi, psi_prime, psi_second, psi_third),
                    analytical_derivatives=analytical_derivatives,
                    grid_config=boundary_grid_config,
                    x_range=self.x_range,
                    solver_name=f"{self.solver_name} ({mode_name})",
                    save_path=f"results/{prefix}{test_func.name.lower()}_results.png"
                )
        
        # 平均誤差と時間を表示
        avg_errors = [e / len(self.test_functions) for e in total_errors]
        avg_time = total_time / len(self.test_functions)
        
        print("-" * 75)
        print(f"{'Average':<15} {avg_errors[0]:<12.2e} {avg_errors[1]:<12.2e} {avg_errors[2]:<12.2e} {avg_time:<12.4f}")
        print("-" * 75)
        
        # 境界条件の情報を表示
        print(f"\n境界条件: ディリクレ + ノイマン (テスト関数の境界値を使用)")
        print(f"各テスト関数ごとに境界値を動的に設定")
        
        return results
    
    def _get_mode_name(self) -> str:
        """係数に基づく微分モード名を取得"""
        mode_names = {
            (1, 0, 0, 0): "PSI",
            (0, 1, 0, 0): "PSI'",
            (0, 0, 1, 0): "PSI''",
            (1, 1, 0, 0): "PSI+PSI'",
            (1, 0, 1, 0): "PSI+PSI''",
            (0, 1, 1, 0): "PSI'+PSI''",
            (1, 1, 1, 0): "PSI+PSI'+PSI''"
        }
        
        coeffs_tuple = tuple(self.coeffs)
        return mode_names.get(coeffs_tuple, f"coeffs={self.coeffs}")