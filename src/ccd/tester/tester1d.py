"""
高精度コンパクト差分法 (CCD) の1次元テスターモジュール

このモジュールは、1次元のCCDソルバーのテスト機能を提供します。
"""

import numpy as np
from core.base.base_tester import BaseTester
from core.solver.solver1d import CCDSolver1D
from equation_set.equation_sets import DerivativeEquationSet1D
from test_function.test_function1d import TestFunction1DFactory


class CCDTester1D(BaseTester):
    """1次元CCDテスタークラス"""
    
    def __init__(self, grid):
        """
        1Dテスターを初期化
        
        Args:
            grid: 1Dグリッドオブジェクト
        """
        if grid.is_2d:
            raise ValueError("1Dテスターは2Dグリッドでは使用できません")
        super().__init__(grid)
    
    def _create_solver(self):
        """1D用ソルバー作成"""
        self.solver = CCDSolver1D(self.equation_set, self.grid, backend=self.backend)
        
        # ソルバーオプションを適用
        if self.solver_method != "direct" or self.solver_options or self.scaling_method:
            self.solver.set_solver(
                method=self.solver_method, 
                options=self.solver_options, 
                scaling_method=self.scaling_method
            )
        
        # システム行列への参照を保持
        self.matrix_A = self.solver.matrix_A
            
    def get_dimension(self):
        """
        次元を返す
        
        Returns:
            int: 1 (1次元)
        """
        return 1
    
    def get_test_function(self, func_name):
        """
        1Dテスト関数取得
        
        Args:
            func_name: テスト関数名
            
        Returns:
            TestFunction1D: 取得したテスト関数
        """
        funcs = TestFunction1DFactory.create_standard_functions()
        func = next((f for f in funcs if f.name == func_name), None)
        
        if not func:
            print(f"警告: 1D関数 '{func_name}' が見つかりません。デフォルト関数を使用します。")
            func = funcs[0]
            
        return func
    
    def _build_rhs(self, test_func):
        """
        1D右辺ベクトル構築（簡略化）
        
        Args:
            test_func: テスト関数
            
        Returns:
            np.ndarray: 右辺ベクトル
        """
        x_min, x_max = self.grid.x_min, self.grid.x_max
        
        # ソース項の計算
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet1D)
        f_values = np.array([
            test_func.f(x) if is_derivative else test_func.d2f(x) 
            for x in self.grid.x
        ])
        
        # 境界条件
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }
        
        self._init_solver()
        return self.solver.rhs_builder.build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact(self, test_func):
        """
        1D厳密解計算（簡略化）
        
        Args:
            test_func: テスト関数
            
        Returns:
            np.ndarray: 厳密解ベクトル
        """
        n_points = self.grid.n_points
        exact = np.zeros(n_points * 4)  # [ψ, ψ', ψ'', ψ'''] x n_points
        
        for i in range(n_points):
            xi = self.grid.get_point(i)
            base_idx = i * 4
            
            exact[base_idx] = test_func.f(xi)      # ψ
            exact[base_idx+1] = test_func.df(xi)    # ψ'
            exact[base_idx+2] = test_func.d2f(xi)   # ψ''
            exact[base_idx+3] = test_func.d3f(xi)   # ψ'''
            
        return exact
    
    def _process_solution(self, test_func):
        """
        1D解処理（簡略化）
        
        Args:
            test_func: テスト関数
            
        Returns:
            Dict: 処理結果を含む辞書
        """
        x = self.grid.get_points()
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet1D)
        
        # NumPy配列に変換
        x_np = self._to_numpy(x)

        # 厳密解計算
        exact_psi = np.array([test_func.f(xi) for xi in x_np])
        exact_psi_prime = np.array([test_func.df(xi) for xi in x_np])
        exact_psi_second = np.array([test_func.d2f(xi) for xi in x_np])
        exact_psi_third = np.array([test_func.d3f(xi) for xi in x_np])
        
        # 右辺の準備
        x_min, x_max = self.grid.x_min, self.grid.x_max
        f_values = np.array([
            test_func.f(xi) if is_derivative else test_func.d2f(xi) 
            for xi in x_np
        ])
        
        # 境界条件
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }

        # ソルバーオプション設定と実行
        if self.perturbation_level is not None and self.solver_method != 'direct' and self.exact_solution is not None:
            solve_options = {
                'tol': self.solver_options.get('tol', 1e-10),
                'maxiter': self.solver_options.get('maxiter', 1000),
                'x0': self.exact_solution
            }
            psi, psi_prime, psi_second, psi_third = self.solver.solve_with_options(
                analyze_before_solve=False, 
                f_values=f_values, 
                solve_options=solve_options, 
                **boundary
            )
        else:
            psi, psi_prime, psi_second, psi_third = self.solver.solve(
                analyze_before_solve=False, 
                f_values=f_values, 
                **boundary
            )

        # NumPy配列に変換
        psi = self._to_numpy(psi)
        psi_prime = self._to_numpy(psi_prime)
        psi_second = self._to_numpy(psi_second)
        psi_third = self._to_numpy(psi_third)

        # 誤差計算
        exact_vals = [exact_psi, exact_psi_prime, exact_psi_second, exact_psi_third]
        numerical_vals = [psi, psi_prime, psi_second, psi_third]
        errors = [float(np.max(np.abs(n - e))) for n, e in zip(numerical_vals, exact_vals)]

        return {
            "function": test_func.name,
            "numerical": numerical_vals,
            "exact": exact_vals,
            "errors": errors,
        }