"""
高精度コンパクト差分法 (CCD) の2次元テスターモジュール

このモジュールは、2次元のCCDソルバーのテスト機能を提供します。
"""

import numpy as np

from core.base.base_tester import CCDTester
from core.solver.solver2d import CCDSolver2D
from equation_set.equation_sets import DerivativeEquationSet2D
from test_function.test_function1d import TestFunction1DFactory
from test_function.test_function2d import TestFunction2DFactory


class CCDTester2D(CCDTester):
    """2D CCDテスター"""
    
    def __init__(self, grid):
        """
        2Dテスターを初期化
        
        Args:
            grid: 2Dグリッドオブジェクト
        """
        if not grid.is_2d:
            raise ValueError("2Dテスターは1Dグリッドでは使用できません")
        super().__init__(grid)
    
    def _create_solver(self):
        """2D用ソルバー作成"""
        self.solver = CCDSolver2D(self.equation_set, self.grid, backend=self.backend)
        
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
            int: 2 (2次元)
        """
        return 2
    
    def get_test_function(self, func_name):
        """
        2Dテスト関数取得
        
        Args:
            func_name: テスト関数名
            
        Returns:
            TestFunction2D: 取得したテスト関数
        """
        # 基本2D関数を検索
        funcs = TestFunction2DFactory.create_standard_functions()
        func = next((f for f in funcs if f.name == func_name), None)
        if func:
            return func
        
        # 1D関数から変換を試みる
        funcs_1d = TestFunction1DFactory.create_standard_functions()
        func_1d = next((f for f in funcs_1d if f.name == func_name), None)
        if func_1d:
            from test_function.test_function2d import TestFunction2D
            return TestFunction2D.from_1d(func_1d, method='product')
        
        # デフォルト関数を返す
        print(f"警告: 2D関数 '{func_name}' が見つかりません。デフォルト関数を使用します。")
        return funcs[0]
    
    def _build_rhs(self, test_func):
        """
        2D右辺構築
        
        Args:
            test_func: テスト関数
            
        Returns:
            np.ndarray: 右辺ベクトル
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        
        # ソース項 (方程式タイプに依存)
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet2D)
        f_values = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x, y = self.grid.get_point(i, j)
                f_values[i, j] = test_func.f(x, y) if is_derivative else (
                    test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)
                )
        
        # 境界値
        boundary = {
            'left_dirichlet': np.array([test_func.f(x_min, y) for y in self.grid.y]),
            'right_dirichlet': np.array([test_func.f(x_max, y) for y in self.grid.y]),
            'bottom_dirichlet': np.array([test_func.f(x, y_min) for x in self.grid.x]),
            'top_dirichlet': np.array([test_func.f(x, y_max) for x in self.grid.x]),
            'left_neumann': np.array([test_func.df_dx(x_min, y) for y in self.grid.y]),
            'right_neumann': np.array([test_func.df_dx(x_max, y) for y in self.grid.y]),
            'bottom_neumann': np.array([test_func.df_dy(x, y_min) for x in self.grid.x]),
            'top_neumann': np.array([test_func.df_dy(x, y_max) for x in self.grid.x])
        }
        
        self._init_solver()
        return self.solver.rhs_builder.build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact(self, test_func):
        """
        2D厳密解計算
        
        Args:
            test_func: テスト関数
            
        Returns:
            np.ndarray: 厳密解ベクトル
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        exact = np.zeros(nx * ny * 7)  # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy] x (nx * ny)
        
        for j in range(ny):
            for i in range(nx):
                idx = (j * nx + i) * 7
                x, y = self.grid.get_point(i, j)
                
                exact[idx] = test_func.f(x, y)          # ψ
                exact[idx+1] = test_func.df_dx(x, y)     # ψ_x
                exact[idx+2] = test_func.d2f_dx2(x, y)   # ψ_xx
                exact[idx+3] = test_func.d3f_dx3(x, y)   # ψ_xxx
                exact[idx+4] = test_func.df_dy(x, y)     # ψ_y
                exact[idx+5] = test_func.d2f_dy2(x, y)   # ψ_yy
                exact[idx+6] = test_func.d3f_dy3(x, y)   # ψ_yyy
                
        return exact
    
    def _process_solution(self, test_func):
        """
        2D解処理（NumPy/CuPy互換性を考慮）
        
        Args:
            test_func: テスト関数
            
        Returns:
            Dict: 処理結果を含む辞書
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        X, Y = self.grid.get_points()
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet2D)

        # 厳密解準備（NumPy配列として）
        exact = {
            'psi': np.zeros((nx, ny)),
            'psi_x': np.zeros((nx, ny)),
            'psi_y': np.zeros((nx, ny)),
            'psi_xx': np.zeros((nx, ny)),
            'psi_yy': np.zeros((nx, ny)),
            'psi_xxx': np.zeros((nx, ny)),
            'psi_yyy': np.zeros((nx, ny))
        }

        # 各点で厳密値計算
        for i in range(nx):
            for j in range(ny):
                x, y = self.grid.get_point(i, j)
                exact['psi'][i, j] = test_func.f(x, y)
                exact['psi_x'][i, j] = test_func.df_dx(x, y)
                exact['psi_y'][i, j] = test_func.df_dy(x, y)
                exact['psi_xx'][i, j] = test_func.d2f_dx2(x, y)
                exact['psi_yy'][i, j] = test_func.d2f_dy2(x, y)
                exact['psi_xxx'][i, j] = test_func.d3f_dx3(x, y)
                exact['psi_yyy'][i, j] = test_func.d3f_dy3(x, y)

        # 右辺値準備
        f_values = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x, y = self.grid.get_point(i, j)
                f_values[i, j] = test_func.f(x, y) if is_derivative else (
                    test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)
                )

        # 境界条件
        boundary = {
            'left_dirichlet': np.array([test_func.f(x_min, y) for y in self.grid.y]),
            'right_dirichlet': np.array([test_func.f(x_max, y) for y in self.grid.y]),
            'bottom_dirichlet': np.array([test_func.f(x, y_min) for x in self.grid.x]),
            'top_dirichlet': np.array([test_func.f(x, y_max) for x in self.grid.x]),
            'left_neumann': np.array([test_func.df_dx(x_min, y) for y in self.grid.y]),
            'right_neumann': np.array([test_func.df_dx(x_max, y) for y in self.grid.y]),
            'bottom_neumann': np.array([test_func.df_dy(x, y_min) for x in self.grid.x]),
            'top_neumann': np.array([test_func.df_dy(x, y_max) for x in self.grid.x])
        }

        # 厳密解を初期値として使用する場合は、直接オプションを渡す
        if self.perturbation_level is not None and self.solver_method != 'direct' and self.exact_solution is not None:
            print("直接x0を渡してsolve()を呼び出します")
            # オプションを作成して厳密解（摂動あり/なし）を含める
            solve_options = {
                'tol': self.solver_options.get('tol', 1e-10),
                'maxiter': self.solver_options.get('maxiter', 1000),
                'x0': self.exact_solution  # 厳密解を直接渡す
            }
            
            # 解計算（カスタムオプション付き）
            sol = self.solver.solve_with_options(
                analyze_before_solve=False, 
                f_values=f_values, 
                solve_options=solve_options, 
                **boundary
            )
        else:
            # 通常の解計算
            sol = self.solver.solve(
                analyze_before_solve=False, 
                f_values=f_values, 
                **boundary
            )
            
        psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy = sol

        # NumPy配列に変換（CuPyの場合）
        psi = self._to_numpy(psi)
        psi_x = self._to_numpy(psi_x)
        psi_y = self._to_numpy(psi_y)
        psi_xx = self._to_numpy(psi_xx)
        psi_yy = self._to_numpy(psi_yy)
        psi_xxx = self._to_numpy(psi_xxx)
        psi_yyy = self._to_numpy(psi_yyy)

        # 誤差計算（NumPy配列として）
        errors = [
            float(np.max(np.abs(psi - exact['psi']))),
            float(np.max(np.abs(psi_x - exact['psi_x']))),
            float(np.max(np.abs(psi_y - exact['psi_y']))),
            float(np.max(np.abs(psi_xx - exact['psi_xx']))),
            float(np.max(np.abs(psi_yy - exact['psi_yy']))),
            float(np.max(np.abs(psi_xxx - exact['psi_xxx']))),
            float(np.max(np.abs(psi_yyy - exact['psi_yyy'])))
        ]

        return {
            "function": test_func.name,
            "numerical": [psi, psi_x, psi_y, psi_xx, psi_yy, psi_xxx, psi_yyy],
            "exact": [v for v in exact.values()],
            "errors": errors
        }