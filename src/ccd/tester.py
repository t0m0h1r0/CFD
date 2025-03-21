from abc import ABC, abstractmethod
import os
import time
import numpy as np
from grid import Grid
from ccd_solver import CCDSolver1D, CCDSolver2D
from equation_sets import EquationSet, DerivativeEquationSet1D, DerivativeEquationSet2D
from scaling import plugin_manager
from matrix_visualizer import MatrixVisualizer


class CCDTester(ABC):
    """CCDテスト基底クラス"""

    def __init__(self, grid):
        self.grid = grid
        self.solver = None
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.equation_set = None
        self.backend = "cpu"  # デフォルトバックエンドをCPUに設定
        self.results_dir = "results"
        self.matrix_basename = None  # 行列可視化用のベース名
        self.perturbation_level = None  # 厳密解に加える摂動レベル (None=厳密解を使用しない、0=摂動なし)
        self.exact_solution = None  # 厳密解を保存
        os.makedirs(self.results_dir, exist_ok=True)

    def setup(self, equation="poisson", method="direct", options=None, scaling=None, backend="cpu"):
        """テスター設定のショートカット"""
        self.equation_set = EquationSet.create(equation, dimension=self.get_dimension())
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling
        self.backend = backend
        return self
        
    def set_solver_options(self, method, options=None, analyze_matrix=False):
        """ソルバーオプションを設定
        
        Args:
            method: 解法メソッド名
            options: ソルバーオプション辞書
            analyze_matrix: 解く前に行列を分析するかどうか
        
        Returns:
            self: メソッドチェーン用
        """
        self.solver_method = method
        self.solver_options = options or {}
        self.analyze_matrix = analyze_matrix
        
        # バックエンドを保存（オプションから取得）
        if options and "backend" in options:
            self.backend = options["backend"]
        
        # ソルバーがすでに作成されている場合は、オプションを設定
        if hasattr(self, 'solver') and self.solver:
            self.solver.set_solver(
                method=self.solver_method, 
                options=self.solver_options, 
                scaling_method=self.scaling_method
            )
        
        return self
        
    def set_equation_set(self, equation_set_name):
        """方程式セットを設定"""
        self.equation_set = EquationSet.create(equation_set_name, dimension=self.get_dimension())
        return self
        
    def _add_perturbation(self, exact_sol, level):
        """厳密解に指定レベルの摂動を加える
        
        Args:
            exact_sol: 厳密解ベクトル
            level: 摂動レベル (0～1の間の値)
            
        Returns:
            摂動が加えられた解ベクトル
        """
        if level == 0:
            return exact_sol  # 摂動なし
            
        # ランダムシードを固定（再現性のため）
        np.random.seed(42)
        
        # 摂動を加える（各要素に対して±levelの範囲で乱数を乗算）
        # 例：level=0.1なら、各要素は厳密値の±10%の範囲で変動
        perturbed_sol = exact_sol * (1.0 + level * (2.0 * np.random.random(exact_sol.shape) - 1.0))
        
        # どれくらい摂動したかの情報を表示
        relative_change = np.abs(perturbed_sol - exact_sol) / (np.abs(exact_sol) + 1e-15)
        avg_change = np.mean(relative_change) * 100
        max_change = np.max(relative_change) * 100
        print(f"摂動を加えました: 平均 {avg_change:.2f}%, 最大 {max_change:.2f}%")
        
        return perturbed_sol

    def run_test(self, test_func):
        """テスト実行"""
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
        
        if not self.equation_set:
            self.equation_set = EquationSet.create("poisson", dimension=self.get_dimension())

        self._init_solver()
        
        # 厳密解を初期値として使用する場合、計算して摂動を加える
        if self.perturbation_level is not None and self.solver_method != 'direct':
            self.exact_solution = self._compute_exact(test_func)
            
            # 厳密解に摂動を加える
            perturbed_solution = self._add_perturbation(self.exact_solution, self.perturbation_level)
            
            print(f"厳密解を初期値として使用 (サイズ: {perturbed_solution.shape}, メソッド: {self.solver_method})")
            # self.solver_optionsにx0を設定（ここでは後のために保存するだけ）
            options = self.solver_options.copy()
            options['x0'] = perturbed_solution
            self.solver.set_solver(
                method=self.solver_method, 
                options=options, 
                scaling_method=self.scaling_method
            )
        else:
            self.exact_solution = None
        
        # ソルバーオプションの再適用（念のため）
        if hasattr(self, 'solver') and self.solver:
            self.solver.set_solver(
                method=self.solver_method, 
                options=self.solver_options, 
                scaling_method=self.scaling_method
            )
        
        return self._process_solution(test_func)
    
    def run_test_with_options(self, test_func):
        """旧APIとの互換性用"""
        return self.run_test(test_func)
    
    def run_convergence_test(self, test_func, grid_sizes, x_range, y_range=None):
        """格子収束性テスト"""
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
            
        results = {}
        settings = {
            'method': self.solver_method,
            'options': self.solver_options,
            'equation_set': self.equation_set,
            'scaling': self.scaling_method,
            'backend': self.backend,
            'perturbation_level': self.perturbation_level
        }

        for n in grid_sizes:
            grid = self._create_grid(n, x_range, y_range)
            tester = (CCDTester1D(grid) if self.get_dimension() == 1 else CCDTester2D(grid))
            tester.setup(
                equation=settings['equation_set'].__class__.__name__.replace('EquationSet', '').replace('1D', '').replace('2D', '').lower() 
                if settings['equation_set'] else "poisson",
                method=settings['method'],
                options=settings['options'],
                scaling=settings['scaling'],
                backend=settings['backend']
            )
            tester.perturbation_level = settings['perturbation_level']
            result = tester.run_test(test_func)
            results[n] = result["errors"]

        return results
    
    def visualize_matrix_system(self, test_func=None):
        """行列システム可視化 (MatrixVisualizer を使用)"""
        if test_func is None:
            # デフォルトテスト関数
            from test_functions import TestFunctionFactory
            test_func = (TestFunctionFactory.create_standard_1d_functions()[3] if self.get_dimension() == 1 
                      else TestFunctionFactory.create_standard_2d_functions()[0])
        elif isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
        
        self._init_solver()
        A = self.matrix_A
        b = self._build_rhs(test_func)
        
        # 解計算
        x = self._solve_for_visualization(A, b)
            
        # 厳密解計算
        exact_x = self._compute_exact(test_func) if x is not None else None
        
        # 外部ビジュアライザーを使用して可視化
        visualizer = MatrixVisualizer(self.results_dir)
        eq_name = self.equation_set.__class__.__name__.replace("EquationSet", "").replace("1D", "").replace("2D", "")
        
        # 標準化されたファイル名を使用
        if self.matrix_basename:
            title = self.matrix_basename
        else:
            title = f"{eq_name}_{test_func.name}"
            
        return visualizer.visualize(
            A, b, x, exact_x, title,
            self.get_dimension(), self.scaling_method
        )
    
    def _solve_for_visualization(self, A, b):
        """可視化用に方程式系を解く"""
        try:
            # LinearSolverを使用
            return self.solver.linear_solver.solve(b, method=self.solver_method, options=self.solver_options)
        except Exception as e:
            print(f"Solver error: {e}")
            # フォールバック: 直接SciPyを使用
            try:
                from scipy.sparse.linalg import spsolve
                return spsolve(A, b)
            except:
                print("SciPy fallback solver also failed")
                return None
    
    def _init_solver(self):
        """ソルバー初期化"""
        if not self.solver:
            self._create_solver()
            
        if self.solver_method != "direct" or self.solver_options or self.scaling_method:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options, 
                                  scaling_method=self.scaling_method)
    
    def _create_grid(self, n, x_range, y_range=None):
        """グリッド作成"""
        if self.get_dimension() == 1:
            return Grid(n, x_range=x_range)
        return Grid(n, n, x_range=x_range, y_range=y_range or x_range)
    
    # NumPyに変換するユーティリティ関数
    def _to_numpy(self, arr):
        """CuPy配列をNumPy配列に変換（必要な場合のみ）"""
        return arr.get() if hasattr(arr, 'get') else arr
    
    @abstractmethod
    def get_dimension(self):
        """次元を返す"""
        pass
        
    @abstractmethod
    def get_test_function(self, func_name):
        """テスト関数取得"""
        pass
    
    @abstractmethod
    def _create_solver(self):
        """ソルバー作成"""
        pass
    
    @abstractmethod
    def _build_rhs(self, test_func):
        """右辺ベクトル構築"""
        pass
    
    @abstractmethod
    def _compute_exact(self, test_func):
        """厳密解計算"""
        pass
    
    @abstractmethod
    def _process_solution(self, test_func):
        """解を処理して結果を返す"""
        pass


class CCDTester1D(CCDTester):
    """1D CCDテスター"""
    
    def __init__(self, grid):
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
        return 1
    
    def get_test_function(self, func_name):
        """1Dテスト関数取得"""
        from test_functions import TestFunctionFactory
        
        funcs = TestFunctionFactory.create_standard_1d_functions()
        func = next((f for f in funcs if f.name == func_name), None)
        
        if not func:
            print(f"Warning: 1D function '{func_name}' not found. Using default function.")
            func = funcs[0]
            
        return func
    
    def _build_rhs(self, test_func):
        """1D右辺構築"""
        x_min, x_max = self.grid.x_min, self.grid.x_max
        
        # ソース項
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet1D)
        f_values = np.array([test_func.f(x) if is_derivative else test_func.d2f(x) for x in self.grid.x])
        
        # 境界値
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }
        
        self._init_solver()
        return self.solver.rhs_builder.build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact(self, test_func):
        """1D厳密解計算"""
        n_points = self.grid.n_points
        exact = np.zeros(n_points * 4)
        
        for i in range(n_points):
            xi = self.grid.get_point(i)
            exact[i*4] = test_func.f(xi)
            exact[i*4+1] = test_func.df(xi)
            exact[i*4+2] = test_func.d2f(xi)
            exact[i*4+3] = test_func.d3f(xi)
            
        return exact
    
    def _process_solution(self, test_func):
        """1D解処理（NumPy/CuPy互換性を修正）"""
        x = self.grid.get_points()
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet1D)
        
        # NumPy配列に変換
        x_np = self._to_numpy(x)

        # 厳密解計算（NumPy配列として）
        exact_psi = np.array([test_func.f(xi) for xi in x_np])
        exact_psi_prime = np.array([test_func.df(xi) for xi in x_np])
        exact_psi_second = np.array([test_func.d2f(xi) for xi in x_np])
        exact_psi_third = np.array([test_func.d3f(xi) for xi in x_np])
        
        # 解計算のための入力準備
        x_min, x_max = self.grid.x_min, self.grid.x_max
        f_values = np.array([test_func.f(xi) if is_derivative else test_func.d2f(xi) for xi in x_np])
        
        # 境界条件
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }

        # 厳密解を初期値として使用する場合は、直接オプションを渡す
        if self.perturbation_level is not None and self.solver_method != 'direct' and self.exact_solution is not None:
            print(f"直接x0を渡してsolve()を呼び出します")
            # オプションを作成して厳密解（摂動あり/なし）を含める
            solve_options = {
                'tol': self.solver_options.get('tol', 1e-10),
                'maxiter': self.solver_options.get('maxiter', 1000),
                'x0': self.exact_solution  # 厳密解を直接渡す
            }
            
            # 解計算（カスタムオプション付き）
            psi, psi_prime, psi_second, psi_third = self.solver.solve_with_options(
                analyze_before_solve=False, f_values=f_values, 
                solve_options=solve_options, **boundary
            )
        else:
            # 通常の解計算
            psi, psi_prime, psi_second, psi_third = self.solver.solve(
                analyze_before_solve=False, f_values=f_values, **boundary
            )

        # NumPy配列に変換（CuPyの場合）
        psi = self._to_numpy(psi)
        psi_prime = self._to_numpy(psi_prime)
        psi_second = self._to_numpy(psi_second)
        psi_third = self._to_numpy(psi_third)

        # 誤差計算（NumPy配列として）
        err_psi = float(np.max(np.abs(psi - exact_psi)))
        err_psi_prime = float(np.max(np.abs(psi_prime - exact_psi_prime)))
        err_psi_second = float(np.max(np.abs(psi_second - exact_psi_second)))
        err_psi_third = float(np.max(np.abs(psi_third - exact_psi_third)))

        return {
            "function": test_func.name,
            "numerical": [psi, psi_prime, psi_second, psi_third],
            "exact": [exact_psi, exact_psi_prime, exact_psi_second, exact_psi_third],
            "errors": [err_psi, err_psi_prime, err_psi_second, err_psi_third],
        }


class CCDTester2D(CCDTester):
    """2D CCDテスター"""
    
    def __init__(self, grid):
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
        return 2
    
    def get_test_function(self, func_name):
        """2Dテスト関数取得"""
        from test_functions import TestFunctionFactory, TestFunction
        
        # 基本2D関数
        funcs = TestFunctionFactory.create_standard_2d_functions()
        func = next((f for f in funcs if f.name == func_name), None)
        if func:
            return func
        
        # 1D関数から変換
        funcs_1d = TestFunctionFactory.create_standard_1d_functions()
        func_1d = next((f for f in funcs_1d if f.name == func_name), None)
        if func_1d:
            return TestFunction.from_1d_to_2d(func_1d, method='product')
        
        # デフォルト
        print(f"Warning: 2D function '{func_name}' not found. Using default function.")
        return funcs[0]
    
    def _build_rhs(self, test_func):
        """2D右辺構築"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        
        # ソース項
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet2D)
        f_values = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x, y = self.grid.get_point(i, j)
                f_values[i, j] = test_func.f(x, y) if is_derivative else (test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y))
        
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
        """2D厳密解計算"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        exact = np.zeros(nx * ny * 7)
        
        for j in range(ny):
            for i in range(nx):
                idx = (j * nx + i) * 7
                x, y = self.grid.get_point(i, j)
                exact[idx] = test_func.f(x, y)
                exact[idx+1] = test_func.df_dx(x, y)
                exact[idx+2] = test_func.d2f_dx2(x, y)
                exact[idx+3] = test_func.d3f_dx3(x, y)
                exact[idx+4] = test_func.df_dy(x, y)
                exact[idx+5] = test_func.d2f_dy2(x, y)
                exact[idx+6] = test_func.d3f_dy3(x, y)
                
        return exact
    
    def _process_solution(self, test_func):
        """2D解処理（NumPy/CuPy互換性を修正）"""
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
                f_values[i, j] = test_func.f(x, y) if is_derivative else (test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y))

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
            print(f"直接x0を渡してsolve()を呼び出します")
            # オプションを作成して厳密解（摂動あり/なし）を含める
            solve_options = {
                'tol': self.solver_options.get('tol', 1e-10),
                'maxiter': self.solver_options.get('maxiter', 1000),
                'x0': self.exact_solution  # 厳密解を直接渡す
            }
            
            # 解計算（カスタムオプション付き）
            sol = self.solver.solve_with_options(
                analyze_before_solve=False, f_values=f_values, 
                solve_options=solve_options, **boundary
            )
        else:
            # 通常の解計算
            sol = self.solver.solve(
                analyze_before_solve=False, f_values=f_values, **boundary
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
            "errors": errors,
        }