from abc import ABC, abstractmethod
import os
import cupy as cp
from grid import Grid
from solver import CCDSolver1D, CCDSolver2D
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
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def setup(self, equation="poisson", method="direct", options=None, scaling=None):
        """テスター設定のショートカット"""
        self.equation_set = EquationSet.create(equation, dimension=self.get_dimension())
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling
        return self
        
    def set_solver_options(self, method, options=None, analyze_matrix=False):
        """cli.py との互換性のため"""
        self.solver_method = method
        self.solver_options = options or {}
        return self
        
    def set_equation_set(self, equation_set_name):
        """cli.py との互換性のため"""
        self.equation_set = EquationSet.create(equation_set_name, dimension=self.get_dimension())
        return self

    def run_test(self, test_func):
        """テスト実行"""
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
        
        if not self.equation_set:
            self.equation_set = EquationSet.create("poisson", dimension=self.get_dimension())

        self._init_solver()
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
            'scaling': self.scaling_method
        }

        for n in grid_sizes:
            grid = self._create_grid(n, x_range, y_range)
            tester = (CCDTester1D(grid) if self.get_dimension() == 1 else CCDTester2D(grid))
            tester.setup(
                equation=settings['equation_set'].__class__.__name__.replace('EquationSet', '').replace('1D', '').replace('2D', '').lower() 
                if settings['equation_set'] else "poisson",
                method=settings['method'],
                options=settings['options'],
                scaling=settings['scaling']
            )
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
        A = self.solver.matrix_A
        b = self._build_rhs(test_func)
        
        # スケーリング
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(A, b)
        
        # 解計算
        x = self._solve_system(A_scaled, b_scaled)
        if x is not None and scaling_info is not None and scaler is not None:
            x = scaler.unscale(x, scaling_info)
            
        # 厳密解計算
        exact_x = self._compute_exact(test_func) if x is not None else None
        print(b[-3:],x[-3:],exact_x[-3:])
        
        # 外部ビジュアライザーを使用して可視化
        visualizer = MatrixVisualizer(self.results_dir)
        eq_name = self.equation_set.__class__.__name__.replace("EquationSet", "").replace("1D", "").replace("2D", "")
        return visualizer.visualize(
            A_scaled, b_scaled, x, exact_x, f"{eq_name}_{test_func.name}",
            self.get_dimension(), self.scaling_method
        )
    
    def _init_solver(self):
        """ソルバー初期化"""
        if not self.solver:
            self._create_solver()
            
        if self.solver_method != "direct" or self.solver_options:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)
        
        if self.scaling_method:
            self.solver.scaling_method = self.scaling_method
    
    def _apply_scaling(self, A, b):
        """スケーリング適用"""
        if self.scaling_method:
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
        return A, b, None, None
    
    def _solve_system(self, A, b):
        """システム解法"""
        try:
            if self.solver_method == "direct":
                from cupyx.scipy.sparse.linalg import spsolve
                return spsolve(A, b)
            
            from cupyx.scipy.sparse.linalg import gmres, cg, cgs, lsqr, minres, lsmr
            solvers = {"gmres": gmres, "cg": cg, "cgs": cgs, "lsqr": lsqr, "minres": minres, "lsmr": lsmr}
            solver = solvers.get(self.solver_method)
            
            if solver:
                x, _ = solver(A, b)
                return x
                
            from cupyx.scipy.sparse.linalg import spsolve
            return spsolve(A, b)
        except Exception as e:
            print(f"解法エラー: {e}")
            return None
    
    def _create_grid(self, n, x_range, y_range=None):
        """グリッド作成"""
        if self.get_dimension() == 1:
            return Grid(n, x_range=x_range)
        return Grid(n, n, x_range=x_range, y_range=y_range or x_range)
    
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
        self.solver = CCDSolver1D(self.equation_set, self.grid)
            
    def get_dimension(self):
        return 1
    
    def get_test_function(self, func_name):
        """1Dテスト関数取得"""
        from test_functions import TestFunctionFactory
        
        funcs = TestFunctionFactory.create_standard_1d_functions()
        func = next((f for f in funcs if f.name == func_name), None)
        
        if not func:
            print(f"警告: 1D関数 '{func_name}' が見つかりません。デフォルト関数を使用します。")
            func = funcs[0]
            
        return func
    
    def _build_rhs(self, test_func):
        """1D右辺構築"""
        x_min, x_max = self.grid.x_min, self.grid.x_max
        
        # ソース項
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet1D)
        f_values = cp.array([test_func.f(x) if is_derivative else test_func.d2f(x) for x in self.grid.x])
        
        # 境界値
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }
        
        self._init_solver()
        return self.solver._build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact(self, test_func):
        """1D厳密解計算"""
        n_points = self.grid.n_points
        exact = cp.zeros(n_points * 4)
        
        for i in range(n_points):
            xi = self.grid.get_point(i)
            exact[i*4] = test_func.f(xi)
            exact[i*4+1] = test_func.df(xi)
            exact[i*4+2] = test_func.d2f(xi)
            exact[i*4+3] = test_func.d3f(xi)
            
        return exact
    
    def _process_solution(self, test_func):
        """1D解処理"""
        x = self.grid.get_points()
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet1D)
        
        # 厳密解計算
        exact_psi = cp.array([test_func.f(xi) for xi in x])
        exact_psi_prime = cp.array([test_func.df(xi) for xi in x])
        exact_psi_second = cp.array([test_func.d2f(xi) for xi in x])
        exact_psi_third = cp.array([test_func.d3f(xi) for xi in x])
        
        # 解計算のための入力準備
        x_min, x_max = self.grid.x_min, self.grid.x_max
        f_values = cp.array([test_func.f(xi) if is_derivative else test_func.d2f(xi) for xi in x])
        
        # 境界条件
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }

        # 解計算
        psi, psi_prime, psi_second, psi_third = self.solver.solve(
            analyze_before_solve=False, f_values=f_values, **boundary
        )

        # 誤差計算
        err_psi = float(cp.max(cp.abs(psi - exact_psi)))
        err_psi_prime = float(cp.max(cp.abs(psi_prime - exact_psi_prime)))
        err_psi_second = float(cp.max(cp.abs(psi_second - exact_psi_second)))
        err_psi_third = float(cp.max(cp.abs(psi_third - exact_psi_third)))

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
        self.solver = CCDSolver2D(self.equation_set, self.grid)
            
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
        print(f"警告: 2D関数 '{func_name}' が見つかりません。デフォルト関数を使用します。")
        return funcs[0]
    
    def _build_rhs(self, test_func):
        """2D右辺構築"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        
        # ソース項
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet2D)
        f_values = cp.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x, y = self.grid.get_point(i, j)
                f_values[i, j] = test_func.f(x, y) if is_derivative else (test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y))
        
        # 境界値
        boundary = {
            'left_dirichlet': cp.array([test_func.f(x_min, y) for y in self.grid.y]),
            'right_dirichlet': cp.array([test_func.f(x_max, y) for y in self.grid.y]),
            'bottom_dirichlet': cp.array([test_func.f(x, y_min) for x in self.grid.x]),
            'top_dirichlet': cp.array([test_func.f(x, y_max) for x in self.grid.x]),
            'left_neumann': cp.array([test_func.df_dx(x_min, y) for y in self.grid.y]),
            'right_neumann': cp.array([test_func.df_dx(x_max, y) for y in self.grid.y]),
            'bottom_neumann': cp.array([test_func.df_dy(x, y_min) for x in self.grid.x]),
            'top_neumann': cp.array([test_func.df_dy(x, y_max) for x in self.grid.x])
        }
        
        self._init_solver()
        return self.solver._build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact(self, test_func):
        """2D厳密解計算"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        exact = cp.zeros(nx * ny * 7)
        
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
        """2D解処理"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        X, Y = self.grid.get_points()
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet2D)

        # 厳密解準備
        exact = {
            'psi': cp.zeros((nx, ny)),
            'psi_x': cp.zeros((nx, ny)),
            'psi_y': cp.zeros((nx, ny)),
            'psi_xx': cp.zeros((nx, ny)),
            'psi_yy': cp.zeros((nx, ny)),
            'psi_xxx': cp.zeros((nx, ny)),
            'psi_yyy': cp.zeros((nx, ny))
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
        f_values = cp.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x, y = self.grid.get_point(i, j)
                f_values[i, j] = test_func.f(x, y) if is_derivative else (test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y))

        # 境界条件
        boundary = {
            'left_dirichlet': cp.array([test_func.f(x_min, y) for y in self.grid.y]),
            'right_dirichlet': cp.array([test_func.f(x_max, y) for y in self.grid.y]),
            'bottom_dirichlet': cp.array([test_func.f(x, y_min) for x in self.grid.x]),
            'top_dirichlet': cp.array([test_func.f(x, y_max) for x in self.grid.x]),
            'left_neumann': cp.array([test_func.df_dx(x_min, y) for y in self.grid.y]),
            'right_neumann': cp.array([test_func.df_dx(x_max, y) for y in self.grid.y]),
            'bottom_neumann': cp.array([test_func.df_dy(x, y_min) for x in self.grid.x]),
            'top_neumann': cp.array([test_func.df_dy(x, y_max) for x in self.grid.x])
        }

        # 解計算
        sol = self.solver.solve(analyze_before_solve=False, f_values=f_values, **boundary)
        psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy = sol

        # 誤差計算
        errors = [
            float(cp.max(cp.abs(psi - exact['psi']))),
            float(cp.max(cp.abs(psi_x - exact['psi_x']))),
            float(cp.max(cp.abs(psi_y - exact['psi_y']))),
            float(cp.max(cp.abs(psi_xx - exact['psi_xx']))),
            float(cp.max(cp.abs(psi_yy - exact['psi_yy']))),
            float(cp.max(cp.abs(psi_xxx - exact['psi_xxx']))),
            float(cp.max(cp.abs(psi_yyy - exact['psi_yyy'])))
        ]

        return {
            "function": test_func.name,
            "numerical": [psi, psi_x, psi_y, psi_xx, psi_yy, psi_xxx, psi_yyy],
            "exact": [v for v in exact.values()],
            "errors": errors,
        }