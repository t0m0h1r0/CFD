import cupy as cp
from grid1d import Grid
from grid2d import Grid2D
from solver1d import CCDSolver
from solver2d import CCD2DSolver
from equation_system1d import EquationSystem
from equation_system2d import EquationSystem2D
from equation_sets1d import EquationSet
from equation_sets2d import EquationSet2D
from test_functions1d import TestFunctionFactory
from test_functions2d import TestFunction2DGenerator

class CCDTester:
    """CCDメソッドのテストを行うクラス（1D/2D両対応）"""

    def __init__(self, grid):
        """
        グリッドを指定して初期化
        
        Args:
            grid: Grid (1D) or Grid2D (2D) object
        """
        self.grid = grid
        self.system = None
        self.solver = None
        self.solver_method = "direct"
        self.solver_options = None
        self.scaling_method = None
        self.analyze_matrix = False
        self.equation_set = None
        
        # 1D or 2D mode
        self.is_2d = isinstance(grid, Grid2D)

    def set_solver_options(self, method, options, analyze_matrix=False):
        """
        ソルバーのオプションを設定
        
        Args:
            method: ソルバー方法 ("direct", "gmres", "cg", "cgs")
            options: ソルバーオプション辞書
            analyze_matrix: 解く前に行列を分析するかどうか
        """
        self.solver_method = method
        self.solver_options = options
        self.analyze_matrix = analyze_matrix
        
        # すでにソルバーが存在する場合は設定を更新
        if self.solver is not None:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)
            self.solver.scaling_method = self.scaling_method

    def set_equation_set(self, equation_set_name):
        """
        使用する方程式セットを設定
        
        Args:
            equation_set_name: 方程式セット名
        """
        if self.is_2d:
            self.equation_set = EquationSet2D.create(equation_set_name)
        else:
            self.equation_set = EquationSet.create(equation_set_name)

    def setup_equation_system(self, test_func, use_dirichlet=True, use_neumann=True):
        """
        方程式システムをセットアップ
        
        Args:
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
        """
        if self.is_2d:
            self.system = EquationSystem2D(self.grid)
            
            if self.equation_set is None:
                self.equation_set = EquationSet2D.create("poisson")
        else:
            self.system = EquationSystem(self.grid)
            
            if self.equation_set is None:
                self.equation_set = EquationSet.create("poisson")
        
        # 方程式システム設定
        self.equation_set.setup_equations(
            self.system, 
            self.grid, 
            test_func, 
            use_dirichlet, 
            use_neumann
        )

        # ソルバー作成または更新
        if self.solver is None:
            if self.is_2d:
                self.solver = CCD2DSolver(self.system, self.grid)
            else:
                self.solver = CCDSolver(self.system, self.grid)
        else:
            self.solver.system = self.system

        # ソルバーオプション設定
        if self.solver_method != "direct" or self.solver_options:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)
        
        # スケーリング手法設定
        if hasattr(self, 'scaling_method') and self.scaling_method is not None:
            self.solver.scaling_method = self.scaling_method

    def get_test_function(self, func_name):
        """
        関数名からテスト関数を取得
        
        Args:
            func_name: テスト関数名
            
        Returns:
            テスト関数オブジェクト
        """
        if self.is_2d:
            # まず基本的な2D関数を生成
            standard_funcs = TestFunction2DGenerator.create_standard_functions()
            
            # 指定された名前の関数を検索
            for func in standard_funcs:
                if func.name == func_name:
                    return func
            
            # 見つからない場合は、1D関数から動的に生成を試みる
            funcs_1d = TestFunctionFactory.create_standard_functions()
            for func_1d in funcs_1d:
                if func_1d.name == func_name:
                    # テンソル積拡張を作成
                    return TestFunction2DGenerator.product_extension(func_1d)
            
            # それでも見つからない場合は、最初の関数を返す
            print(f"警告: 関数 '{func_name}' が見つかりませんでした。デフォルト関数を使用します。")
            return standard_funcs[0]
        else:
            # 1Dテスト関数
            test_funcs = TestFunctionFactory.create_standard_functions()
            selected_func = next((f for f in test_funcs if f.name == func_name), test_funcs[0])
            return selected_func

    def run_test_with_options(self, test_func, use_dirichlet=True, use_neumann=True):
        """
        テスト実行
        
        Args:
            test_func: テスト関数または関数名
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
            
        Returns:
            テスト結果の辞書
        """
        # test_funcが文字列の場合、対応するテスト関数を取得
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
        
        self.setup_equation_system(test_func, use_dirichlet, use_neumann)

        if self.analyze_matrix:
            self.solver.analyze_system()

        # ソリューション計算
        if self.is_2d:
            psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy = self.solver.solve(analyze_before_solve=False)
            
            # グリッド点を取得
            X, Y = self.grid.get_points()

            # 厳密解を計算
            exact_psi = cp.zeros_like(psi)
            exact_psi_x = cp.zeros_like(psi_x)
            exact_psi_y = cp.zeros_like(psi_y)
            exact_psi_xx = cp.zeros_like(psi_xx)
            exact_psi_yy = cp.zeros_like(psi_yy)
            exact_psi_xxx = cp.zeros_like(psi_xxx)
            exact_psi_yyy = cp.zeros_like(psi_yyy)

            # 各グリッド点で厳密値を計算
            for i in range(self.grid.nx_points):
                for j in range(self.grid.ny_points):
                    x, y = self.grid.get_point(i, j)
                    exact_psi[i, j] = test_func.f(x, y)
                    exact_psi_x[i, j] = test_func.df_dx(x, y)
                    exact_psi_y[i, j] = test_func.df_dy(x, y)
                    exact_psi_xx[i, j] = test_func.d2f_dx2(x, y)
                    exact_psi_yy[i, j] = test_func.d2f_dy2(x, y)
                    exact_psi_xxx[i, j] = test_func.d3f_dx3(x, y)
                    exact_psi_yyy[i, j] = test_func.d3f_dy3(x, y)

            # 誤差を計算
            err_psi = float(cp.max(cp.abs(psi - exact_psi)))
            err_psi_x = float(cp.max(cp.abs(psi_x - exact_psi_x)))
            err_psi_y = float(cp.max(cp.abs(psi_y - exact_psi_y)))
            err_psi_xx = float(cp.max(cp.abs(psi_xx - exact_psi_xx)))
            err_psi_yy = float(cp.max(cp.abs(psi_yy - exact_psi_yy)))
            err_psi_xxx = float(cp.max(cp.abs(psi_xxx - exact_psi_xxx)))
            err_psi_yyy = float(cp.max(cp.abs(psi_yyy - exact_psi_yyy)))

            return {
                "function": test_func.name,
                "numerical": [psi, psi_x, psi_y, psi_xx, psi_yy, psi_xxx, psi_yyy],
                "exact": [exact_psi, exact_psi_x, exact_psi_y, exact_psi_xx, exact_psi_yy, exact_psi_xxx, exact_psi_yyy],
                "errors": [err_psi, err_psi_x, err_psi_y, err_psi_xx, err_psi_yy, err_psi_xxx, err_psi_yyy],
            }
        else:
            # 1D solver
            psi, psi_prime, psi_second, psi_third = self.solver.solve(analyze_before_solve=False)

            x = self.grid.get_points()
            exact_psi = cp.array([test_func.f(xi) for xi in x])
            exact_psi_prime = cp.array([test_func.df(xi) for xi in x])
            exact_psi_second = cp.array([test_func.d2f(xi) for xi in x])
            exact_psi_third = cp.array([test_func.d3f(xi) for xi in x])

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

    def run_grid_convergence_test(self, test_func, grid_sizes, x_range, y_range=None, use_dirichlet=True, use_neumann=True):
        """
        格子収束性テスト
        
        Args:
            test_func: テスト関数または関数名
            grid_sizes: テストするグリッドサイズのリスト
            x_range: x方向の範囲
            y_range: y方向の範囲（2Dのみ）
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
            
        Returns:
            各グリッドサイズの結果を持つ辞書
        """
        results = {}
        original_method = self.solver_method
        original_options = self.solver_options
        original_analyze = self.analyze_matrix
        original_equation_set = self.equation_set
        original_scaling_method = self.scaling_method if hasattr(self, 'scaling_method') else None

        # test_funcが文字列の場合、対応するテスト関数を取得
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)

        for n in grid_sizes:
            if self.is_2d:
                if y_range is None:
                    y_range = x_range  # デフォルトでx_rangeと同じ

                # 両方向で同じ点数を使用
                grid = Grid2D(n, n, x_range, y_range)
            else:
                grid = Grid(n, x_range)
                
            tester = CCDTester(grid)
            tester.set_solver_options(original_method, original_options, original_analyze)
            tester.equation_set = original_equation_set
            tester.scaling_method = original_scaling_method
            
            result = tester.run_test_with_options(test_func, use_dirichlet, use_neumann)
            results[n] = result["errors"]

        return results