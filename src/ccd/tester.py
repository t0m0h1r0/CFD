from abc import ABC, abstractmethod
import cupy as cp
from grid import Grid
from solver import create_ccd_solver
from equation_system import EquationSystem
from equation_sets import EquationSet

class BaseCCDTester(ABC):
    """CCDメソッドのテストを行う抽象基底クラス"""

    def __init__(self, grid):
        """
        グリッドを指定して初期化
        
        Args:
            grid: Gridオブジェクト (1Dまたは2D)
        """
        self.grid = grid
        self.system = None
        self.solver = None
        self.solver_method = "direct"
        self.solver_options = None
        self.scaling_method = None
        self.analyze_matrix = False
        self.equation_set = None

    def set_solver_options(self, method, options, analyze_matrix=False):
        """
        ソルバーのオプションを設定
        
        Args:
            method: ソルバー方法 ("direct", "gmres", "cg", "cgs")
            options: ソルバーオプション辞書
            analyze_matrix: 解く前に行列を分析するかどうか
        """
        self.solver_method = method
        self.solver_options = options if options else {}
        self.analyze_matrix = analyze_matrix
        
        # すでにソルバーが存在する場合は設定を更新
        if self.solver is not None:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)
            if self.scaling_method is not None:
                self.solver.scaling_method = self.scaling_method

    def set_equation_set(self, equation_set_name):
        """
        使用する方程式セットを設定
        
        Args:
            equation_set_name: 方程式セット名
        """
        # 次元に基づいて適切なセットを作成
        dimension = self.get_dimension()
        self.equation_set = EquationSet.create(equation_set_name, dimension=dimension)

    def setup_equation_system(self, test_func, use_dirichlet=True, use_neumann=True):
        """
        方程式システムをセットアップ
        
        Args:
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
        """
        # 方程式システムの作成
        self.system = EquationSystem(self.grid)
        
        # 方程式セットが未設定の場合はデフォルトを使用
        if self.equation_set is None:
            dimension = self.get_dimension()
            self.equation_set = EquationSet.create("poisson", dimension=dimension)
        
        # 方程式システム設定
        self.equation_set.setup_equations(
            self.system, 
            self.grid, 
            test_func, 
            use_dirichlet, 
            use_neumann
        )

        # 適切なソルバーを作成または更新
        if self.solver is None:
            self.solver = create_ccd_solver(self.system, self.grid)
        else:
            self.solver.system = self.system

        # ソルバーオプション設定
        if self.solver_method != "direct" or self.solver_options:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)
        
        # スケーリング手法設定
        if self.scaling_method is not None:
            self.solver.scaling_method = self.scaling_method
            
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

        # ソリューション計算と結果処理
        return self._process_solution(test_func)

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
        # 結果を保存する辞書
        results = {}
        
        # 現在の設定をバックアップ
        original_method = self.solver_method
        original_options = self.solver_options
        original_analyze = self.analyze_matrix
        original_equation_set = self.equation_set
        original_scaling_method = self.scaling_method

        # test_funcが文字列の場合、対応するテスト関数を取得
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)

        # ファクトリメソッドを使って次元ごとの適切なテスタークラスを取得
        tester_factory = self.create_dimension_specific_tester

        # 各グリッドサイズでテスト実行
        for n in grid_sizes:
            # 次元に基づいてグリッドを作成
            grid = self._create_grid_for_convergence_test(n, x_range, y_range)
                
            # 新しいテスターを作成し設定を引き継ぐ
            tester = tester_factory(grid)
            tester.set_solver_options(original_method, original_options, original_analyze)
            tester.equation_set = original_equation_set
            tester.scaling_method = original_scaling_method
            
            # テスト実行と結果保存
            result = tester.run_test_with_options(test_func, use_dirichlet, use_neumann)
            results[n] = result["errors"]

        return results
        
    @abstractmethod
    def get_dimension(self):
        """次元を返す（1または2）"""
        pass
        
    @abstractmethod
    def get_test_function(self, func_name):
        """
        関数名からテスト関数を取得
        
        Args:
            func_name: テスト関数名
            
        Returns:
            テスト関数オブジェクト
        """
        pass
        
    @abstractmethod
    def _process_solution(self, test_func):
        """
        数値解と誤差を計算して結果を返す
        
        Args:
            test_func: テスト関数
            
        Returns:
            テスト結果の辞書
        """
        pass
        
    @abstractmethod
    def _create_grid_for_convergence_test(self, n, x_range, y_range=None):
        """
        収束性テスト用のグリッドを作成
        
        Args:
            n: グリッドサイズ
            x_range: x方向の範囲
            y_range: y方向の範囲（2Dのみ）
            
        Returns:
            Grid オブジェクト
        """
        pass
        
    @abstractmethod
    def create_dimension_specific_tester(self, grid):
        """
        次元ごとの適切なテスターを作成
        
        Args:
            grid: Grid オブジェクト
            
        Returns:
            BaseCCDTester のサブクラスインスタンス
        """
        pass


class CCDTester1D(BaseCCDTester):
    """1D CCDメソッドのテストを行うクラス"""
    
    def __init__(self, grid):
        """
        グリッドを指定して初期化
        
        Args:
            grid: 1D Grid オブジェクト
        """
        super().__init__(grid)
        if grid.is_2d:
            raise ValueError("1Dテスターは2Dグリッドでは使用できません")
            
    def get_dimension(self):
        """次元を返す"""
        return 1
    
    def get_test_function(self, func_name):
        """1Dテスト関数を取得"""
        from test_functions1d import TestFunctionFactory
        
        test_funcs = TestFunctionFactory.create_standard_functions()
        selected_func = next((f for f in test_funcs if f.name == func_name), None)
        
        if selected_func is None:
            print(f"警告: 1D関数 '{func_name}' が見つかりませんでした。デフォルト関数を使用します。")
            selected_func = test_funcs[0]
            
        return selected_func
        
    def _process_solution(self, test_func):
        """1D解の処理"""
        # ソルバーで解を計算
        psi, psi_prime, psi_second, psi_third = self.solver.solve(analyze_before_solve=False)

        # グリッド点での厳密解を計算
        x = self.grid.get_points()
        exact_psi = cp.array([test_func.f(xi) for xi in x])
        exact_psi_prime = cp.array([test_func.df(xi) for xi in x])
        exact_psi_second = cp.array([test_func.d2f(xi) for xi in x])
        exact_psi_third = cp.array([test_func.d3f(xi) for xi in x])

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
        
    def _create_grid_for_convergence_test(self, n, x_range, y_range=None):
        """収束性テスト用の1Dグリッドを作成"""
        return Grid(n, x_range=x_range)
        
    def create_dimension_specific_tester(self, grid):
        """1Dテスターを作成"""
        return CCDTester1D(grid)


class CCDTester2D(BaseCCDTester):
    """2D CCDメソッドのテストを行うクラス"""
    
    def __init__(self, grid):
        """
        グリッドを指定して初期化
        
        Args:
            grid: 2D Grid オブジェクト
        """
        super().__init__(grid)
        if not grid.is_2d:
            raise ValueError("2Dテスターは1Dグリッドでは使用できません")
            
    def get_dimension(self):
        """次元を返す"""
        return 2
    
    def get_test_function(self, func_name):
        """2Dテスト関数を取得"""
        from test_functions2d import TestFunction2DGenerator
        from test_functions1d import TestFunctionFactory
        
        # まず基本的な2D関数を生成
        standard_funcs = TestFunction2DGenerator.create_standard_functions()
        
        # 指定された名前の関数を検索
        selected_func = next((f for f in standard_funcs if f.name == func_name), None)
        if selected_func is not None:
            return selected_func
        
        # 見つからない場合は、1D関数から動的に生成を試みる
        funcs_1d = TestFunctionFactory.create_standard_functions()
        func_1d = next((f for f in funcs_1d if f.name == func_name), None)
        
        if func_1d is not None:
            # テンソル積拡張を作成
            return TestFunction2DGenerator.product_extension(func_1d)
        
        # それでも見つからない場合は、最初の関数を返す
        print(f"警告: 2D関数 '{func_name}' が見つかりませんでした。デフォルト関数を使用します。")
        return standard_funcs[0]
        
    def _process_solution(self, test_func):
        """2D解の処理"""
        # ソルバーで解を計算
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
        
    def _create_grid_for_convergence_test(self, n, x_range, y_range=None):
        """収束性テスト用の2Dグリッドを作成"""
        if y_range is None:
            y_range = x_range  # デフォルトでx_rangeと同じ
        return Grid(n, n, x_range=x_range, y_range=y_range)
        
    def create_dimension_specific_tester(self, grid):
        """2Dテスターを作成"""
        return CCDTester2D(grid)


def create_ccd_tester(grid):
    """
    グリッド次元に応じて適切なテスターを作成
    
    Args:
        grid: Grid オブジェクト (1Dまたは2D)
        
    Returns:
        BaseCCDTesterのサブクラスインスタンス
    """
    if grid.is_2d:
        return CCDTester2D(grid)
    else:
        return CCDTester1D(grid)

class CCDTester:
    """CCDメソッドのテストを行う後方互換性ラッパークラス"""

    def __init__(self, grid):
        """
        グリッドを指定して初期化
        
        Args:
            grid: Gridオブジェクト (1Dまたは2D)
        """
        self.is_2d = grid.is_2d
        self._tester = create_ccd_tester(grid)

    def __getattr__(self, name):
        """内部テスターに属性アクセスを委譲"""
        return getattr(self._tester, name)

    # 以下、主要なメソッドを明示的に委譲
    
    def set_solver_options(self, method, options, analyze_matrix=False):
        """ソルバーのオプションを設定"""
        return self._tester.set_solver_options(method, options, analyze_matrix)

    def set_equation_set(self, equation_set_name):
        """使用する方程式セットを設定"""
        return self._tester.set_equation_set(equation_set_name)
        
    def setup_equation_system(self, test_func, use_dirichlet=True, use_neumann=True):
        """方程式システムをセットアップ"""
        return self._tester.setup_equation_system(test_func, use_dirichlet, use_neumann)

    def get_test_function(self, func_name):
        """関数名からテスト関数を取得"""
        return self._tester.get_test_function(func_name)
        
    def run_test_with_options(self, test_func, use_dirichlet=True, use_neumann=True):
        """テスト実行"""
        return self._tester.run_test_with_options(test_func, use_dirichlet, use_neumann)
        
    def run_grid_convergence_test(self, test_func, grid_sizes, x_range, y_range=None, use_dirichlet=True, use_neumann=True):
        """格子収束性テスト"""
        return self._tester.run_grid_convergence_test(test_func, grid_sizes, x_range, y_range, use_dirichlet, use_neumann)