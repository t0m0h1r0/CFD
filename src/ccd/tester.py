# tester.py
import cupy as cp
from typing import Dict, List, Optional, Tuple, Any
from grid import Grid
from solver import CCDSolver
from equation_system import EquationSystem
from equation_sets import EquationSet
from test_functions import TestFunction


class CCDTester:
    """CCDメソッドのテストを行うクラス"""

    def __init__(self, grid: Grid):
        """
        初期化

        Args:
            grid: 計算格子
        """
        self.grid = grid
        self.system = None
        self.solver = None
        self.solver_method = "direct"
        self.solver_options = None
        self.analyze_matrix = False
        self.equation_set = None

    def set_solver_options(self, method: str, options: Dict[str, Any], analyze_matrix: bool = False):
        """
        ソルバーの種類とオプションを設定

        Args:
            method: ソルバーの種類 ('direct', 'gmres', 'cg', 'cgs')
            options: ソルバーのオプション
            analyze_matrix: 行列の疎性を分析するかどうか
        """
        self.solver_method = method
        self.solver_options = options
        self.analyze_matrix = analyze_matrix

    def set_equation_set(self, equation_set_name: str):
        """
        使用する方程式セットを設定

        Args:
            equation_set_name: 方程式セットの名前 ('poisson', 'higher_derivative', 'custom')
        """
        self.equation_set = EquationSet.create(equation_set_name)
        print(f"方程式セットを '{equation_set_name}' に設定しました")

    def setup_equation_system(
        self, 
        test_func: TestFunction, 
        use_dirichlet: bool = True,
        use_neumann: bool = True
    ):
        """
        方程式システムとソルバーを設定する（または再設定する）

        Args:
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
        """
        # システムを構築
        self.system = EquationSystem(self.grid)

        # 方程式セットが設定されていない場合はデフォルトのポアソン方程式を使用
        if self.equation_set is None:
            self.equation_set = EquationSet.create("poisson")

        # 方程式セットを使って方程式を設定
        self.equation_set.setup_equations(
            self.system, 
            self.grid, 
            test_func, 
            use_dirichlet, 
            use_neumann
        )

        # ソルバーを作成または更新
        if self.solver is None:
            self.solver = CCDSolver(self.system, self.grid)
        else:
            self.solver.system = self.system  # システム参照を更新

        # ソルバーの設定
        if self.solver_method != "direct" or self.solver_options:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)

    def run_test_with_options(
        self,
        test_func: TestFunction,
        use_dirichlet: bool = True,
        use_neumann: bool = True,
    ) -> Dict:
        """
        より柔軟なオプションでテストを実行

        Args:
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか（デフォルトはTrue）
            use_neumann: ノイマン境界条件を使用するかどうか（デフォルトはTrue）

        Returns:
            テスト結果の辞書
        """
        # 方程式システムとソルバーを設定
        self.setup_equation_system(test_func, use_dirichlet, use_neumann)

        # 行列を分析（オプション）
        if self.analyze_matrix:
            self.solver.analyze_system()

        # 解く
        psi, psi_prime, psi_second, psi_third = self.solver.solve(analyze_before_solve=False)

        # 解析解を計算
        x = self.grid.get_points()
        exact_psi = cp.array([test_func.f(xi) for xi in x])
        exact_psi_prime = cp.array([test_func.df(xi) for xi in x])
        exact_psi_second = cp.array([test_func.d2f(xi) for xi in x])
        exact_psi_third = cp.array([test_func.d3f(xi) for xi in x])

        # 誤差を計算（CuPy配列のまま）
        err_psi = float(cp.max(cp.abs(psi - exact_psi)))
        err_psi_prime = float(cp.max(cp.abs(psi_prime - exact_psi_prime)))
        err_psi_second = float(cp.max(cp.abs(psi_second - exact_psi_second)))
        err_psi_third = float(cp.max(cp.abs(psi_third - exact_psi_third)))

        # 結果を返す
        return {
            "function": test_func.name,
            "numerical": [psi, psi_prime, psi_second, psi_third],
            "exact": [exact_psi, exact_psi_prime, exact_psi_second, exact_psi_third],
            "errors": [err_psi, err_psi_prime, err_psi_second, err_psi_third],
        }

    def run_grid_convergence_test(
        self,
        test_func: TestFunction,
        grid_sizes: List[int],
        x_range: Tuple[float, float],
        use_dirichlet: bool = True,
        use_neumann: bool = True,
    ) -> Dict[int, List[float]]:
        """
        グリッドサイズによる収束性テストを実行

        Args:
            test_func: テスト関数
            grid_sizes: グリッドサイズのリスト
            x_range: 計算範囲
            use_dirichlet: ディリクレ境界条件を使用するかどうか（デフォルトはTrue）
            use_neumann: ノイマン境界条件を使用するかどうか（デフォルトはTrue）

        Returns:
            グリッドサイズごとの誤差 {grid_size: [err_psi, err_psi', err_psi'', err_psi''']}
        """
        results = {}

        # 現在のソルバー設定とEquationSetを保存
        original_method = self.solver_method
        original_options = self.solver_options
        original_analyze = self.analyze_matrix
        original_equation_set = self.equation_set

        for n in grid_sizes:
            # グリッドを作成
            grid = Grid(n, x_range)

            # このクラスのインスタンスを新しいグリッドで作成
            tester = CCDTester(grid)
            
            # ソルバー設定とEquationSetを引き継ぐ
            tester.set_solver_options(original_method, original_options, original_analyze)
            tester.equation_set = original_equation_set

            # テストを実行
            result = tester.run_test_with_options(
                test_func,
                use_dirichlet=use_dirichlet,
                use_neumann=use_neumann,
            )

            # 結果を保存
            results[n] = result["errors"]

        return results