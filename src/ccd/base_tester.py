"""
高精度コンパクト差分法 (CCD) テスターの基底クラス

このモジュールは、CCDソルバーのテストと評価を行うための
基底クラスと共通機能を提供します。
"""

from abc import ABC, abstractmethod
import os
import numpy as np

from grid1d import Grid1D
from grid2d import Grid2D
from equation_sets import EquationSet
from matrix_visualizer import MatrixVisualizer


class CCDTester(ABC):
    """CCDテスト基底クラス"""

    def __init__(self, grid):
        """
        テスターを初期化
        
        Args:
            grid: 計算グリッド
        """
        self.grid = grid
        self.solver = None
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.equation_set = None
        self.backend = "cpu"  # デフォルトバックエンド
        self.results_dir = "results"
        self.matrix_basename = None  # 行列可視化用のベース名
        self.perturbation_level = None  # 厳密解に加える摂動レベル (None=厳密解を使用しない、0=摂動なし)
        self.exact_solution = None  # 厳密解を保存
        
        # 結果ディレクトリの作成
        os.makedirs(self.results_dir, exist_ok=True)

    def setup(self, equation="poisson", method="direct", options=None, scaling=None, backend="cpu"):
        """
        テスター設定のショートカット
        
        Args:
            equation: 方程式セット名
            method: ソルバーメソッド名
            options: ソルバーオプション辞書
            scaling: スケーリング手法名
            backend: 計算バックエンド名
        
        Returns:
            self: メソッドチェーン用
        """
        self.equation_set = EquationSet.create(equation, dimension=self.get_dimension())
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling
        self.backend = backend
        return self
        
    def set_solver_options(self, method, options=None, analyze_matrix=False):
        """
        ソルバーオプションを設定
        
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
        
        # ソルバーが既に作成されている場合は、オプションを設定
        if hasattr(self, 'solver') and self.solver:
            self.solver.set_solver(
                method=self.solver_method, 
                options=self.solver_options, 
                scaling_method=self.scaling_method
            )
        
        return self
        
    def set_equation_set(self, equation_set_name):
        """
        方程式セットを設定
        
        Args:
            equation_set_name: 方程式セットの名前
            
        Returns:
            self: メソッドチェーン用
        """
        self.equation_set = EquationSet.create(equation_set_name, dimension=self.get_dimension())
        return self
        
    def _add_perturbation(self, exact_sol, level):
        """
        厳密解に指定レベルの摂動を加える
        
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
        """
        テスト実行
        
        Args:
            test_func: テスト関数（文字列指定も可能）
        
        Returns:
            結果の辞書
        """
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
        """
        格子収束性テスト
        
        Args:
            test_func: テスト関数（文字列指定も可能）
            grid_sizes: テストする格子サイズのリスト
            x_range: x方向の範囲
            y_range: y方向の範囲（1Dの場合はNone）
            
        Returns:
            各格子サイズごとの結果辞書
        """
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
            tester_class = self.__class__  # 現在のテスタークラスを使用
            tester = tester_class(grid)
            
            # 方程式セット名を取得
            equation_name = "poisson"
            if settings['equation_set']:
                eq_class_name = settings['equation_set'].__class__.__name__
                equation_name = eq_class_name.replace('EquationSet', '').replace('1D', '').replace('2D', '').lower()
                
            tester.setup(
                equation=equation_name,
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
        """
        行列システム可視化 (MatrixVisualizer を使用)
        
        Args:
            test_func: テスト関数（Noneの場合はデフォルト関数を使用）
            
        Returns:
            出力ファイルパス
        """
        if test_func is None:
            # デフォルトテスト関数
            from test_function_factory import TestFunctionFactory
            if self.get_dimension() == 1:
                test_func = TestFunctionFactory.create_standard_1d_functions()[3]  # Sine
            else:
                test_func = TestFunctionFactory.create_standard_2d_functions()[0]  # Sine2D
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
        
        # 方程式セット名を取得
        eq_name = "Unknown"
        if self.equation_set:
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
        """
        可視化用に方程式系を解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            解ベクトル
        """
        try:
            # LinearSolverを使用
            return self.solver.linear_solver.solve(b, method=self.solver_method, options=self.solver_options)
        except Exception as e:
            print(f"Solver error: {e}")
            # フォールバック: 直接SciPyを使用
            try:
                from scipy.sparse.linalg import spsolve
                return spsolve(A, b)
            except Exception as fallback_error:  # 具体的な例外に変更
                print(f"SciPy fallback solver also failed: {fallback_error}")
                return None
    
    def _init_solver(self):
        """ソルバー初期化"""
        if not self.solver:
            self._create_solver()
            
        if self.solver_method != "direct" or self.solver_options or self.scaling_method:
            self.solver.set_solver(
                method=self.solver_method, 
                options=self.solver_options, 
                scaling_method=self.scaling_method
            )
    
    def _create_grid(self, n, x_range, y_range=None):
        """
        グリッド作成
        
        Args:
            n: 格子点数
            x_range: x方向の範囲
            y_range: y方向の範囲（1Dの場合はNone）
            
        Returns:
            生成したグリッド
        """
        if self.get_dimension() == 1:
            return Grid1D(n, x_range=x_range)
        return Grid2D(n, n, x_range=x_range, y_range=y_range or x_range)
    
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