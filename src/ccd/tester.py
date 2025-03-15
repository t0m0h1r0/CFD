from abc import ABC, abstractmethod
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from grid import Grid
from solver import CCDSolver1D, CCDSolver2D
from equation_sets import EquationSet, DerivativeEquationSet1D, DerivativeEquationSet2D
from scaling import plugin_manager


class TestStrategy(ABC):
    """テスト戦略のインターフェース (Strategy Pattern)"""
    
    @abstractmethod
    def execute(self, tester, test_func):
        """テスト戦略を実行"""
        pass


class SolutionProcessor(ABC):
    """ソリューション処理のインターフェース (SRP)"""
    
    @abstractmethod
    def process_solution(self, solver, test_func):
        """ソリューションを処理して結果を返す"""
        pass


class MatrixSystemVisualizer:
    """行列システム可視化の責任を持つクラス (SRP)"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize(self, A, b, x, exact_x, title, equation_name, dimension, scaling_method=None):
        """行列システムを可視化して結果のファイルパスを返す"""
        # 出力パス生成
        scale_suffix = f"_{scaling_method}" if scaling_method else ""
        output_path = os.path.join(
            self.output_dir, 
            f"{equation_name}_{dimension}D_{title}{scale_suffix}_matrix.png"
        )
        
        # 可視化実行
        self._visualize_system(A, b, x, exact_x, title, output_path, dimension)
        
        return output_path
    
    def _visualize_system(self, A, b, x, exact_x, title, output_path, dimension):
        """システム Ax = b を単一グリッドに統合して可視化"""
        # CuPy/SciPy配列をNumPy配列に変換
        def to_numpy(arr):
            if arr is None:
                return None
            if hasattr(arr, 'toarray'):
                return arr.toarray().get() if hasattr(arr, 'get') else arr.toarray()
            elif hasattr(arr, 'get'):
                return arr.get()
            return arr

        # 行列とベクトルを変換
        A_np = to_numpy(A)
        b_np = to_numpy(b).reshape(-1, 1) if b is not None else None
        x_np = to_numpy(x).reshape(-1, 1) if x is not None else None
        exact_x_np = to_numpy(exact_x).reshape(-1, 1) if exact_x is not None else None
        
        # 誤差計算
        error_np = np.abs(x_np - exact_x_np) if x_np is not None and exact_x_np is not None else None
        
        # 合成データ作成 (A|x|exact|error|b)
        composite_data = np.abs(A_np)
        offset = A_np.shape[1]
        col_positions = {"A": offset // 2}  # A行列の中央
        
        # ベクトルを結合
        if x_np is not None:
            composite_data = np.hstack((composite_data, np.abs(x_np)))
            col_positions["x"] = offset
            offset += 1
            
        if exact_x_np is not None:
            composite_data = np.hstack((composite_data, np.abs(exact_x_np)))
            col_positions["exact"] = offset
            offset += 1
            
        if error_np is not None:
            composite_data = np.hstack((composite_data, error_np))
            col_positions["error"] = offset
            offset += 1
            
        if b_np is not None:
            composite_data = np.hstack((composite_data, np.abs(b_np)))
            col_positions["b"] = offset
        
        # 可視化
        plt.figure(figsize=(12, 8))
        
        # 対数スケールで表示
        non_zero = composite_data[composite_data > 0]
        if len(non_zero) > 0:
            plt.imshow(
                composite_data, 
                norm=LogNorm(vmin=non_zero.min(), vmax=composite_data.max()),
                cmap='viridis', 
                aspect='auto'
            )
            plt.colorbar(label='Absolute Value (Log Scale)')
        
        # A行列とベクトル間に区切り線を追加
        plt.axvline(x=A_np.shape[1]-0.5, color='r', linestyle='-')
        
        # タイトル
        full_title = f"{dimension}D {title}"
        if scaling_method := output_path.split('_')[-2] if '_' in output_path else None:
            if scaling_method not in ['matrix', 'verify']:
                full_title += f" (Scaling: {scaling_method})"
            
        plt.title(full_title)
        plt.xlabel("Component")
        plt.ylabel("Row/Index")
        
        # 列ラベル追加
        for label, pos in col_positions.items():
            plt.text(pos, -5, label, ha='center')
        
        # 統計情報
        info_texts = []
        
        # 行列情報
        if A_np is not None:
            n_elements = A_np.size
            n_nonzero = np.count_nonzero(A_np)
            sparsity = 1.0 - (n_nonzero / n_elements)
            info_texts.append(f"Matrix: {A_np.shape[0]}×{A_np.shape[1]}, Sparsity: {sparsity:.4f}")
        
        # 誤差情報
        if error_np is not None:
            max_error = np.max(error_np)
            avg_error = np.mean(error_np)
            info_texts.append(f"Error: Max={max_error:.4e}, Avg={avg_error:.4e}")
            
            # 成分ごとの誤差
            if len(x_np) % 4 == 0 and len(x_np) > 4:  # 1Dの場合
                component_names = ["ψ", "ψ'", "ψ''", "ψ'''"]
                component_errors = []
                
                for i, name in enumerate(component_names):
                    indices = range(i, len(x_np), 4)
                    comp_error = np.max(np.abs(x_np[indices] - exact_x_np[indices]))
                    component_errors.append(f"{name}: {comp_error:.4e}")
                
                info_texts.append("Component Errors: " + ", ".join(component_errors))
            elif len(x_np) % 7 == 0 and len(x_np) > 7:  # 2Dの場合
                component_names = ["ψ", "ψ_x", "ψ_xx", "ψ_xxx", "ψ_y", "ψ_yy", "ψ_yyy"]
                component_errors = []
                
                for i, name in enumerate(component_names):
                    indices = range(i, len(x_np), 7)
                    comp_error = np.max(np.abs(x_np[indices] - exact_x_np[indices]))
                    component_errors.append(f"{name}: {comp_error:.4e}")
                
                info_texts.append("Component Errors: " + ", ".join(component_errors[:3]) + "...")
        
        # 統計情報テキスト表示
        for i, text in enumerate(info_texts):
            plt.figtext(0.5, 0.01 - i*0.03, text, ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1 + len(info_texts)*0.03)
        
        # 保存
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class CCDTester(ABC):
    """CCDメソッドのテストを行う抽象基底クラス (Template Method Pattern)"""

    def __init__(self, grid):
        self.grid = grid
        self.solver = None
        self.solver_method = "direct"
        self.solver_options = None
        self.scaling_method = None
        self.analyze_matrix = False
        self.equation_set = None
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 依存オブジェクト
        self._solution_processor = self._create_solution_processor()
        self._matrix_visualizer = MatrixSystemVisualizer(self.results_dir)

    def set_solver_options(self, method, options=None, analyze_matrix=False):
        """ソルバーのオプションを設定"""
        self.solver_method = method
        self.solver_options = options or {}
        self.analyze_matrix = analyze_matrix
        
        # すでにソルバーが存在する場合は設定を更新
        if self.solver is not None:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)
            if self.scaling_method is not None:
                self.solver.scaling_method = self.scaling_method
                
        return self  # メソッドチェーン用

    def set_equation_set(self, equation_set_name):
        """使用する方程式セットを設定"""
        self.equation_set = EquationSet.create(equation_set_name, dimension=self.get_dimension())
        return self  # メソッドチェーン用
    
    def set_output_directory(self, directory):
        """出力ディレクトリを設定"""
        self.results_dir = directory
        os.makedirs(directory, exist_ok=True)
        self._matrix_visualizer = MatrixSystemVisualizer(directory)
        return self  # メソッドチェーン用

    def run_test(self, test_func):
        """単一テスト関数のテストを実行"""
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
        
        # 方程式セットが未設定の場合はデフォルトを使用
        if self.equation_set is None:
            self.set_equation_set("poisson")

        # ソルバーを初期化
        self._ensure_solver_initialized()

        # 行列解析（要求された場合）
        if self.analyze_matrix:
            self.solver.analyze_system()

        # テスト戦略を実行
        return SingleFunctionTestStrategy().execute(self, test_func)

    def run_test_with_options(self, test_func):
        """下位互換性のために run_test のエイリアスを提供"""
        return self.run_test(test_func)
        
    def visualize_matrix_system(self, test_func=None):
        """行列システムの可視化"""
        # テスト関数が指定されていない場合はデフォルトを使用
        if test_func is None:
            dimension = self.get_dimension()
            from test_functions import TestFunctionFactory
            test_func = (TestFunctionFactory.create_standard_1d_functions()[3] if dimension == 1 
                    else TestFunctionFactory.create_standard_2d_functions()[0])
        elif isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
    
    def run_convergence_test(self, test_func, grid_sizes, x_range, y_range=None):
        """格子収束性テストを実行"""
        if isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
            
        # 方程式セットが未設定の場合はデフォルトを使用
        if self.equation_set is None:
            self.set_equation_set("poisson")
            
        return GridConvergenceTestStrategy(
            grid_sizes, x_range, y_range
        ).execute(self, test_func)
    
    def visualize_matrix_system(self, test_func=None):
        """行列システムの可視化"""
        # テスト関数が指定されていない場合はデフォルトを使用
        if test_func is None:
            dimension = self.get_dimension()
            from test_functions import TestFunctionFactory
            test_func = (TestFunctionFactory.create_standard_1d_functions()[3] if dimension == 1 
                       else TestFunctionFactory.create_standard_2d_functions()[0])
        elif isinstance(test_func, str):
            test_func = self.get_test_function(test_func)
        
        # ソルバーを初期化
        self._ensure_solver_initialized()
        
        # 行列システム構築
        A = self.solver.matrix_A
        b = self._build_rhs_vector(test_func)
        
        # スケーリング適用
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(A, b)
        
        # 解ベクトル計算
        x = self._solve_system(A_scaled, b_scaled)
        if x is not None and scaling_info is not None and scaler is not None:
            x = scaler.unscale(x, scaling_info)
            
        # 厳密解計算
        exact_x = self._compute_exact_solution(test_func) if x is not None else None
        
        # 可視化
        equation_name = self.equation_set.__class__.__name__.replace("EquationSet", "").replace("1D", "").replace("2D", "")
        return self._matrix_visualizer.visualize(
            A_scaled, b_scaled, x, exact_x, test_func.name, 
            equation_name, self.get_dimension(), self.scaling_method
        )
    
    def _ensure_solver_initialized(self):
        """ソルバーが初期化されていることを確認"""
        if self.solver is None:
            self._create_solver()
            
            # ソルバーオプション設定
            if self.solver_method != "direct" or self.solver_options:
                self.solver.set_solver(method=self.solver_method, options=self.solver_options)
            
            # スケーリング手法設定
            if self.scaling_method is not None:
                self.solver.scaling_method = self.scaling_method
    
    def _apply_scaling(self, A, b):
        """スケーリングを適用"""
        if self.scaling_method:
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
                
        return A, b, None, None
    
    def _solve_system(self, A, b):
        """線形システムを解く"""
        try:
            if self.solver_method == "direct":
                from cupyx.scipy.sparse.linalg import spsolve
                return spsolve(A, b)
            else:
                from cupyx.scipy.sparse.linalg import gmres, cg, cgs, lsqr, minres, lsmr
                solver_funcs = {
                    "gmres": gmres, "cg": cg, "cgs": cgs, 
                    "lsqr": lsqr, "minres": minres, "lsmr": lsmr
                }
                if self.solver_method in solver_funcs:
                    x, _ = solver_funcs[self.solver_method](A, b)
                    return x
                else:
                    from cupyx.scipy.sparse.linalg import spsolve
                    return spsolve(A, b)
        except Exception as e:
            print(f"警告: ソルバーでの解法に失敗しました: {e}")
            return None
    
    @abstractmethod
    def get_dimension(self):
        """次元を返す（1または2）"""
        pass
        
    @abstractmethod
    def get_test_function(self, func_name):
        """関数名からテスト関数を取得"""
        pass
    
    @abstractmethod
    def _create_solution_processor(self):
        """ソリューション処理オブジェクトを作成"""
        pass
    
    @abstractmethod
    def _create_solver(self):
        """次元に応じた適切なソルバーを作成"""
        pass
    
    @abstractmethod
    def _build_rhs_vector(self, test_func):
        """右辺ベクトルを構築"""
        pass
    
    @abstractmethod
    def _compute_exact_solution(self, test_func):
        """厳密解を計算"""
        pass


class SingleFunctionTestStrategy(TestStrategy):
    """単一関数テスト戦略"""
    
    def execute(self, tester, test_func):
        # ソリューション処理
        return tester._solution_processor.process_solution(tester.solver, test_func)


class GridConvergenceTestStrategy(TestStrategy):
    """格子収束性テスト戦略"""
    
    def __init__(self, grid_sizes, x_range, y_range=None):
        self.grid_sizes = grid_sizes
        self.x_range = x_range
        self.y_range = y_range
    
    def execute(self, tester, test_func):
        # 結果を保存する辞書
        results = {}
        
        # 現在の設定をバックアップ
        original_settings = {
            'method': tester.solver_method,
            'options': tester.solver_options,
            'analyze': tester.analyze_matrix,
            'equation_set': tester.equation_set,
            'scaling_method': tester.scaling_method
        }

        # 各グリッドサイズでテスト実行
        for n in self.grid_sizes:
            # 新しいテスターを作成
            grid = self._create_grid(tester.get_dimension(), n)
            new_tester = (CCDTester1D(grid) if tester.get_dimension() == 1 else CCDTester2D(grid))
            
            # 設定を引き継ぐ
            new_tester.set_solver_options(
                original_settings['method'], 
                original_settings['options'], 
                original_settings['analyze']
            )
            new_tester.equation_set = original_settings['equation_set']
            new_tester.scaling_method = original_settings['scaling_method']
            
            # テスト実行と結果保存
            result = new_tester.run_test(test_func)
            results[n] = result["errors"]

        return results
    
    def _create_grid(self, dimension, n):
        """次元に応じたグリッドを作成"""
        if dimension == 1:
            return Grid(n, x_range=self.x_range)
        else:
            y_range = self.y_range or self.x_range
            return Grid(n, n, x_range=self.x_range, y_range=y_range)


class SolutionProcessor1D(SolutionProcessor):
    """1Dソリューション処理"""
    
    def process_solution(self, solver, test_func):
        """1D数値解と誤差を計算して結果を返す"""
        # グリッド点での厳密解を計算
        grid = solver.grid
        x = grid.get_points()
        exact_psi = cp.array([test_func.f(xi) for xi in x])
        exact_psi_prime = cp.array([test_func.df(xi) for xi in x])
        exact_psi_second = cp.array([test_func.d2f(xi) for xi in x])
        exact_psi_third = cp.array([test_func.d3f(xi) for xi in x])
        
        # 右辺の値と境界条件の値を準備
        x_min, x_max = grid.x_min, grid.x_max
        
        # 方程式セットのタイプに応じて適切な関数値を使用
        is_derivative_set = isinstance(solver.equation_set, DerivativeEquationSet1D)
        
        if is_derivative_set:
            f_values = cp.array([test_func.f(xi) for xi in x])
        else:
            f_values = cp.array([test_func.d2f(xi) for xi in x])
        
        # 境界条件の値
        left_dirichlet = test_func.f(x_min)
        right_dirichlet = test_func.f(x_max)
        left_neumann = test_func.df(x_min)
        right_neumann = test_func.df(x_max)

        # ソルバーで解を計算
        psi, psi_prime, psi_second, psi_third = solver.solve(
            analyze_before_solve=False,
            f_values=f_values,
            left_dirichlet=left_dirichlet,
            right_dirichlet=right_dirichlet,
            left_neumann=left_neumann,
            right_neumann=right_neumann
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


class SolutionProcessor2D(SolutionProcessor):
    """2Dソリューション処理"""
    
    def process_solution(self, solver, test_func):
        """2D数値解と誤差を計算して結果を返す"""
        # グリッド点を取得
        grid = solver.grid
        X, Y = grid.get_points()
        nx, ny = grid.nx_points, grid.ny_points
        x_min, x_max = grid.x_min, grid.x_max
        y_min, y_max = grid.y_min, grid.y_max

        # 厳密解を計算
        exact_psi = cp.zeros((nx, ny))
        exact_psi_x = cp.zeros((nx, ny))
        exact_psi_y = cp.zeros((nx, ny))
        exact_psi_xx = cp.zeros((nx, ny))
        exact_psi_yy = cp.zeros((nx, ny))
        exact_psi_xxx = cp.zeros((nx, ny))
        exact_psi_yyy = cp.zeros((nx, ny))

        # 各グリッド点で厳密値を計算
        for i in range(nx):
            for j in range(ny):
                x, y = grid.get_point(i, j)
                exact_psi[i, j] = test_func.f(x, y)
                exact_psi_x[i, j] = test_func.df_dx(x, y)
                exact_psi_y[i, j] = test_func.df_dy(x, y)
                exact_psi_xx[i, j] = test_func.d2f_dx2(x, y)
                exact_psi_yy[i, j] = test_func.d2f_dy2(x, y)
                exact_psi_xxx[i, j] = test_func.d3f_dx3(x, y)
                exact_psi_yyy[i, j] = test_func.d3f_dy3(x, y)

        # 方程式セットのタイプに応じて適切な関数値を使用
        is_derivative_set = isinstance(solver.equation_set, DerivativeEquationSet2D)

        # 右辺の値を準備
        f_values = cp.zeros((nx, ny))
        
        if is_derivative_set:
            for i in range(nx):
                for j in range(ny):
                    x, y = grid.get_point(i, j)
                    f_values[i, j] = test_func.f(x, y)
        else:
            for i in range(nx):
                for j in range(ny):
                    x, y = grid.get_point(i, j)
                    f_values[i, j] = test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)

        # 境界条件の値
        boundary_conditions = {
            'left_dirichlet': cp.array([test_func.f(x_min, y) for y in grid.y]),
            'right_dirichlet': cp.array([test_func.f(x_max, y) for y in grid.y]),
            'bottom_dirichlet': cp.array([test_func.f(x, y_min) for x in grid.x]),
            'top_dirichlet': cp.array([test_func.f(x, y_max) for x in grid.x]),
            'left_neumann': cp.array([test_func.df_dx(x_min, y) for y in grid.y]),
            'right_neumann': cp.array([test_func.df_dx(x_max, y) for y in grid.y]),
            'bottom_neumann': cp.array([test_func.df_dy(x, y_min) for x in grid.x]),
            'top_neumann': cp.array([test_func.df_dy(x, y_max) for x in grid.x])
        }

        # ソルバーで解を計算
        psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy = solver.solve(
            analyze_before_solve=False,
            f_values=f_values,
            **boundary_conditions
        )

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


class CCDTester1D(CCDTester):
    """1D CCDメソッドのテストを行うクラス"""
    
    def __init__(self, grid):
        if grid.is_2d:
            raise ValueError("1Dテスターは2Dグリッドでは使用できません")
        super().__init__(grid)
    
    def _create_solver(self):
        """1D用ソルバーを作成"""
        if self.solver is None:
            self.solver = CCDSolver1D(self.equation_set, self.grid)
        else:
            if hasattr(self.solver, 'equation_set'):
                self.solver.equation_set = self.equation_set
            
    def get_dimension(self):
        """次元を返す"""
        return 1
    
    def get_test_function(self, func_name):
        """1Dテスト関数を取得"""
        from test_functions import TestFunctionFactory
        
        test_funcs = TestFunctionFactory.create_standard_1d_functions()
        selected_func = next((f for f in test_funcs if f.name == func_name), None)
        
        if selected_func is None:
            print(f"警告: 1D関数 '{func_name}' が見つかりませんでした。デフォルト関数を使用します。")
            selected_func = test_funcs[0]
            
        return selected_func
    
    def _create_solution_processor(self):
        """1Dソリューション処理オブジェクトを作成"""
        return SolutionProcessor1D()
    
    def _build_rhs_vector(self, test_func):
        """1D右辺ベクトルを構築"""
        x_min, x_max = self.grid.x_min, self.grid.x_max
        
        # ソース項
        f_values = cp.array([test_func.d2f(x) for x in self.grid.x])
        
        # 境界値
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }
        
        self._ensure_solver_initialized()
        return self.solver._build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact_solution(self, test_func):
        """1D厳密解を計算"""
        n_points = self.grid.n_points
        exact_solution = cp.zeros(n_points * 4)
        
        for i in range(n_points):
            xi = self.grid.get_point(i)
            exact_solution[i*4] = test_func.f(xi)
            exact_solution[i*4+1] = test_func.df(xi)
            exact_solution[i*4+2] = test_func.d2f(xi)
            exact_solution[i*4+3] = test_func.d3f(xi)
            
        return exact_solution


class CCDTester2D(CCDTester):
    """2D CCDメソッドのテストを行うクラス"""
    
    def __init__(self, grid):
        if not grid.is_2d:
            raise ValueError("2Dテスターは1Dグリッドでは使用できません")
        super().__init__(grid)
    
    def _create_solver(self):
        """2D用ソルバーを作成"""
        if self.solver is None:
            self.solver = CCDSolver2D(self.equation_set, self.grid)
        else:
            if hasattr(self.solver, 'equation_set'):
                self.solver.equation_set = self.equation_set
            
    def get_dimension(self):
        """次元を返す"""
        return 2
    
    def get_test_function(self, func_name):
        """2Dテスト関数を取得"""
        from test_functions import TestFunctionFactory, TestFunction
        
        # まず基本的な2D関数を生成
        standard_funcs = TestFunctionFactory.create_standard_2d_functions()
        
        # 指定された名前の関数を検索
        selected_func = next((f for f in standard_funcs if f.name == func_name), None)
        if selected_func is not None:
            return selected_func
        
        # 見つからない場合は、1D関数から動的に生成を試みる
        funcs_1d = TestFunctionFactory.create_standard_1d_functions()
        func_1d = next((f for f in funcs_1d if f.name == func_name), None)
        
        if func_1d is not None:
            # テンソル積拡張を作成
            return TestFunction.from_1d_to_2d(func_1d, method='product')
        
        # それでも見つからない場合は、最初の関数を返す
        print(f"警告: 2D関数 '{func_name}' が見つかりませんでした。デフォルト関数を使用します。")
        return standard_funcs[0]
    
    def _create_solution_processor(self):
        """2Dソリューション処理オブジェクトを作成"""
        return SolutionProcessor2D()
    
    def _build_rhs_vector(self, test_func):
        """2D右辺ベクトルを構築"""
        # 2Dケース
        nx, ny = self.grid.nx_points, self.grid.ny_points
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        
        # ソース項
        f_values = cp.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x, y = self.grid.get_point(i, j)
                f_values[i, j] = test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)
        
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
        
        self._ensure_solver_initialized()
        return self.solver._build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact_solution(self, test_func):
        """2D厳密解を計算"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        exact_solution = cp.zeros(nx * ny * 7)
        
        for j in range(ny):
            for i in range(nx):
                idx = (j * nx + i) * 7
                x, y = self.grid.get_point(i, j)
                exact_solution[idx] = test_func.f(x, y)
                exact_solution[idx+1] = test_func.df_dx(x, y)
                exact_solution[idx+2] = test_func.d2f_dx2(x, y)
                exact_solution[idx+3] = test_func.d3f_dx3(x, y)
                exact_solution[idx+4] = test_func.df_dy(x, y)
                exact_solution[idx+5] = test_func.d2f_dy2(x, y)
                exact_solution[idx+6] = test_func.d3f_dy3(x, y)
                
        return exact_solution

