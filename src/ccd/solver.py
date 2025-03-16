"""
高精度コンパクト差分法 (CCD) のための線形方程式系ソルバーモジュール

このモジュールは1次元・2次元のポアソン方程式および高階微分方程式を解くための
ソルバーを提供します。equation_system.pyと連携して方程式系を構築し解きます。
"""

import os
import time
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from equation_system import EquationSystem
from scaling import plugin_manager

# Add these imports at the top of the file
import numpy as np
import scipy.sparse as sp_cpu
import scipy.sparse.linalg as splinalg_cpu

class LinearSystemSolver:
    """線形方程式系 Ax=b を解くためのクラス"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        """
        線形方程式系ソルバーを初期化
        
        Args:
            method: 解法メソッド ("direct", "gmres", "cg", "cgs", "minres", "lsqr")
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        self.method = method
        self.options = options or {}
        self.scaling_method = scaling_method
        self.last_iterations = None
        self.monitor_convergence = self.options.get("monitor_convergence", False)
    

    # Add these helper methods to the LinearSystemSolver class
    def _solve_with_gpu(self, A, b, callback=None):
        """GPU (CuPy) を使用して線形方程式系を解く"""
        if self.method == "direct":
            x = splinalg.spsolve(A, b)
            iterations = None
        elif self.method == "gmres":
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            restart = self.options.get("restart", 100)
            x0 = cp.ones_like(b)
            x, iterations = splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                    restart=restart, callback=callback)
        elif self.method in ["cg", "cgs", "minres"]:
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg, self.method)
            x0 = cp.ones_like(b)
            x, iterations = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        elif self.method in ["lsqr", "lsmr"]:
            solver_func = getattr(splinalg, self.method)
            maxiter = self.options.get("maxiter", 1000)
            if self.method == "lsmr":
                x0 = cp.ones_like(b)
                x = solver_func(A, b, x0=x0, maxiter=maxiter)[0]
            else:
                x = solver_func(A, b)[0]
            iterations = None
        else:
            print(f"未知の解法 {self.method}。直接解法を使用します。")
            x = splinalg.spsolve(A, b)
            iterations = None
        
        return x, iterations

    def _solve_with_cpu(self, A, b):
        """CPU (SciPy) を使用して線形方程式系を解く"""
        # CuPy 配列を NumPy 配列に変換
        A_cpu = A.get() if hasattr(A, 'get') else A
        b_cpu = b.get() if hasattr(b, 'get') else b
        
        # CSR 形式の行列に変換 (必要な場合)
        if not isinstance(A_cpu, sp_cpu.csr_matrix):
            A_cpu = sp_cpu.csr_matrix(A_cpu)
        
        # CPU (SciPy) での解法
        if self.method == "direct" or self.method not in ["gmres", "cg", "cgs", "minres", "lsqr", "lsmr"]:
            x_cpu = splinalg_cpu.spsolve(A_cpu, b_cpu)
            iterations = None
        elif self.method == "gmres":
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            restart = self.options.get("restart", 100)
            x0_cpu = np.ones_like(b_cpu)
            x_cpu, iterations = splinalg_cpu.gmres(A_cpu, b_cpu, x0=x0_cpu, tol=tol, maxiter=maxiter, restart=restart)
        elif self.method in ["cg", "cgs", "minres"]:
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg_cpu, self.method)
            x0_cpu = np.ones_like(b_cpu)
            x_cpu, iterations = solver_func(A_cpu, b_cpu, x0=x0_cpu, tol=tol, maxiter=maxiter)
        elif self.method in ["lsqr", "lsmr"]:
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg_cpu, self.method)
            x_cpu = solver_func(A_cpu, b_cpu)[0]
            iterations = None
        
        # NumPy 配列を CuPy 配列に変換
        x = cp.array(x_cpu)
        return x, iterations

    # Replace the existing solve method with this new version
    def solve(self, A, b):
        """
        線形方程式系を解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            解ベクトル x
        """
        # スケーリングを適用
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(A, b)
        
        # 方程式を解く
        start_time = time.time()
        residuals = []
        
        # モニタリングコールバック
        if self.monitor_convergence:
            def callback(xk):
                residual = cp.linalg.norm(b_scaled - A_scaled @ xk) / cp.linalg.norm(b_scaled)
                residuals.append(float(residual))
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        else:
            callback = None
        
        # GPU で計算を試行
        try:
            x, iterations = self._solve_with_gpu(A_scaled, b_scaled, callback)
        except Exception as e:
            error_msg = str(e)
            # GPU メモリ不足の場合
            if "CUSOLVER_STATUS_ALLOC_FAILED" in error_msg or "CUDA" in error_msg or "GPU" in error_msg or "cuSOLVER" in error_msg:
                print(f"GPU メモリ不足のエラー: {e}")
                print("CPU (SciPy) に切り替えて計算を実行します...")
                x, iterations = self._solve_with_cpu(A_scaled, b_scaled)
                print("CPU (SciPy) での計算が完了しました。")
            else:
                # その他のエラーの場合は、まず CuPy の直接解法を試す
                print(f"解法エラー: {e}。直接解法にフォールバックします。")
                try:
                    x = splinalg.spsolve(A_scaled, b_scaled)
                    iterations = None
                except Exception as e2:
                    # それでもダメなら SciPy を使用
                    print(f"CuPy の直接解法でもエラー: {e2}。SciPy に切り替えます。")
                    x, iterations = self._solve_with_cpu(A_scaled, b_scaled)
                    print("CPU (SciPy) での計算が完了しました。")
        
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        
        # 実行情報表示
        print(f"解法実行: {self.method}, 経過時間: {elapsed:.4f}秒")
        if iterations:
            print(f"反復回数: {iterations}")
        
        # スケーリングを解除
        if scaling_info is not None and scaler is not None:
            x = scaler.unscale(x, scaling_info)
            
        # 収束履歴の可視化
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
        
        return x
    
    def _apply_scaling(self, A, b):
        """
        行列と右辺ベクトルにスケーリングを適用
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scaling_info, scaler)
        """
        if self.scaling_method is not None:
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング: {scaler.name}")
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
        
        return A, b, None, None
    
    def _visualize_convergence(self, residuals):
        """収束履歴をグラフ化（必要な場合）"""
        if not residuals:
            return
            
        # 出力ディレクトリを確保
        output_dir = self.options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 残差の推移グラフ
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals)+1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{self.method.upper()} ソルバーの収束履歴')
        
        # 保存
        prefix = self.options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_{self.method}.png")
        plt.savefig(filename, dpi=150)
        plt.close()


class BaseCCDSolver(ABC):
    """コンパクト差分法ソルバーの抽象基底クラス"""

    def __init__(self, equation_set, grid):
        """
        ソルバーを初期化
        
        Args:
            equation_set: 使用する方程式セット
            grid: グリッドオブジェクト
        """
        self.equation_set = equation_set
        self.grid = grid
        self.linear_solver = LinearSystemSolver()
        
        # システムを初期化し、行列Aを構築
        self.system = EquationSystem(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()
    
    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        ソルバーの設定
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        self.linear_solver = LinearSystemSolver(method, options, scaling_method)
    
    @property
    def scaling_method(self):
        """スケーリング手法を取得"""
        return self.linear_solver.scaling_method
    
    @scaling_method.setter
    def scaling_method(self, value):
        """スケーリング手法を設定"""
        self.linear_solver.scaling_method = value
    
    @property
    def last_iterations(self):
        """最後の反復回数を取得"""
        return self.linear_solver.last_iterations

    def analyze_system(self):
        """
        行列システムの疎性を分析
        
        Returns:
            疎性情報の辞書
        """
        A = self.matrix_A
        total_size = A.shape[0]
        nnz = A.nnz
        sparsity = 1.0 - (nnz / (total_size * total_size))
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)
        
        print("\n行列構造分析:")
        print(f"  行列サイズ: {total_size} x {total_size}")
        print(f"  非ゼロ要素数: {nnz}")
        print(f"  疎性率: {sparsity:.6f}")
        print(f"  メモリ使用量(密行列): {memory_dense_MB:.2f} MB")
        print(f"  メモリ使用量(疎行列): {memory_sparse_MB:.2f} MB")
        
        return {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }

    def solve(self, analyze_before_solve=True, f_values=None, **boundary_values):
        """
        システムを解く
        
        Args:
            analyze_before_solve: 解く前に行列を分析するかどうか
            f_values: 支配方程式の右辺値
            **boundary_values: 境界値の辞書（ディメンションに依存）
            
        Returns:
            解コンポーネント
        """
        # 行列を分析（要求された場合）
        if analyze_before_solve:
            self.analyze_system()
            
        # 右辺ベクトルbを構築
        b = self._build_rhs_vector(f_values, **boundary_values)
        
        # 線形システムを解く
        sol = self.linear_solver.solve(self.matrix_A, b)

        # 解ベクトルから各要素を抽出
        return self._extract_solution(sol)

    @abstractmethod
    def _build_rhs_vector(self, f_values=None, **boundary_values):
        """右辺ベクトルを構築（次元による具体実装）"""
        pass
    
    @abstractmethod
    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出（次元による具体実装）"""
        pass


class CCDSolver1D(BaseCCDSolver):
    """1次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        """
        1Dソルバーを初期化
        
        Args:
            equation_set: 方程式セット
            grid: 1D グリッドオブジェクト
        """
        if grid.is_2d:
            raise ValueError("1Dソルバーは2Dグリッドでは使用できません")
            
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        left_neumann=None, right_neumann=None, **kwargs):
        """
        1D右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値（全格子点の配列）
            left_dirichlet, right_dirichlet: ディリクレ境界値
            left_neumann, right_neumann: ノイマン境界値
            
        Returns:
            右辺ベクトル
        """
        n = self.grid.n_points
        var_per_point = 4  # [ψ, ψ', ψ'', ψ''']
        b = cp.zeros(n * var_per_point)
        
        # 境界条件に関する情報を出力
        boundary_info = []
        if self.enable_dirichlet:
            boundary_info.append(f"ディリクレ境界条件: 左={left_dirichlet}, 右={right_dirichlet}")
        if self.enable_neumann:
            boundary_info.append(f"ノイマン境界条件: 左={left_neumann}, 右={right_neumann}")
        if boundary_info:
            print(f"[1Dソルバー] " + "; ".join(boundary_info))
        
        # 各格子点に対して処理
        for i in range(n):
            base_idx = i * var_per_point
            location = self.system._get_point_location(i)
            location_equations = self.system.equations[location]
            
            # 方程式を種類別に分類
            eq_by_type = {"governing": None, "dirichlet": None, "neumann": None, "auxiliary": []}
            for eq in location_equations:
                eq_type = self.system._identify_equation_type(eq, i)
                if eq_type == "auxiliary":
                    eq_by_type["auxiliary"].append(eq)
                elif eq_type:  # Noneでない場合
                    eq_by_type[eq_type] = eq
            
            # ソース項の処理（支配方程式に対応する行）
            if eq_by_type["governing"] and f_values is not None:
                b[base_idx] = f_values[i]
            
            # 境界条件の処理
            if location == 'left':
                if self.enable_dirichlet and left_dirichlet is not None:
                    dirichlet_row = self._find_dirichlet_row(base_idx, location_equations, i)
                    if dirichlet_row is not None:
                        b[dirichlet_row] = left_dirichlet
                        
                if self.enable_neumann and left_neumann is not None:
                    neumann_row = self._find_neumann_row(base_idx, location_equations, i)
                    if neumann_row is not None:
                        b[neumann_row] = left_neumann
                        
            elif location == 'right':
                if self.enable_dirichlet and right_dirichlet is not None:
                    dirichlet_row = self._find_dirichlet_row(base_idx, location_equations, i)
                    if dirichlet_row is not None:
                        b[dirichlet_row] = right_dirichlet
                        
                if self.enable_neumann and right_neumann is not None:
                    neumann_row = self._find_neumann_row(base_idx, location_equations, i)
                    if neumann_row is not None:
                        b[neumann_row] = right_neumann
        
        return b
    
    def _find_dirichlet_row(self, base_idx, equations, i):
        """ディリクレ境界条件に対応する行インデックスを見つける"""
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "dirichlet":
                return base_idx + row_offset
        return None
    
    def _find_neumann_row(self, base_idx, equations, i):
        """ノイマン境界条件に対応する行インデックスを見つける"""
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "neumann":
                return base_idx + row_offset
        return None

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出"""
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third


class CCDSolver2D(BaseCCDSolver):
    """2次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        """
        2Dソルバーを初期化
        
        Args:
            equation_set: 方程式セット
            grid: 2D グリッドオブジェクト
        """
        if not grid.is_2d:
            raise ValueError("2Dソルバーは1Dグリッドでは使用できません")
            
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        bottom_dirichlet=None, top_dirichlet=None, left_neumann=None, 
                        right_neumann=None, bottom_neumann=None, top_neumann=None, **kwargs):
        """
        2D右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値 (nx×ny配列)
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: 境界値
            left_neumann, right_neumann, bottom_neumann, top_neumann: 境界導関数
            
        Returns:
            右辺ベクトル
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        var_per_point = 7  # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        b = cp.zeros(nx * ny * var_per_point)
        
        # 境界条件の状態を出力
        self._print_boundary_info(
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet,
            left_neumann, right_neumann, bottom_neumann, top_neumann
        )
        
        # 各格子点に対して処理
        for j in range(ny):
            for i in range(nx):
                base_idx = (j * nx + i) * var_per_point
                location = self.system._get_point_location(i, j)
                location_equations = self.system.equations[location]
                
                # 方程式を種類別に分類
                eq_by_type = {
                    "governing": None, 
                    "dirichlet": None, 
                    "neumann_x": None, 
                    "neumann_y": None, 
                    "auxiliary": []
                }
                
                for eq in location_equations:
                    eq_type = self.system._identify_equation_type(eq, i, j)
                    if eq_type == "auxiliary":
                        eq_by_type["auxiliary"].append(eq)
                    elif eq_type:  # Noneでない場合
                        eq_by_type[eq_type] = eq
                
                # 各行に割り当てる方程式を決定
                assignments = self.system._assign_equations_2d(eq_by_type, i, j)
                
                # ソース項の処理（支配方程式に対応する行）
                governing_row = self._find_governing_row(assignments)
                if governing_row is not None and f_values is not None:
                    b[base_idx + governing_row] = f_values[i, j]
                
                # 境界条件の処理
                self._apply_boundary_values(
                    b, base_idx, location, assignments,
                    left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet,
                    left_neumann, right_neumann, bottom_neumann, top_neumann,
                    i, j
                )
        
        return b
    
    def _print_boundary_info(self, left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet,
                          left_neumann, right_neumann, bottom_neumann, top_neumann):
        """境界条件の情報を出力"""
        if self.enable_dirichlet:
            print(f"[2Dソルバー] ディリクレ境界条件: "
                  f"左={left_dirichlet is not None}, 右={right_dirichlet is not None}, "
                  f"下={bottom_dirichlet is not None}, 上={top_dirichlet is not None}")
        if self.enable_neumann:
            print(f"[2Dソルバー] ノイマン境界条件: "
                  f"左={left_neumann is not None}, 右={right_neumann is not None}, "
                  f"下={bottom_neumann is not None}, 上={top_neumann is not None}")
    
    def _find_governing_row(self, assignments):
        """支配方程式が割り当てられた行を見つける"""
        from equation.poisson import PoissonEquation, PoissonEquation2D
        from equation.original import OriginalEquation, OriginalEquation2D
        
        for row, eq in enumerate(assignments):
            if isinstance(eq, (PoissonEquation, PoissonEquation2D, OriginalEquation, OriginalEquation2D)):
                return row
        return None
    
    def _apply_boundary_values(self, b, base_idx, location, assignments,
                            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet,
                            left_neumann, right_neumann, bottom_neumann, top_neumann,
                            i, j):
        """適切な場所に境界値を設定"""
        # インポートは関数内で行い、依存関係をローカルに限定
        from equation.boundary import (
            DirichletBoundaryEquation, NeumannBoundaryEquation,
            DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
        )
        
        # 境界値の取得
        def get_boundary_value(value, idx):
            if isinstance(value, (list, cp.ndarray)) and idx < len(value):
                return value[idx]
            return value
        
        # 各行に割り当てられた方程式をチェック
        for row, eq in enumerate(assignments):
            # ディリクレ条件
            if isinstance(eq, DirichletBoundaryEquation2D) and self.enable_dirichlet:
                if 'left' in location and left_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(left_dirichlet, j)
                elif 'right' in location and right_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(right_dirichlet, j)
                elif 'bottom' in location and bottom_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(bottom_dirichlet, i)
                elif 'top' in location and top_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(top_dirichlet, i)
            
            # X方向ノイマン条件
            elif isinstance(eq, NeumannXBoundaryEquation2D) and self.enable_neumann:
                if 'left' in location and left_neumann is not None:
                    b[base_idx + row] = get_boundary_value(left_neumann, j)
                elif 'right' in location and right_neumann is not None:
                    b[base_idx + row] = get_boundary_value(right_neumann, j)
            
            # Y方向ノイマン条件
            elif isinstance(eq, NeumannYBoundaryEquation2D) and self.enable_neumann:
                if 'bottom' in location and bottom_neumann is not None:
                    b[base_idx + row] = get_boundary_value(bottom_neumann, i)
                elif 'top' in location and top_neumann is not None:
                    b[base_idx + row] = get_boundary_value(top_neumann, i)

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # 解配列を初期化
        psi = cp.zeros((nx, ny))
        psi_x = cp.zeros((nx, ny))
        psi_xx = cp.zeros((nx, ny))
        psi_xxx = cp.zeros((nx, ny))
        psi_y = cp.zeros((nx, ny))
        psi_yy = cp.zeros((nx, ny))
        psi_yyy = cp.zeros((nx, ny))
        
        # 各グリッド点の値を抽出
        for j in range(ny):
            for i in range(nx):
                idx = (j * nx + i) * n_unknowns
                psi[i, j] = sol[idx]
                psi_x[i, j] = sol[idx + 1]
                psi_xx[i, j] = sol[idx + 2]
                psi_xxx[i, j] = sol[idx + 3]
                psi_y[i, j] = sol[idx + 4]
                psi_yy[i, j] = sol[idx + 5]
                psi_yyy[i, j] = sol[idx + 6]
        
        return psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy