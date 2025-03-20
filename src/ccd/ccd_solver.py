"""
高精度コンパクト差分法 (CCD) を用いた偏微分方程式ソルバーモジュール

このモジュールは、ポアソン方程式および高階微分方程式を1次元・2次元で
解くためのソルバークラスを提供します。方程式システムの構築や境界条件の
適用も含め、高精度コンパクト差分法による数値解法の全体を担当します。
"""

import numpy as np
from abc import ABC, abstractmethod

from equation_system import EquationSystem
from rhs_builder import RHSBuilder1D, RHSBuilder2D
# 新しい線形ソルバーパッケージをインポート
from linear_solver import create_solver


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
        
        # システムを初期化し、行列Aを構築 (CPU処理)
        self.system = EquationSystem(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()
        
        # デフォルトソルバーとRHSビルダーを設定
        self._create_rhs_builder()
        # 新しいAPIに合わせてLinearSolverを初期化
        self.linear_solver = create_solver(
            self.matrix_A,
            enable_dirichlet=self.enable_dirichlet,
            enable_neumann=self.enable_neumann,
            backend="cuda"  # デフォルトバックエンド
        )
        
        # デフォルト解法メソッド
        self.method = "direct"
    
    @abstractmethod
    def _create_rhs_builder(self):
        """次元に応じたRHSビルダーを作成"""
        pass
    
    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        ソルバーの設定
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        # オプションの初期化
        options = options or {}
        
        # バックエンド指定を取得
        backend = options.get("backend", "cuda")
        
        # 直接線形ソルバーに渡すオプションを抽出
        solver_options = {}
        for key, value in options.items():
            # LinearSolverで使用するオプションのみ抽出
            if key in ["tol", "maxiter", "restart", "inner_m", "outer_k", "m", "k"]:
                solver_options[key] = value
        
        # 新たにソルバーを作成
        self.linear_solver = create_solver(
            self.matrix_A, 
            enable_dirichlet=self.enable_dirichlet,
            enable_neumann=self.enable_neumann,
            scaling_method=scaling_method,
            backend=backend
        )
        
        # LinearSolverにオプションとmehtodを実際に渡す
        self.linear_solver.solver_method = method
        self.linear_solver.solver_options = solver_options
        
        # 解法メソッドを保存
        self.method = method
    
    @property
    def scaling_method(self):
        """スケーリング手法を取得"""
        return self.linear_solver.scaling_method if hasattr(self.linear_solver, 'scaling_method') else None
    
    @scaling_method.setter
    def scaling_method(self, value):
        """スケーリング手法を設定"""
        # 新しいソルバーを作成する必要がある
        backend = "cuda"  # デフォルトバックエンド
        self.set_solver(method=self.method, scaling_method=value, options={"backend": backend})
    
    @property
    def last_iterations(self):
        """最後の反復回数を取得"""
        return self.linear_solver.last_iterations if hasattr(self.linear_solver, 'last_iterations') else None
    
    def get_boundary_settings(self):
        """境界条件の設定を取得"""
        return self.enable_dirichlet, self.enable_neumann

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
        
        print("\nMatrix structure analysis:")
        print(f"  Matrix size: {total_size} x {total_size}")
        print(f"  Non-zero elements: {nnz}")
        print(f"  Sparsity: {sparsity:.6f}")
        print(f"  Memory usage (dense): {memory_dense_MB:.2f} MB")
        print(f"  Memory usage (sparse): {memory_sparse_MB:.2f} MB")
        
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
        b = self.rhs_builder.build_rhs_vector(f_values, **boundary_values)
        
        # 線形システムを解く（既に設定済みのメソッドとオプションを使用）
        sol = self.linear_solver.solve(b)

        # 解ベクトルから各要素を抽出
        return self._extract_solution(sol)
    
    @abstractmethod
    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出（次元による具体実装）"""
        pass
    
    def _to_numpy(self, arr):
        """CuPy配列をNumPy配列に変換する (必要な場合のみ)"""
        if hasattr(arr, 'get'):
            return arr.get()
        return arr


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
    
    def _create_rhs_builder(self):
        """1次元RHSビルダーを作成"""
        self.rhs_builder = RHSBuilder1D(
            self.system, 
            self.grid, 
            enable_dirichlet=self.enable_dirichlet, 
            enable_neumann=self.enable_neumann
        )

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
    
    def _create_rhs_builder(self):
        """2次元RHSビルダーを作成"""
        self.rhs_builder = RHSBuilder2D(
            self.system, 
            self.grid,
            enable_dirichlet=self.enable_dirichlet, 
            enable_neumann=self.enable_neumann
        )

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # 解配列を初期化 (NumPy配列)
        psi = np.zeros((nx, ny))
        psi_x = np.zeros((nx, ny))
        psi_xx = np.zeros((nx, ny))
        psi_xxx = np.zeros((nx, ny))
        psi_y = np.zeros((nx, ny))
        psi_yy = np.zeros((nx, ny))
        psi_yyy = np.zeros((nx, ny))
        
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