"""
高精度コンパクト差分法 (CCD) を用いた偏微分方程式ソルバーの基底クラス

このモジュールは、ポアソン方程式および高階微分方程式を解くための
基底ソルバークラスを提供します。共通機能の実装と、次元に依存しない
処理を担当します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np

from equation_system import EquationSystem
# linear_solver パッケージからソルバー作成関数をインポート
from linear_solver import create_solver


class BaseCCDSolver(ABC):
    """コンパクト差分法ソルバーの抽象基底クラス"""

    def __init__(self, equation_set, grid, backend="cpu"):
        """
        ソルバーを初期化
        
        Args:
            equation_set: 使用する方程式セット
            grid: グリッドオブジェクト
            backend: 計算バックエンド ('cpu', 'cuda', 'jax')
        """
        self.equation_set = equation_set
        self.grid = grid
        self.backend = backend
        
        # システムを初期化し、行列Aを構築 (CPU処理)
        self.system = EquationSystem(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()
        
        # デフォルトソルバーとRHSビルダーを設定
        self._create_rhs_builder()
        # LinearSolverを初期化
        self.linear_solver = create_solver(
            self.matrix_A,
            enable_dirichlet=self.enable_dirichlet,
            enable_neumann=self.enable_neumann,
            backend=self.backend
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
        backend = options.get("backend", self.backend)
        
        # 直接線形ソルバーに渡すオプションを抽出
        solver_options = {}
        for key, value in options.items():
            # LinearSolverで使用するオプションのみ抽出
            if key in ["tol", "maxiter", "restart", "inner_m", "outer_k", "m", "k", "x0"]:
                solver_options[key] = value
        
        # x0が含まれている場合は確認用に表示
        if "x0" in solver_options:
            print(f"x0が設定されました (shape: {solver_options['x0'].shape})")
        
        # 新たにソルバーを作成
        self.linear_solver = create_solver(
            self.matrix_A, 
            enable_dirichlet=self.enable_dirichlet,
            enable_neumann=self.enable_neumann,
            scaling_method=scaling_method,
            backend=backend
        )
        
        # バックエンドを更新
        self.backend = backend
        
        # LinearSolverにオプションとmethodを設定
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
        self.set_solver(method=self.method, scaling_method=value, options={"backend": self.backend})
    
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
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8バイト/倍精度浮動小数点数
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 値8バイト + インデックス4バイト
        
        print("\n行列構造の分析:")
        print(f"  行列サイズ: {total_size} x {total_size}")
        print(f"  非ゼロ要素: {nnz}")
        print(f"  疎性率: {sparsity:.6f}")
        print(f"  メモリ使用量 (密行列): {memory_dense_MB:.2f} MB")
        print(f"  メモリ使用量 (疎行列): {memory_sparse_MB:.2f} MB")
        
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
        
        # 現在のオプションを確認（デバッグ情報）
        if hasattr(self.linear_solver, 'solver_options'):
            print(f"solveメソッド内のソルバーオプション: {self.linear_solver.solver_options}")
        
        # 線形システムを解く（既に設定済みのメソッドとオプションを使用）
        sol = self.linear_solver.solve(b)

        # 解ベクトルから各要素を抽出（次元に依存する処理）
        return self._extract_solution(sol)
    
    def solve_with_options(self, analyze_before_solve=True, f_values=None, solve_options=None, **boundary_values):
        """
        カスタムオプションを使用してシステムを解く
        
        Args:
            analyze_before_solve: 解く前に行列を分析するかどうか
            f_values: 支配方程式の右辺値
            solve_options: 直接渡すソルバーオプション
            **boundary_values: 境界値の辞書（ディメンションに依存）
            
        Returns:
            解コンポーネント
        """
        # 行列を分析（要求された場合）
        if analyze_before_solve:
            self.analyze_system()
            
        # 右辺ベクトルbを構築
        b = self.rhs_builder.build_rhs_vector(f_values, **boundary_values)
        
        # オプションを確認（デバッグ情報）
        if solve_options and "x0" in solve_options:
            print(f"solve_with_optionsメソッド内のx0: {solve_options['x0'].shape}")
        
        # 線形システムを解く（指定されたオプションとメソッドを使用）
        sol = self.linear_solver.solve(b, method=self.method, options=solve_options)

        # 解ベクトルから各要素を抽出（次元に依存する処理）
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
