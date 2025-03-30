"""
方程式システムの基底クラスを定義するモジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
格子点の方程式をシステム全体として管理するための基底クラスと共通機能を提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Union, Optional
import numpy as np
import scipy.sparse as sp_cpu


class BaseEquationSystem(ABC):
    """方程式システムの抽象基底クラス"""

    def __init__(self, grid):
        """
        方程式システムを初期化
        
        Args:
            grid: 計算格子オブジェクト (1D, 2D, または 3D)
        """
        self.grid = grid
        self.is_2d = grid.is_2d if hasattr(grid, 'is_2d') else False
        self.is_3d = grid.is_3d if hasattr(grid, 'is_3d') else False
        
        # 次元を特定
        if self.is_3d:
            self.dimension = 3
        elif self.is_2d:
            self.dimension = 2
        else:
            self.dimension = 1
            
        # 変数の数（次元に応じて決定）
        self.var_per_point = self._get_vars_per_point()
        
        # 領域ごとの方程式コレクション（サブクラスで初期化）
        self.equations = {}
        
        # 位置キャッシュ（計算効率化のため）
        self._location_cache = {}
        
        # 次元に応じた初期化
        self._initialize_equations()
    
    def _get_vars_per_point(self) -> int:
        """次元に応じた1点あたりの変数数を返す"""
        if self.is_3d:
            return 10  # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        elif self.is_2d:
            return 7   # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        else:
            return 4   # [ψ, ψ', ψ'', ψ''']
    
    @abstractmethod
    def _initialize_equations(self):
        """
        次元に応じた方程式コレクションを初期化
        サブクラスで実装する
        """
        pass
    
    def add_equation(self, region: str, equation) -> None:
        """
        指定された領域に方程式を追加
        
        Args:
            region: 領域識別子 ('interior', 'left', 'right', 等)
            equation: 追加する方程式オブジェクト
        """
        # 方程式にグリッドを設定（必要な場合）
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        # 指定された領域に方程式を追加
        if region in self.equations:
            self.equations[region].append(equation)
        else:
            raise ValueError(f"未知の領域: {region}")
    
    def add_equations(self, region: str, equations: List) -> None:
        """
        指定された領域に複数の方程式を一度に追加
        
        Args:
            region: 領域識別子
            equations: 追加する方程式のリスト
        """
        for equation in equations:
            self.add_equation(region, equation)
    
    def add_dominant_equation(self, equation) -> None:
        """
        支配方程式をすべての領域に追加
        
        Args:
            equation: 支配方程式
        """
        for region in self.equations:
            self.add_equation(region, equation)
    
    def build_matrix_system(self):
        """
        行列システムを構築（共通実装）
        
        Returns:
            システム行列 (CSR形式、CPU)
        """
        # 簡易的な検証
        self._validate_equations()
        
        # システムのサイズを計算
        system_size = self._calculate_system_size()
        
        # LIL形式で行列を初期化（メモリ効率の良い構築）
        A_lil = sp_cpu.lil_matrix((system_size, system_size))
        
        # グリッドサイズ情報を取得（次元に応じた処理）
        grid_sizes = self._get_grid_sizes()
        
        # 各格子点について処理
        self._build_matrix_for_all_points(A_lil, grid_sizes)
        
        # LIL行列をCSR形式に変換（計算効率向上のため）
        return A_lil.tocsr()
    
    def _build_matrix_for_all_points(self, A_lil, grid_sizes):
        """全ての格子点に対して行列を構築する"""
        if self.dimension == 1:
            nx = grid_sizes[0]
            for i in range(nx):
                self._build_matrix_for_point(A_lil, i)
        elif self.dimension == 2:
            nx, ny = grid_sizes
            for j in range(ny):
                for i in range(nx):
                    self._build_matrix_for_point(A_lil, i, j)
        else:  # dimension == 3
            nx, ny, nz = grid_sizes
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        self._build_matrix_for_point(A_lil, i, j, k)
    
    def _build_matrix_for_point(self, A_lil, *indices):
        """
        特定の格子点に対する行列要素を構築
        
        Args:
            A_lil: LIL形式の行列
            *indices: 格子点のインデックス (i, j, k)
        """
        # 点の位置と基準インデックスを取得
        location = self._get_point_location(*indices)
        base_idx = self._calculate_base_index(indices)
        
        # その位置に対応する方程式群を取得
        location_equations = self.equations[location]
        
        # 方程式を種類別に分類
        eq_by_type = self._classify_equations(location_equations, *indices)
        
        # 各種方程式の存在確認
        if not eq_by_type["governing"]:
            coord_str = ", ".join(str(idx) for idx in indices)
            raise ValueError(f"点 ({coord_str}) に支配方程式が設定されていません")
        
        # 方程式の割り当て（次元に応じて適切なメソッドを呼び出す）
        if self.dimension == 1:
            assignments = self._assign_equations_1d(eq_by_type, *indices)
        elif self.dimension == 2:
            assignments = self._assign_equations_2d(eq_by_type, *indices)
        else:  # dimension == 3
            assignments = self._assign_equations_3d(eq_by_type, *indices)
        
        # 各行の方程式から係数を行列に追加
        for row_offset, eq in enumerate(assignments):
            if eq is None:
                coord_str = ", ".join(str(idx) for idx in indices)
                raise ValueError(f"点 ({coord_str}) の {row_offset} 行目に方程式が割り当てられていません")
            
            row = base_idx + row_offset
            
            # ステンシル係数を取得
            stencil_coeffs = eq.get_stencil_coefficients(*indices)
            
            # 係数を行列に追加
            self._add_coefficients_to_matrix(A_lil, row, stencil_coeffs, indices)
    
    def _add_coefficients_to_matrix(self, A_lil, row, stencil_coeffs, point_indices):
        """
        ステンシル係数を行列に追加
        
        Args:
            A_lil: LIL形式の行列
            row: 行インデックス
            stencil_coeffs: ステンシル係数の辞書
            point_indices: 現在の格子点インデックス
        """
        grid_sizes = self._get_grid_sizes()
        
        for offset, coeffs in stencil_coeffs.items():
            # 相対オフセットから絶対インデックスを計算
            neighbor_indices = self._get_neighbor_indices(point_indices, offset, grid_sizes)
            
            # 範囲内チェック
            if self._is_valid_indices(neighbor_indices, grid_sizes):
                col_base = self._calculate_base_index(neighbor_indices)
                
                # 各変数の係数を設定
                for k, coeff in enumerate(coeffs):
                    if coeff != 0.0:  # 非ゼロ要素のみ追加
                        A_lil[row, col_base + k] = float(self._to_numpy(coeff))
    
    def _get_neighbor_indices(self, point_indices, offset, grid_sizes):
        """インデックスにオフセットを適用して隣接点のインデックスを得る"""
        if self.dimension == 1:
            return (point_indices[0] + offset,)
        elif self.dimension == 2:
            return (point_indices[0] + offset[0], point_indices[1] + offset[1])
        else:  # dimension == 3
            return (point_indices[0] + offset[0], point_indices[1] + offset[1], point_indices[2] + offset[2])
    
    def _is_valid_indices(self, indices, grid_sizes):
        """インデックスが有効範囲内かチェック"""
        for idx, size in zip(indices, grid_sizes):
            if idx < 0 or idx >= size:
                return False
        return True
    
    def _calculate_system_size(self) -> int:
        """システムの合計サイズを計算"""
        grid_sizes = self._get_grid_sizes()
        
        if self.dimension == 1:
            return grid_sizes[0] * self.var_per_point
        elif self.dimension == 2:
            return grid_sizes[0] * grid_sizes[1] * self.var_per_point
        else:  # dimension == 3
            return grid_sizes[0] * grid_sizes[1] * grid_sizes[2] * self.var_per_point
    
    def _calculate_base_index(self, indices) -> int:
        """
        グリッドインデックスから行列の基準インデックスを計算
        
        Args:
            indices: グリッドインデックス (i,) or (i, j) or (i, j, k)
            
        Returns:
            行列内の基準インデックス
        """
        if self.dimension == 1:
            i = indices[0]
            return i * self.var_per_point
        elif self.dimension == 2:
            i, j = indices
            nx = self.grid.nx_points
            return (j * nx + i) * self.var_per_point
        else:  # dimension == 3
            i, j, k = indices
            nx, ny = self.grid.nx_points, self.grid.ny_points
            return (k * ny * nx + j * nx + i) * self.var_per_point
    
    def _get_grid_sizes(self) -> Tuple[int, ...]:
        """グリッドサイズ情報を返す"""
        if self.dimension == 1:
            return (self.grid.n_points,)
        elif self.dimension == 2:
            return (self.grid.nx_points, self.grid.ny_points)
        else:  # dimension == 3
            return (self.grid.nx_points, self.grid.ny_points, self.grid.nz_points)
    
    @abstractmethod
    def _get_point_location(self, *indices) -> str:
        """
        格子点の位置タイプを判定
        
        Args:
            *indices: 格子点のインデックス
            
        Returns:
            位置を表す文字列
        """
        pass
    
    @abstractmethod
    def _assign_equations_1d(self, eq_by_type, i, j=None, k=None):
        """1D格子点における方程式の割り当てを決定"""
        pass

    @abstractmethod
    def _assign_equations_2d(self, eq_by_type, i, j, k=None):
        """2D格子点における方程式の割り当てを決定"""
        pass

    @abstractmethod
    def _assign_equations_3d(self, eq_by_type, i, j, k):
        """3D格子点における方程式の割り当てを決定"""
        pass
    
    def _classify_equations(self, equations: List, *indices) -> Dict[str, Any]:
        """
        方程式を種類別に分類
        
        Args:
            equations: 方程式のリスト
            *indices: 格子点のインデックス
        
        Returns:
            方程式タイプをキーとする辞書
        """
        # 次元に応じた適切な分類メソッドを呼び出す
        if self.dimension == 1:
            return self._classify_equations_1d(equations, indices[0])
        elif self.dimension == 2:
            return self._classify_equations_2d(equations, indices[0], indices[1])
        else:  # dimension == 3
            return self._classify_equations_3d(equations, indices[0], indices[1], indices[2])
    
    def _classify_equations_1d(self, equations, i):
        """1D方程式の分類"""
        eq_by_type = {
            "governing": None, 
            "dirichlet": None, 
            "neumann": None, 
            "auxiliary": []
        }
        
        for eq in equations:
            eq_type = self._identify_equation_type(eq, i)
            if eq_type == "auxiliary":
                eq_by_type["auxiliary"].append(eq)
            elif eq_type:  # Noneでない場合
                eq_by_type[eq_type] = eq
        
        return eq_by_type
    
    def _classify_equations_2d(self, equations, i, j):
        """2D方程式の分類"""
        eq_by_type = {
            "governing": None, 
            "dirichlet": None, 
            "neumann_x": None, 
            "neumann_y": None, 
            "auxiliary": []
        }
        
        for eq in equations:
            eq_type = self._identify_equation_type(eq, i, j)
            if eq_type == "auxiliary":
                eq_by_type["auxiliary"].append(eq)
            elif eq_type:  # Noneでない場合
                eq_by_type[eq_type] = eq
        
        return eq_by_type
    
    def _classify_equations_3d(self, equations, i, j, k):
        """3D方程式の分類"""
        eq_by_type = {
            "governing": None, 
            "dirichlet": None, 
            "neumann_x": None, 
            "neumann_y": None, 
            "neumann_z": None,
            "auxiliary": []
        }
        
        for eq in equations:
            eq_type = self._identify_equation_type(eq, i, j, k)
            if eq_type == "auxiliary":
                eq_by_type["auxiliary"].append(eq)
            elif eq_type:  # Noneでない場合
                eq_by_type[eq_type] = eq
        
        return eq_by_type
    
    def _identify_equation_type(self, equation, *indices) -> Optional[str]:
        """
        方程式の種類を識別
        
        Args:
            equation: 対象の方程式
            *indices: 格子点のインデックス
            
        Returns:
            方程式の種類または None (無効な場合)
        """
        # 方程式が有効かチェック
        if not equation.is_valid_at(*indices):
            return None
        
        # 方程式自身に種類を問い合わせる
        if hasattr(equation, 'get_equation_type'):
            return equation.get_equation_type()
        
        # フォールバック
        return "auxiliary"
    
    def _validate_equations(self) -> None:
        """方程式セットの基本的な検証"""
        for region, eqs in self.equations.items():
            if not eqs:
                raise ValueError(f"{region}領域に方程式が設定されていません")
    
    def _to_numpy(self, value):
        """
        CuPy配列をNumPy配列に変換する (必要な場合のみ)
        
        Args:
            value: 変換する値
            
        Returns:
            NumPy配列またはスカラー
        """
        if hasattr(value, 'get'):
            return value.get()
        return value
        
    def analyze_sparsity(self) -> Dict[str, Any]:
        """
        行列の疎性を分析
        
        Returns:
            疎性分析結果の辞書
        """
        A = self.build_matrix_system()
        
        total_size = A.shape[0]
        nnz = A.nnz
        max_possible_nnz = total_size * total_size
        sparsity = 1.0 - (nnz / max_possible_nnz)
        
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8 bytes per double
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 8 bytes for value + 4 bytes for indices
        
        return {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }
    
    def analyze_matrix_memory(self):
        """
        行列構築のメモリ効率を分析
        
        Returns:
            メモリ使用情報の辞書
        """
        system_size = self._calculate_system_size()
        
        # CSR行列を構築（ベンチマーク用）
        A = self.build_matrix_system()
        nnz = A.nnz
        
        # メモリ使用量の推定
        memory_dense_MB = (system_size * system_size * 8) / (1024 * 1024)  # 密行列 (8バイト/要素)
        memory_csr_MB = (nnz * 12) / (1024 * 1024)  # CSR形式 (値8バイト + インデックス4バイト)
        memory_lil_MB = (nnz * 16) / (1024 * 1024)  # LIL形式 (値8バイト + インデックス8バイト)
        
        # 中間処理の一時メモリ（リスト形式時）
        memory_lists_MB = (nnz * 20) / (1024 * 1024)  # 値8バイト + 2つのインデックス各4バイト + Pythonオーバーヘッド4バイト
        
        print("\n行列メモリ使用量の分析:")
        print(f"  行列サイズ: {system_size} x {system_size}")
        print(f"  非ゼロ要素数: {nnz}")
        print(f"  疎密比率: {nnz / (system_size * system_size):.6f}")
        print(f"  密行列形式使用量: {memory_dense_MB:.2f} MB")
        print(f"  CSR形式使用量: {memory_csr_MB:.2f} MB")
        print(f"  LIL形式使用量: {memory_lil_MB:.2f} MB")
        print(f"  リスト形式(一時記憶)使用量: {memory_lists_MB:.2f} MB")
        print(f"  メモリ節約量(リスト→LIL): {memory_lists_MB - memory_lil_MB:.2f} MB")
        
        return {
            "matrix_size": system_size,
            "non_zeros": nnz,
            "density": nnz / (system_size * system_size),
            "memory_dense_MB": memory_dense_MB,
            "memory_csr_MB": memory_csr_MB,
            "memory_lil_MB": memory_lil_MB,
            "memory_lists_MB": memory_lists_MB,
            "memory_savings_MB": memory_lists_MB - memory_lil_MB
        }