"""
方程式システムの基底クラスを定義するモジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
格子点の方程式をシステム全体として管理するための基底クラスと共通機能を提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Callable
import numpy as np
import scipy.sparse as sp_cpu


class BaseEquationSystem(ABC):
    """方程式システムの抽象基底クラス"""

    # 次元ごとの設定値マップ
    _DIMENSION_CONFIG = {
        1: {"vars_per_point": 4, "eq_types": ["governing", "dirichlet", "neumann", "auxiliary"]},
        2: {"vars_per_point": 7, "eq_types": ["governing", "dirichlet", "neumann_x", "neumann_y", "auxiliary"]},
        3: {"vars_per_point": 10, "eq_types": ["governing", "dirichlet", "neumann_x", "neumann_y", "neumann_z", "auxiliary"]}
    }

    def __init__(self, grid):
        """
        方程式システムを初期化
        
        Args:
            grid: 計算格子オブジェクト (1D, 2D, または 3D)
        """
        self.grid = grid
        
        # 次元を特定
        self.is_2d = grid.is_2d if hasattr(grid, 'is_2d') else False
        self.is_3d = grid.is_3d if hasattr(grid, 'is_3d') else False
        self.dimension = 3 if self.is_3d else (2 if self.is_2d else 1)
        
        # 次元ごとの設定を適用
        config = self._DIMENSION_CONFIG[self.dimension]
        self.var_per_point = config["vars_per_point"]
        self._eq_type_keys = config["eq_types"]
        
        # グリッドサイズ情報
        self._grid_sizes = self._get_grid_sizes()
        
        # 領域ごとの方程式コレクション（サブクラスで初期化）
        self.equations = {}
        
        # 位置キャッシュ（計算効率化のため）
        self._location_cache = {}
        
        # 次元に応じた方程式割り当て関数をマッピング
        self._assign_funcs = {
            1: self._assign_equations_1d,
            2: self._assign_equations_2d,
            3: self._assign_equations_3d
        }
        
        # 次元に応じた初期化
        self._initialize_equations()
    
    @abstractmethod
    def _initialize_equations(self):
        """次元に応じた方程式コレクションを初期化（サブクラスで実装）"""
        pass
    
    def add_equation(self, region: str, equation) -> None:
        """指定された領域に方程式を追加"""
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        if region in self.equations:
            self.equations[region].append(equation)
        else:
            raise ValueError(f"未知の領域: {region}")
    
    def add_equations(self, region: str, equations: List) -> None:
        """指定された領域に複数の方程式を一度に追加"""
        for equation in equations:
            self.add_equation(region, equation)
    
    def add_dominant_equation(self, equation) -> None:
        """支配方程式をすべての領域に追加"""
        for region in self.equations:
            self.add_equation(region, equation)
    
    def build_matrix_system(self):
        """行列システムを構築（共通実装）"""
        # 方程式セットの検証
        self._validate_equations()
        
        # システムのサイズを計算
        system_size = self._calculate_system_size()
        
        # LIL形式で行列を初期化
        A_lil = sp_cpu.lil_matrix((system_size, system_size))
        
        # 次元に応じた格子点ループ
        if self.dimension == 1:
            self._loop_over_grid_points_1d(lambda i: self._build_matrix_for_point(A_lil, i))
        elif self.dimension == 2:
            self._loop_over_grid_points_2d(lambda i, j: self._build_matrix_for_point(A_lil, i, j))
        else:  # dimension == 3
            self._loop_over_grid_points_3d(lambda i, j, k: self._build_matrix_for_point(A_lil, i, j, k))
            
        # 完成した行列を返す
        return A_lil.tocsr()
    
    def _loop_over_grid_points_1d(self, func: Callable):
        """1D格子点をループしてコールバック関数を実行"""
        nx = self._grid_sizes[0]
        for i in range(nx):
            func(i)
    
    def _loop_over_grid_points_2d(self, func: Callable):
        """2D格子点をループしてコールバック関数を実行"""
        nx, ny = self._grid_sizes
        for j in range(ny):
            for i in range(nx):
                func(i, j)
    
    def _loop_over_grid_points_3d(self, func: Callable):
        """3D格子点をループしてコールバック関数を実行"""
        nx, ny, nz = self._grid_sizes
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    func(i, j, k)
    
    def _build_matrix_for_point(self, A_lil, *indices):
        """特定の格子点に対する行列要素を構築"""
        # 点の位置と基準インデックスを取得
        location = self._get_point_location(*indices)
        base_idx = self._calculate_base_index(indices)
        
        # その位置に対応する方程式群を取得
        location_equations = self.equations[location]
        
        # 方程式を種類別に分類
        eq_by_type = self._classify_equations(location_equations, *indices)
        
        # 支配方程式の存在確認
        if not eq_by_type["governing"]:
            coord_str = ", ".join(str(idx) for idx in indices)
            raise ValueError(f"点 ({coord_str}) に支配方程式が設定されていません")
        
        # 次元に応じた方程式割り当て関数を呼び出す
        assign_func = self._assign_funcs[self.dimension]
        assignments = assign_func(eq_by_type, *indices)
        
        # 各行の方程式から係数を行列に追加
        for row_offset, eq in enumerate(assignments):
            if eq is None:
                coord_str = ", ".join(str(idx) for idx in indices)
                raise ValueError(f"点 ({coord_str}) の {row_offset} 行目に方程式が割り当てられていません")
            
            # ステンシル係数を取得して行列に追加
            self._add_equation_coefficients(A_lil, base_idx + row_offset, eq, indices)
    
    def _add_equation_coefficients(self, A_lil, row, equation, indices):
        """方程式のステンシル係数を行列に追加"""
        stencil_coeffs = equation.get_stencil_coefficients(*indices)
        
        for offset, coeffs in stencil_coeffs.items():
            # 相対オフセットから絶対インデックスと基準インデックスを計算
            neighbor_indices = self._get_neighbor_indices(indices, offset)
            
            # グリッド範囲内チェック
            if self._is_valid_indices(neighbor_indices):
                col_base = self._calculate_base_index(neighbor_indices)
                
                # 非ゼロ係数のみ追加
                for k, coeff in enumerate(coeffs):
                    if coeff != 0.0:
                        A_lil[row, col_base + k] = float(self._to_numpy(coeff))
    
    def _get_neighbor_indices(self, point_indices, offset):
        """インデックスにオフセットを適用して隣接点のインデックスを得る"""
        if self.dimension == 1:
            return (point_indices[0] + offset,)
        elif self.dimension == 2:
            return (point_indices[0] + offset[0], point_indices[1] + offset[1])
        else:  # dimension == 3
            return (point_indices[0] + offset[0], point_indices[1] + offset[1], point_indices[2] + offset[2])
    
    def _is_valid_indices(self, indices):
        """インデックスが有効範囲内かチェック"""
        return all(0 <= idx < size for idx, size in zip(indices, self._grid_sizes))
    
    def _get_grid_sizes(self) -> Tuple[int, ...]:
        """グリッドサイズ情報を返す"""
        if self.dimension == 1:
            return (self.grid.n_points,)
        elif self.dimension == 2:
            return (self.grid.nx_points, self.grid.ny_points)
        else:  # dimension == 3
            return (self.grid.nx_points, self.grid.ny_points, self.grid.nz_points)
    
    def _calculate_system_size(self) -> int:
        """システムの合計サイズを計算"""
        # 格子点数 × 1点あたりの変数数
        total_points = np.prod(self._grid_sizes)
        return total_points * self.var_per_point
    
    def _calculate_base_index(self, indices) -> int:
        """グリッドインデックスから行列の基準インデックスを計算"""
        if self.dimension == 1:
            i = indices[0]
            return i * self.var_per_point
        elif self.dimension == 2:
            i, j = indices
            nx = self._grid_sizes[0]
            return (j * nx + i) * self.var_per_point
        else:  # dimension == 3
            i, j, k = indices
            nx, ny = self._grid_sizes[:2]
            return (k * ny * nx + j * nx + i) * self.var_per_point
    
    def _classify_equations(self, equations: List, *indices) -> Dict[str, Any]:
        """方程式を種類別に分類"""
        # 方程式タイプの辞書を初期化
        eq_by_type = {key: None for key in self._eq_type_keys}
        eq_by_type["auxiliary"] = []  # auxiliary は常にリスト
        
        # 各方程式のタイプを判定
        for eq in equations:
            eq_type = self._identify_equation_type(eq, *indices)
            
            if eq_type == "auxiliary":
                eq_by_type["auxiliary"].append(eq)
            elif eq_type and eq_type in eq_by_type:
                eq_by_type[eq_type] = eq
        
        return eq_by_type
    
    def _identify_equation_type(self, equation, *indices) -> Optional[str]:
        """方程式の種類を識別"""
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
        """CuPy配列をNumPy配列に変換する (必要な場合のみ)"""
        return value.get() if hasattr(value, 'get') else value
    
    def analyze_matrix(self, build_matrix=True):
        """
        行列の解析情報を取得
        
        Args:
            build_matrix: 行列を構築するかどうか
            
        Returns:
            解析情報の辞書
        """
        # 基本情報を計算
        system_size = self._calculate_system_size()
        
        if build_matrix:
            # 行列を構築して詳細情報を取得
            A = self.build_matrix_system()
            nnz = A.nnz
            sparsity = 1.0 - (nnz / (system_size * system_size))
            
            # メモリ使用量の推定
            memory_dense_MB = (system_size * system_size * 8) / (1024 * 1024)
            memory_csr_MB = (nnz * 12) / (1024 * 1024)
            memory_lil_MB = (nnz * 16) / (1024 * 1024)
            memory_lists_MB = (nnz * 20) / (1024 * 1024)
            
            # 結果を辞書にまとめる
            return {
                "matrix_size": system_size,
                "non_zeros": nnz,
                "sparsity": sparsity,
                "memory": {
                    "dense_MB": memory_dense_MB,
                    "csr_MB": memory_csr_MB,
                    "lil_MB": memory_lil_MB,
                    "lists_MB": memory_lists_MB,
                    "savings_MB": memory_lists_MB - memory_lil_MB
                }
            }
        else:
            # 行列を構築せずに基本情報のみ返す
            return {
                "matrix_size": system_size,
                "dimension": self.dimension,
                "grid_sizes": self._grid_sizes,
                "vars_per_point": self.var_per_point
            }
    
    def analyze_sparsity(self) -> Dict[str, Any]:
        """行列の疎性を分析（互換性のため残す）"""
        analysis = self.analyze_matrix()
        return {
            "matrix_size": analysis["matrix_size"],
            "non_zeros": analysis["non_zeros"],
            "sparsity": analysis["sparsity"],
            "memory_dense_MB": analysis["memory"]["dense_MB"],
            "memory_sparse_MB": analysis["memory"]["csr_MB"]
        }
    
    def analyze_matrix_memory(self):
        """行列構築のメモリ効率を分析（互換性のため残す）"""
        analysis = self.analyze_matrix()
        
        # 詳細な出力
        print("\n行列メモリ使用量の分析:")
        print(f"  行列サイズ: {analysis['matrix_size']} x {analysis['matrix_size']}")
        print(f"  非ゼロ要素数: {analysis['non_zeros']}")
        print(f"  疎密比率: {analysis['sparsity']:.6f}")
        print(f"  密行列形式使用量: {analysis['memory']['dense_MB']:.2f} MB")
        print(f"  CSR形式使用量: {analysis['memory']['csr_MB']:.2f} MB")
        print(f"  LIL形式使用量: {analysis['memory']['lil_MB']:.2f} MB")
        print(f"  リスト形式(一時記憶)使用量: {analysis['memory']['lists_MB']:.2f} MB")
        print(f"  メモリ節約量(リスト→LIL): {analysis['memory']['savings_MB']:.2f} MB")
        
        # 旧形式の戻り値を維持（互換性のため）
        return {
            "matrix_size": analysis["matrix_size"],
            "non_zeros": analysis["non_zeros"],
            "density": 1.0 - analysis["sparsity"],
            "memory_dense_MB": analysis["memory"]["dense_MB"],
            "memory_csr_MB": analysis["memory"]["csr_MB"],
            "memory_lil_MB": analysis["memory"]["lil_MB"],
            "memory_lists_MB": analysis["memory"]["lists_MB"],
            "memory_savings_MB": analysis["memory"]["savings_MB"]
        }
    
    # 抽象メソッド（サブクラスで実装）
    @abstractmethod
    def _get_point_location(self, *indices) -> str:
        """格子点の位置タイプを判定"""
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