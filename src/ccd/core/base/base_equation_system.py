"""
方程式システムの基底クラスを定義するモジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
格子点の方程式をシステム全体として管理するための基底クラスと共通機能を提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

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
        
        # 領域ごとの方程式コレクション（サブクラスで初期化）
        self.equations = {}
        
        # 次元に応じた初期化
        self._initialize_equations()
    
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
    
    @abstractmethod
    def build_matrix_system(self):
        """
        行列システムを構築
        
        Returns:
            システム行列 (CSR形式、CPU)
        """
        pass
    
    def _validate_equations(self) -> None:
        """方程式セットの基本的な検証"""
        for region, eqs in self.equations.items():
            if not eqs:
                raise ValueError(f"{region}領域に方程式が設定されていません")
    
    @abstractmethod
    def _get_point_location(self, i: int, j: int = None, k: int = None) -> str:
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス (2D/3Dのみ)
            k: z方向のインデックス (3Dのみ)
            
        Returns:
            位置を表す文字列 ('interior', 'left', 等)
        """
        pass
    
    def _identify_equation_type(self, equation, i: int, j: int = None, k: int = None) -> str:
        """
        方程式の種類を識別
        
        Args:
            equation: 対象の方程式
            i: x方向のインデックス
            j: y方向のインデックス (2D/3Dのみ)
            k: z方向のインデックス (3Dのみ)
            
        Returns:
            方程式の種類 ('governing', 'dirichlet', 等) または None (無効な場合)
        """
        # 方程式が有効かチェック
        if self.is_3d:
            if not equation.is_valid_at(i, j, k):
                return None
        elif self.is_2d:
            if not equation.is_valid_at(i, j):
                return None
        else:
            if not equation.is_valid_at(i):
                return None
        
        # 方程式自身に種類を問い合わせる
        if hasattr(equation, 'get_equation_type'):
            return equation.get_equation_type()
        
        # もし get_equation_type メソッドが実装されていない場合のフォールバック
        # (後方互換性のため)
        return "auxiliary"
    
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
        # 次元に応じたサイズの計算
        if self.is_3d:
            nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
            var_per_point = 10  # 1点あたりの変数数 [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
            system_size = nx * ny * nz * var_per_point
        elif self.is_2d:
            nx, ny = self.grid.nx_points, self.grid.ny_points
            var_per_point = 7  # 1点あたりの変数数 [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
            system_size = nx * ny * var_per_point
        else:
            n = self.grid.n_points
            var_per_point = 4  # 1点あたりの変数数 [ψ, ψ', ψ'', ψ''']
            system_size = n * var_per_point
        
        # CSR行列を構築（ベンチマーク用）
        A = self.build_matrix_system()
        nnz = A.nnz
        
        # メモリ使用量の推定
        memory_dense_MB = (system_size * system_size * 8) / (1024 * 1024)  # 密行列 (8バイト/要素)
        memory_csr_MB = (nnz * 12) / (1024 * 1024)  # CSR形式 (値8バイト + インデックス4バイト)
        memory_lil_MB = (nnz * 16) / (1024 * 1024)  # LIL形式 (値8バイト + インデックス8バイト)
        
        # 中間処理の一時メモリ（リスト形式時）
        memory_lists_MB = (nnz * 20) / (1024 * 1024)  # 値8バイト + 2つのインデックス各4バイト + Pythonオーバーヘッド4バイト
        
        print(f"\n行列メモリ使用量の分析:")
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