import cupy as cp
import cupyx.scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Any

# 必要な方程式タイプを最初にインポート
from equation.poisson import PoissonEquation, PoissonEquation2D
from equation.original import OriginalEquation, OriginalEquation2D
from equation.boundary import (
    DirichletBoundaryEquation, NeumannBoundaryEquation,
    DirichletXBoundaryEquation2D, DirichletYBoundaryEquation2D,
    NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
)

class EquationSystem:
    """1D/2D両対応の方程式システムを管理するクラス"""

    def __init__(self, grid):
        """
        方程式システムを初期化
        
        Args:
            grid: Grid (1D/2D) オブジェクト
        """
        self.grid = grid
        self.is_2d = grid.is_2d
        
        # 基本方程式コレクション（すべての次元で共通）
        self.left_boundary_equations = []
        self.interior_equations = []
        self.right_boundary_equations = []
        
        # 2D特有の方程式コレクション
        if self.is_2d:
            self.bottom_boundary_equations = []
            self.top_boundary_equations = []
            self.corner_equations = {
                'left_bottom': [],
                'right_bottom': [],
                'left_top': [],
                'right_top': []
            }

    def validate_equation_system(self):
        """方程式システムの妥当性を検証"""
        required_equation_sets = [
            (self.left_boundary_equations, "左境界方程式"),
            (self.interior_equations, "内部方程式"),
            (self.right_boundary_equations, "右境界方程式")
        ]
        
        if self.is_2d:
            required_equation_sets.extend([
                (self.bottom_boundary_equations, "下境界方程式"),
                (self.top_boundary_equations, "上境界方程式")
            ])
        
        for equations, description in required_equation_sets:
            if not equations:
                raise ValueError(f"{description}が設定されていません。")

    def _set_grid_to_equation(self, equation):
        """方程式にグリッドを設定（必要な場合）"""
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
        return equation

    def add_left_boundary_equation(self, equation):
        """
        左境界方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        equation = self._set_grid_to_equation(equation)
        self.left_boundary_equations.append(equation)
        
        # 2Dの場合、左側の角にも同じ方程式を追加
        if self.is_2d:
            self.corner_equations['left_bottom'].append(equation)
            self.corner_equations['left_top'].append(equation)

    def add_interior_equation(self, equation):
        """
        内部方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        self.interior_equations.append(self._set_grid_to_equation(equation))

    def add_right_boundary_equation(self, equation):
        """
        右境界方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        equation = self._set_grid_to_equation(equation)
        self.right_boundary_equations.append(equation)
        
        # 2Dの場合、右側の角にも同じ方程式を追加
        if self.is_2d:
            self.corner_equations['right_bottom'].append(equation)
            self.corner_equations['right_top'].append(equation)

    def add_equation(self, equation):
        """
        全ての領域に方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        equation = self._set_grid_to_equation(equation)
        
        # 基本境界に追加
        self.left_boundary_equations.append(equation)
        self.interior_equations.append(equation)
        self.right_boundary_equations.append(equation)
        
        # 2Dの場合、その他の境界と角にも追加
        if self.is_2d:
            self.bottom_boundary_equations.append(equation)
            self.top_boundary_equations.append(equation)
            
            for corner in self.corner_equations.values():
                corner.append(equation)

    # 2D固有のメソッド群
    def add_interior_x_equation(self, equation):
        """内部点のx方向の方程式を追加 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
        
        equation = self._set_grid_to_equation(equation)
        self.interior_equations.append(equation)
        self.bottom_boundary_equations.append(equation)
        self.top_boundary_equations.append(equation)
    
    def add_interior_y_equation(self, equation):
        """内部点のy方向の方程式を追加 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
        
        equation = self._set_grid_to_equation(equation)
        self.interior_equations.append(equation)
        self.left_boundary_equations.append(equation)
        self.right_boundary_equations.append(equation)

    def add_bottom_boundary_equation(self, equation):
        """下境界の方程式を追加 (j=0) (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
        
        equation = self._set_grid_to_equation(equation)
        self.bottom_boundary_equations.append(equation)
        self.corner_equations['left_bottom'].append(equation)
        self.corner_equations['right_bottom'].append(equation)
    
    def add_top_boundary_equation(self, equation):
        """上境界の方程式を追加 (j=ny-1) (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
        
        equation = self._set_grid_to_equation(equation)
        self.top_boundary_equations.append(equation)
        self.corner_equations['left_top'].append(equation)
        self.corner_equations['right_top'].append(equation)

    # グリッド点の位置判定メソッド
    def is_boundary_point(self, i, j=None):
        """境界点かどうかを判定"""
        return self.grid.is_boundary_point(i, j)
        
    def is_interior_point(self, i, j=None):
        """内部点かどうかを判定"""
        return self.grid.is_interior_point(i, j)

    # 2D用の位置判定メソッド
    def _check_2d_grid(self):
        """2Dグリッドチェック用ヘルパーメソッド"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")

    def is_left_boundary(self, i, j):
        """左境界かどうかを判定（角を除く）(2D only)"""
        self._check_2d_grid()
        return i == 0 and 0 < j < self.grid.ny_points - 1
    
    def is_right_boundary(self, i, j):
        """右境界かどうかを判定（角を除く）(2D only)"""
        self._check_2d_grid()
        return i == self.grid.nx_points - 1 and 0 < j < self.grid.ny_points - 1
    
    def is_bottom_boundary(self, i, j):
        """下境界かどうかを判定（角を除く）(2D only)"""
        self._check_2d_grid()
        return j == 0 and 0 < i < self.grid.nx_points - 1
    
    def is_top_boundary(self, i, j):
        """上境界かどうかを判定（角を除く）(2D only)"""
        self._check_2d_grid()
        return j == self.grid.ny_points - 1 and 0 < i < self.grid.nx_points - 1
    
    def is_corner_point(self, i, j, corner_type):
        """
        指定された角かどうかを判定 (2D only)
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            corner_type: 角のタイプ ('left_bottom', 'right_bottom', 'left_top', 'right_top')
            
        Returns:
            bool: 指定された角かどうか
        """
        self._check_2d_grid()
        
        nx, ny = self.grid.nx_points, self.grid.ny_points
        
        if corner_type == 'left_bottom':
            return i == 0 and j == 0
        elif corner_type == 'right_bottom':
            return i == nx - 1 and j == 0
        elif corner_type == 'left_top':
            return i == 0 and j == ny - 1
        elif corner_type == 'right_top':
            return i == nx - 1 and j == ny - 1
        
        return False

    def build_matrix_system(self):
        """
        行列システムを構築
        
        Returns:
            システム行列
        """
        return self._build_matrix_system_2d() if self.is_2d else self._build_matrix_system_1d()

    def _classify_equation(self, equation, i, j=None):
        """
        方程式をタイプごとに分類するヘルパーメソッド
        
        Args:
            equation: 検査する方程式
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス (2Dのみ)
            
        Returns:
            str: 方程式タイプを表す文字列
        """
        if not equation.is_valid_at(i, j) if self.is_2d else not equation.is_valid_at(i):
            return None
            
        # 1D/2D共通の方程式タイプ
        if isinstance(equation, (PoissonEquation, PoissonEquation2D)):
            return "governing"
        elif isinstance(equation, (OriginalEquation, OriginalEquation2D)):
            return "governing"
        elif isinstance(equation, DirichletBoundaryEquation):
            return "dirichlet"
        elif isinstance(equation, NeumannBoundaryEquation):
            return "neumann"
        
        # 2D固有の方程式タイプ
        if self.is_2d:
            if isinstance(equation, DirichletXBoundaryEquation2D):
                return "dirichlet_x"
            elif isinstance(equation, NeumannXBoundaryEquation2D):
                return "neumann_x"
            elif isinstance(equation, DirichletYBoundaryEquation2D):
                return "dirichlet_y"
            elif isinstance(equation, NeumannYBoundaryEquation2D):
                return "neumann_y"
        
        # その他の方程式は補助方程式として扱う
        return "auxiliary"

    def _build_matrix_system_1d(self):
        """1D行列システムの構築"""
        # システムの妥当性を検証
        self.validate_equation_system()
        
        n = self.grid.n_points
        size = 4 * n  # 1Dは従来通り4変数
        
        data = []
        row_indices = []
        col_indices = []

        for i in range(n):
            # 基本インデックス
            base_idx = i * 4
            
            # この点で適用する方程式リスト
            if i == 0:
                applicable_equations = self.left_boundary_equations
            elif i == n - 1:
                applicable_equations = self.right_boundary_equations
            else:
                applicable_equations = self.interior_equations
            
            # 方程式をタイプ別に分類
            categorized_eqs = {
                "governing": None,
                "dirichlet": None,
                "neumann": None,
                "auxiliary": []
            }
            
            # 各方程式をタイプ別に分類
            for eq in applicable_equations:
                eq_type = self._classify_equation(eq, i)
                if eq_type == "auxiliary":
                    categorized_eqs["auxiliary"].append(eq)
                elif eq_type:  # Noneでない場合
                    categorized_eqs[eq_type] = eq
            
            # 方程式が不足している場合の検証
            if not categorized_eqs["governing"]:
                raise ValueError(f"点 {i} に支配方程式が見つかりません。方程式システムを確認してください。")
            
            if not categorized_eqs["dirichlet"] and not categorized_eqs["auxiliary"]:
                raise ValueError(f"点 {i} にψ'の方程式が見つかりません。")
            
            if not categorized_eqs["neumann"] and not categorized_eqs["auxiliary"]:
                raise ValueError(f"点 {i} にψ''の方程式が見つかりません。")
            
            if not categorized_eqs["auxiliary"]:
                raise ValueError(f"点 {i} にψ'''の方程式が見つかりません。")
            
            # 1D用に方程式を特定の順序で配置
            # 0: ψ - 常に支配方程式
            self._add_equation_to_matrix_1d(
                categorized_eqs["governing"], i, base_idx, 0, 
                data, row_indices, col_indices
            )
            
            # 1: ψ' - ディリクレ境界または補助方程式
            eq = categorized_eqs["dirichlet"] if categorized_eqs["dirichlet"] else categorized_eqs["auxiliary"].pop(0)
            self._add_equation_to_matrix_1d(
                eq, i, base_idx, 1, data, row_indices, col_indices
            )
            
            # 2: ψ'' - ノイマン境界または補助方程式
            eq = categorized_eqs["neumann"] if categorized_eqs["neumann"] else categorized_eqs["auxiliary"].pop(0)
            self._add_equation_to_matrix_1d(
                eq, i, base_idx, 2, data, row_indices, col_indices
            )
            
            # 3: ψ''' - 補助方程式
            self._add_equation_to_matrix_1d(
                categorized_eqs["auxiliary"].pop(0), i, base_idx, 3, 
                data, row_indices, col_indices
            )

        # 疎行列を構築
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(size, size)
        )

        return A

    def _build_matrix_system_2d(self):
        """2D行列システムの構築"""
        # システムの妥当性を検証
        self.validate_equation_system()
        
        nx, ny = self.grid.nx_points, self.grid.ny_points
        
        # 7 unknowns per grid point: ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        n_unknowns_per_point = 7
        system_size = n_unknowns_per_point * nx * ny
        
        data = []
        row_indices = []
        col_indices = []
        
        # 全グリッド点に対して
        for j in range(ny):
            for i in range(nx):
                # グリッド点の基本インデックス
                base_idx = (j * nx + i) * n_unknowns_per_point
                
                # この点の位置タイプに基づいて適用する方程式を決定
                applicable_equations = self._get_applicable_equations_2d(i, j)
                
                # 方程式をタイプ別に分類
                categorized_eqs = {
                    "governing": None,
                    "dirichlet_x": None,
                    "neumann_x": None,
                    "dirichlet_y": None,
                    "neumann_y": None,
                    "auxiliary": []
                }
                
                # 各方程式をタイプ別に分類
                for eq in applicable_equations:
                    eq_type = self._classify_equation(eq, i, j)
                    if eq_type == "auxiliary":
                        categorized_eqs["auxiliary"].append(eq)
                    elif eq_type:  # Noneでない場合
                        categorized_eqs[eq_type] = eq
                
                # 方程式が見つからない場合は例外
                if not categorized_eqs.get("governing"):
                    raise ValueError(f"点 ({i}, {j}) に支配方程式が見つかりません。方程式システムを確認してください。")
                
                # 各変数に対応する方程式を配置
                # 0: ψ - 常に支配方程式
                self._add_equation_to_matrix_2d(
                    categorized_eqs["governing"], i, j, base_idx, 0, 
                    data, row_indices, col_indices
                )
                
                # 1: ψ_x - x方向のディリクレ境界または補助方程式
                eq = categorized_eqs.get("dirichlet_x") or categorized_eqs["auxiliary"].pop(0)
                self._add_equation_to_matrix_2d(
                    eq, i, j, base_idx, 1, data, row_indices, col_indices
                )
                
                # 2: ψ_xx - x方向のノイマン境界または補助方程式
                eq = categorized_eqs.get("neumann_x") or categorized_eqs["auxiliary"].pop(0)
                self._add_equation_to_matrix_2d(
                    eq, i, j, base_idx, 2, data, row_indices, col_indices
                )
                
                # 3: ψ_xxx - 補助方程式
                self._add_equation_to_matrix_2d(
                    categorized_eqs["auxiliary"].pop(0), i, j, base_idx, 3, 
                    data, row_indices, col_indices
                )
                
                # 4: ψ_y - y方向のディリクレ境界または補助方程式
                eq = categorized_eqs.get("dirichlet_y") or categorized_eqs["auxiliary"].pop(0)
                self._add_equation_to_matrix_2d(
                    eq, i, j, base_idx, 4, data, row_indices, col_indices
                )
                
                # 5: ψ_yy - y方向のノイマン境界または補助方程式
                eq = categorized_eqs.get("neumann_y") or categorized_eqs["auxiliary"].pop(0)
                self._add_equation_to_matrix_2d(
                    eq, i, j, base_idx, 5, data, row_indices, col_indices
                )
                
                # 6: ψ_yyy - 補助方程式
                self._add_equation_to_matrix_2d(
                    categorized_eqs["auxiliary"].pop(0), i, j, base_idx, 6, 
                    data, row_indices, col_indices
                )
        
        # 疎行列を作成
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))),
            shape=(system_size, system_size)
        )
        
        return A
        
    def _get_applicable_equations_2d(self, i, j):
        """
        2Dグリッド点の位置に基づいて適用する方程式リストを取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            list: 適用可能な方程式のリスト
        """
        # 内部点
        if self.is_interior_point(i, j):
            return self.interior_equations
        
        # 角点
        for corner_type, equations in self.corner_equations.items():
            if self.is_corner_point(i, j, corner_type):
                return equations
        
        # 境界点（角を除く）
        if self.is_left_boundary(i, j):
            return self.left_boundary_equations
        elif self.is_right_boundary(i, j):
            return self.right_boundary_equations
        elif self.is_bottom_boundary(i, j):
            return self.bottom_boundary_equations
        elif self.is_top_boundary(i, j):
            return self.top_boundary_equations
            
        # 該当しない場合は空リストを返す（通常はここに到達しない）
        return []

    def _add_equation_to_matrix_1d(self, eq, i, base_idx, row_offset, data, row_indices, col_indices):
        """
        特定の行オフセットに1D方程式を追加
        
        Args:
            eq: 追加する方程式
            i: グリッド点インデックス
            base_idx: 現在のグリッド点の基本インデックス
            row_offset: グリッド点内の変数オフセット (0-3)
            data: 行列値のリスト
            row_indices: 行インデックスのリスト
            col_indices: 列インデックスのリスト
        """
        # 行インデックス
        row = base_idx + row_offset
        
        # ステンシル係数を取得
        stencil_coeffs = eq.get_stencil_coefficients(i=i)
        
        # 行列に係数を追加
        n = self.grid.n_points
        for offset, coeffs in stencil_coeffs.items():
            j = i + offset
            if 0 <= j < n:
                col_base = j * 4
                for m, coeff in enumerate(coeffs):
                    if coeff != 0.0:
                        row_indices.append(row)
                        col_indices.append(col_base + m)
                        data.append(float(coeff))
        
    def _add_equation_to_matrix_2d(self, eq, i, j, base_idx, row_offset, data, row_indices, col_indices):
        """
        特定の行オフセットに2D方程式を追加
        
        Args:
            eq: 追加する方程式
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            base_idx: 現在のグリッド点の基本インデックス
            row_offset: グリッド点内の変数オフセット (0-6)
            data: 行列値のリスト
            row_indices: 行インデックスのリスト
            col_indices: 列インデックスのリスト
        """
        # 行インデックス
        row = base_idx + row_offset
        
        # ステンシル係数を取得
        stencil_coeffs = eq.get_stencil_coefficients(i=i, j=j)
        
        # 行列に係数を追加
        nx, ny = self.grid.nx_points, self.grid.ny_points
        for (di, dj), coeffs in stencil_coeffs.items():
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                col_base = (nj * nx + ni) * 7
                for m, coeff in enumerate(coeffs):
                    if coeff != 0.0:
                        row_indices.append(row)
                        col_indices.append(col_base + m)
                        data.append(float(coeff))

    def analyze_sparsity(self):
        """
        行列の疎性を分析
        
        Returns:
            Dict: 疎性分析結果の辞書
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