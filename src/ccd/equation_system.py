"""
方程式システムの定義と管理を行うモジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
格子点の方程式をシステム全体として効率的に管理し、最終的に線形方程式系の
行列を構築するための機能を提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as sp
from typing import Dict, List, Tuple, Any

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
        
        # 領域ごとの方程式コレクション
        self.equations = {
            'interior': [],   # 内部点
            'left': [],       # 左境界
            'right': [],      # 右境界
        }
        
        # 2D用の追加領域
        if self.is_2d:
            self.equations.update({
                'bottom': [],         # 下境界
                'top': [],            # 上境界
                'left_bottom': [],    # 左下角
                'right_bottom': [],   # 右下角
                'left_top': [],       # 左上角
                'right_top': []       # 右上角
            })

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
        行列システムを構築
        
        Returns:
            システム行列 (CSR形式)
        """
        # 簡易的な検証
        self._validate_equations()
        
        # 次元に応じた構築メソッドを呼び出し
        if self.is_2d:
            return self._build_2d_matrix()
        else:
            return self._build_1d_matrix()
    
    def _validate_equations(self) -> None:
        """方程式セットの基本的な検証"""
        for region, eqs in self.equations.items():
            if not eqs:
                raise ValueError(f"{region}領域に方程式が設定されていません")
    
    def _get_point_location(self, i: int, j: int = None) -> str:
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス (2Dのみ)
            
        Returns:
            位置を表す文字列 ('interior', 'left', 等)
        """
        if not self.is_2d:
            # 1Dの場合
            n = self.grid.n_points
            if i == 0:
                return 'left'
            elif i == n - 1:
                return 'right'
            else:
                return 'interior'
        else:
            # 2Dの場合
            nx, ny = self.grid.nx_points, self.grid.ny_points
            
            # 内部点
            if 0 < i < nx - 1 and 0 < j < ny - 1:
                return 'interior'
            
            # 角点
            if i == 0 and j == 0:
                return 'left_bottom'
            elif i == nx - 1 and j == 0:
                return 'right_bottom'
            elif i == 0 and j == ny - 1:
                return 'left_top'
            elif i == nx - 1 and j == ny - 1:
                return 'right_top'
            
            # 境界エッジ（角を除く）
            if i == 0:
                return 'left'
            elif i == nx - 1:
                return 'right'
            elif j == 0:
                return 'bottom'
            elif j == ny - 1:
                return 'top'
            
            # 念のため
            return 'interior'
    
    def _identify_equation_type(self, equation, i: int, j: int = None) -> str:
        """
        方程式の種類を識別
        
        Args:
            equation: 対象の方程式
            i: x方向のインデックス
            j: y方向のインデックス (2Dのみ)
            
        Returns:
            方程式の種類 ('govering', 'dirichlet', 等) または None (無効な場合)
        """
        # 方程式が有効かチェック
        if self.is_2d:
            if not equation.is_valid_at(i, j):
                return None
        else:
            if not equation.is_valid_at(i):
                return None
        
        # クラス名から方程式種別を判定 (importはローカルに行う)
        from equation.poisson import PoissonEquation, PoissonEquation2D
        from equation.original import OriginalEquation, OriginalEquation2D
        from equation.boundary import (
            DirichletBoundaryEquation, NeumannBoundaryEquation,
            DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
        )
        
        # 1D/2D共通の方程式タイプ
        if isinstance(equation, (PoissonEquation, PoissonEquation2D, OriginalEquation, OriginalEquation2D)):
            return "governing"
        elif isinstance(equation, (DirichletBoundaryEquation, DirichletBoundaryEquation2D)):
            return "dirichlet"
        elif isinstance(equation, NeumannBoundaryEquation):
            return "neumann"
        
        # 2D固有のタイプ
        if self.is_2d:
            if isinstance(equation, NeumannXBoundaryEquation2D):
                return "neumann_x"
            elif isinstance(equation, NeumannYBoundaryEquation2D):
                return "neumann_y"
        
        # それ以外は補助方程式
        return "auxiliary"
    
    def _build_1d_matrix(self):
        """1D用の行列システムを構築"""
        n = self.grid.n_points
        var_per_point = 4  # 1点あたりの変数数 [ψ, ψ', ψ'', ψ''']
        system_size = n * var_per_point
        
        # 行列要素の蓄積用
        data = []
        row_indices = []
        col_indices = []
        
        # 各格子点について処理
        for i in range(n):
            base_idx = i * var_per_point
            location = self._get_point_location(i)
            location_equations = self.equations[location]
            
            # 方程式を種類別に分類
            eq_by_type = {"governing": None, "dirichlet": None, "neumann": None, "auxiliary": []}
            for eq in location_equations:
                eq_type = self._identify_equation_type(eq, i)
                if eq_type == "auxiliary":
                    eq_by_type["auxiliary"].append(eq)
                elif eq_type:  # Noneでない場合
                    eq_by_type[eq_type] = eq
            
            # 各種方程式の存在確認
            if not eq_by_type["governing"]:
                raise ValueError(f"点 {i} に支配方程式が設定されていません")
            
            # 各行に適切な方程式を配置
            assignments = [None] * var_per_point  # 各行の方程式
            
            # 0行目: 支配方程式 (ψ)
            assignments[0] = eq_by_type["governing"]
            
            # 1行目: ディリクレ境界または最初の補助方程式 (ψ')
            if eq_by_type["dirichlet"]:
                assignments[1] = eq_by_type["dirichlet"]
            elif eq_by_type["auxiliary"]:
                assignments[1] = eq_by_type["auxiliary"].pop(0)
            else:
                raise ValueError(f"点 {i} の1行目に設定する方程式がありません")
            
            # 2行目: ノイマン境界または次の補助方程式 (ψ'')
            if eq_by_type["neumann"]:
                assignments[2] = eq_by_type["neumann"]
            elif eq_by_type["auxiliary"]:
                assignments[2] = eq_by_type["auxiliary"].pop(0)
            else:
                raise ValueError(f"点 {i} の2行目に設定する方程式がありません")
            
            # 3行目: 残りの補助方程式 (ψ''')
            if eq_by_type["auxiliary"]:
                assignments[3] = eq_by_type["auxiliary"].pop(0)
            else:
                raise ValueError(f"点 {i} の3行目に設定する方程式がありません")
            
            # 各行の方程式から係数を行列に追加
            for row_offset, eq in enumerate(assignments):
                row = base_idx + row_offset
                
                # ステンシル係数を取得して行列に追加
                stencil_coeffs = eq.get_stencil_coefficients(i=i)
                for offset, coeffs in stencil_coeffs.items():
                    j = i + offset
                    if 0 <= j < n:  # グリッド範囲内チェック
                        col_base = j * var_per_point
                        for k, coeff in enumerate(coeffs):
                            if coeff != 0.0:  # 非ゼロ要素のみ追加
                                row_indices.append(row)
                                col_indices.append(col_base + k)
                                data.append(float(coeff))
        
        # CSR形式の疎行列を構築
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(system_size, system_size)
        )
        
        return A
    
    def _build_2d_matrix(self):
        """2D用の行列システムを構築"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        var_per_point = 7  # 1点あたりの変数数 [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        system_size = nx * ny * var_per_point
        
        # 行列要素の蓄積用
        data = []
        row_indices = []
        col_indices = []
        
        # 各格子点について処理
        for j in range(ny):
            for i in range(nx):
                base_idx = (j * nx + i) * var_per_point
                location = self._get_point_location(i, j)
                location_equations = self.equations[location]
                
                # 方程式を種類別に分類
                eq_by_type = {
                    "governing": None, 
                    "dirichlet": None, 
                    "neumann_x": None, 
                    "neumann_y": None, 
                    "auxiliary": []
                }
                
                for eq in location_equations:
                    eq_type = self._identify_equation_type(eq, i, j)
                    if eq_type == "auxiliary":
                        eq_by_type["auxiliary"].append(eq)
                    elif eq_type:  # Noneでない場合
                        eq_by_type[eq_type] = eq
                
                # 各種方程式の存在確認
                if not eq_by_type["governing"]:
                    raise ValueError(f"点 ({i}, {j}) に支配方程式が設定されていません")
                
                # 方程式の割り当て
                assignments = self._assign_equations_2d(eq_by_type, i, j)
                
                # 各行の方程式から係数を行列に追加
                for row_offset, eq in enumerate(assignments):
                    if eq is None:
                        raise ValueError(f"点 ({i}, {j}) の {row_offset} 行目に方程式が割り当てられていません")
                    
                    row = base_idx + row_offset
                    
                    # ステンシル係数を取得して行列に追加
                    stencil_coeffs = eq.get_stencil_coefficients(i=i, j=j)
                    for (di, dj), coeffs in stencil_coeffs.items():
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < ny:  # グリッド範囲内チェック
                            col_base = (nj * nx + ni) * var_per_point
                            for k, coeff in enumerate(coeffs):
                                if coeff != 0.0:  # 非ゼロ要素のみ追加
                                    row_indices.append(row)
                                    col_indices.append(col_base + k)
                                    data.append(float(coeff))
        
        # CSR形式の疎行列を構築
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(system_size, system_size)
        )
        
        return A
    
    def _assign_equations_2d(self, eq_by_type: Dict[str, Any], i: int, j: int) -> List:
        """
        2D格子点における方程式の割り当てを決定
        
        Args:
            eq_by_type: 種類別に分類された方程式
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            各行に割り当てる方程式のリスト [0行目, 1行目, ..., 6行目]
        """
        governing = eq_by_type["governing"]
        dirichlet = eq_by_type["dirichlet"]
        neumann_x = eq_by_type["neumann_x"]
        neumann_y = eq_by_type["neumann_y"]
        auxiliary = eq_by_type["auxiliary"]
        
        # 方程式の不足時に使うデフォルト方程式
        fallback = governing
        
        # 割り当て結果
        assignments = [None] * 7
        
        # 0行目: 支配方程式 (ψ)
        assignments[0] = governing
        
        # 方程式を割り当てる関数 (不足時はデフォルト)
        def assign_with_fallback(idx, primary, secondary=None, tertiary=None):
            if primary:
                assignments[idx] = primary
            elif secondary and auxiliary:
                assignments[idx] = auxiliary.pop(0)
            elif tertiary:
                assignments[idx] = tertiary
            else:
                assignments[idx] = fallback
        
        # 1行目: x方向のディリクレ境界または補助方程式 (ψ_x)
        assign_with_fallback(1, dirichlet, True)
        
        # 2行目: x方向のノイマン境界または補助方程式 (ψ_xx)
        assign_with_fallback(2, neumann_x, True, dirichlet)
        
        # 3行目: 補助方程式 (ψ_xxx)
        assign_with_fallback(3, None, True, neumann_x or dirichlet)
        
        # 4行目: y方向のディリクレ境界または補助方程式 (ψ_y)
        assign_with_fallback(4, dirichlet if dirichlet and dirichlet != assignments[1] else None, True)
        
        # 5行目: y方向のノイマン境界または補助方程式 (ψ_yy)
        assign_with_fallback(5, neumann_y, True, dirichlet)
        
        # 6行目: 補助方程式 (ψ_yyy)
        assign_with_fallback(6, None, True, neumann_y or dirichlet)
        
        return assignments
    
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