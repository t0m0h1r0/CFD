"""
方程式システムの定義と管理を行うモジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
格子点の方程式をシステム全体として効率的に管理し、最終的に線形方程式系の
行列を構築するための機能を提供します。
"""

import numpy as np
import scipy.sparse as sp_cpu
from typing import Dict, List, Any

class EquationSystem:
    """1D/2D/3D対応の方程式システムを管理するクラス"""

    def __init__(self, grid):
        """
        方程式システムを初期化
        
        Args:
            grid: Grid (1D/2D/3D) オブジェクト
        """
        self.grid = grid
        self.is_2d = grid.is_2d
        self.is_3d = grid.is_3d
        
        # 領域ごとの方程式コレクション
        self.equations = {
            'interior': [],   # 内部点
            'left': [],       # 左境界
            'right': [],      # 右境界
        }
        
        # 2D用の追加領域
        if self.is_2d or self.is_3d:
            self.equations.update({
                'bottom': [],         # 下境界
                'top': [],            # 上境界
                'left_bottom': [],    # 左下角
                'right_bottom': [],   # 右下角
                'left_top': [],       # 左上角
                'right_top': []       # 右上角
            })
            
        # 3D用の追加領域
        if self.is_3d:
            self.equations.update({
                'front': [],          # 前面境界
                'back': [],           # 後面境界
                
                # エッジ領域 (12本)
                'left_front': [],     # 左前エッジ
                'left_back': [],      # 左後エッジ
                'right_front': [],    # 右前エッジ
                'right_back': [],     # 右後エッジ
                'bottom_front': [],   # 下前エッジ
                'bottom_back': [],    # 下後エッジ
                'top_front': [],      # 上前エッジ
                'top_back': [],       # 上後エッジ
                'left_bottom': [],    # 左下エッジ
                'left_top': [],       # 左上エッジ
                'right_bottom': [],   # 右下エッジ
                'right_top': [],      # 右上エッジ
                
                # 頂点領域 (8個)
                'left_bottom_front': [],   # 左下前頂点
                'left_bottom_back': [],    # 左下後頂点
                'left_top_front': [],      # 左上前頂点
                'left_top_back': [],       # 左上後頂点
                'right_bottom_front': [],  # 右下前頂点
                'right_bottom_back': [],   # 右下後頂点
                'right_top_front': [],     # 右上前頂点
                'right_top_back': []       # 右上後頂点
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
            システム行列 (CSR形式、CPU)
        """
        # 簡易的な検証
        self._validate_equations()
        
        # 次元に応じた構築メソッドを呼び出し
        if self.is_3d:
            return self._build_3d_matrix()
        elif self.is_2d:
            return self._build_2d_matrix()
        else:
            return self._build_1d_matrix()
    
    def _validate_equations(self) -> None:
        """方程式セットの基本的な検証"""
        for region, eqs in self.equations.items():
            if not eqs:
                raise ValueError(f"{region}領域に方程式が設定されていません")
    
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
        if not self.is_2d and not self.is_3d:
            # 1Dの場合
            n = self.grid.n_points
            if i == 0:
                return 'left'
            elif i == n - 1:
                return 'right'
            else:
                return 'interior'
        elif not self.is_3d:
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
        else:
            # 3Dの場合
            nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
            
            # 内部点
            if 0 < i < nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
                return 'interior'
            
            # 頂点 (8個)
            if i == 0 and j == 0 and k == 0:
                return 'left_bottom_front'
            elif i == nx - 1 and j == 0 and k == 0:
                return 'right_bottom_front'
            elif i == 0 and j == ny - 1 and k == 0:
                return 'left_top_front'
            elif i == nx - 1 and j == ny - 1 and k == 0:
                return 'right_top_front'
            elif i == 0 and j == 0 and k == nz - 1:
                return 'left_bottom_back'
            elif i == nx - 1 and j == 0 and k == nz - 1:
                return 'right_bottom_back'
            elif i == 0 and j == ny - 1 and k == nz - 1:
                return 'left_top_back'
            elif i == nx - 1 and j == ny - 1 and k == nz - 1:
                return 'right_top_back'
            
            # エッジ (12本、頂点を除く)
            if i == 0 and j == 0 and 0 < k < nz - 1:
                return 'left_bottom'
            elif i == nx - 1 and j == 0 and 0 < k < nz - 1:
                return 'right_bottom'
            elif i == 0 and j == ny - 1 and 0 < k < nz - 1:
                return 'left_top'
            elif i == nx - 1 and j == ny - 1 and 0 < k < nz - 1:
                return 'right_top'
            elif i == 0 and 0 < j < ny - 1 and k == 0:
                return 'left_front'
            elif i == nx - 1 and 0 < j < ny - 1 and k == 0:
                return 'right_front'
            elif i == 0 and 0 < j < ny - 1 and k == nz - 1:
                return 'left_back'
            elif i == nx - 1 and 0 < j < ny - 1 and k == nz - 1:
                return 'right_back'
            elif 0 < i < nx - 1 and j == 0 and k == 0:
                return 'bottom_front'
            elif 0 < i < nx - 1 and j == ny - 1 and k == 0:
                return 'top_front'
            elif 0 < i < nx - 1 and j == 0 and k == nz - 1:
                return 'bottom_back'
            elif 0 < i < nx - 1 and j == ny - 1 and k == nz - 1:
                return 'top_back'
            
            # 面 (6面、エッジと頂点を除く)
            if i == 0 and 0 < j < ny - 1 and 0 < k < nz - 1:
                return 'left'
            elif i == nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
                return 'right'
            elif 0 < i < nx - 1 and j == 0 and 0 < k < nz - 1:
                return 'bottom'
            elif 0 < i < nx - 1 and j == ny - 1 and 0 < k < nz - 1:
                return 'top'
            elif 0 < i < nx - 1 and 0 < j < ny - 1 and k == 0:
                return 'front'
            elif 0 < i < nx - 1 and 0 < j < ny - 1 and k == nz - 1:
                return 'back'
            
            # 念のため
            return 'interior'
    
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
        
        # クラス名から方程式種別を判定 (importはローカルに行う)
        from equation.poisson import PoissonEquation, PoissonEquation2D, PoissonEquation3D
        from equation.original import OriginalEquation, OriginalEquation2D, OriginalEquation3D
        from equation.boundary import (
            DirichletBoundaryEquation, NeumannBoundaryEquation,
            DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D,
            DirichletBoundaryEquation3D, NeumannXBoundaryEquation3D, NeumannYBoundaryEquation3D, NeumannZBoundaryEquation3D
        )
        
        # 1D/2D/3D共通の方程式タイプ
        if isinstance(equation, (PoissonEquation, PoissonEquation2D, PoissonEquation3D, 
                               OriginalEquation, OriginalEquation2D, OriginalEquation3D)):
            return "governing"
        elif isinstance(equation, (DirichletBoundaryEquation, DirichletBoundaryEquation2D, DirichletBoundaryEquation3D)):
            return "dirichlet"
        elif isinstance(equation, NeumannBoundaryEquation):
            return "neumann"
        
        # 2D/3D固有のタイプ
        if self.is_2d or self.is_3d:
            if isinstance(equation, NeumannXBoundaryEquation2D) or isinstance(equation, NeumannXBoundaryEquation3D):
                return "neumann_x"
            elif isinstance(equation, NeumannYBoundaryEquation2D) or isinstance(equation, NeumannYBoundaryEquation3D):
                return "neumann_y"
        
        # 3D固有のタイプ
        if self.is_3d:
            if isinstance(equation, NeumannZBoundaryEquation3D):
                return "neumann_z"
        
        # それ以外は補助方程式
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

    # 既存の _build_1d_matrix, _build_2d_matrix メソッドは省略
    
    def _build_3d_matrix(self):
        """3D用の行列システムを構築 (CPU版)"""
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        var_per_point = 10  # 1点あたりの変数数 [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        system_size = nx * ny * nz * var_per_point
        
        # 行列要素の蓄積用 (CPU処理)
        data = []
        row_indices = []
        col_indices = []
        
        # 各格子点について処理
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    base_idx = ((k * ny + j) * nx + i) * var_per_point
                    location = self._get_point_location(i, j, k)
                    location_equations = self.equations[location]
                    
                    # 方程式を種類別に分類
                    eq_by_type = {
                        "governing": None, 
                        "dirichlet": None, 
                        "neumann_x": None, 
                        "neumann_y": None, 
                        "neumann_z": None, 
                        "auxiliary": []
                    }
                    
                    for eq in location_equations:
                        eq_type = self._identify_equation_type(eq, i, j, k)
                        if eq_type == "auxiliary":
                            eq_by_type["auxiliary"].append(eq)
                        elif eq_type:  # Noneでない場合
                            eq_by_type[eq_type] = eq
                    
                    # 各行に割り当てる方程式を決定
                    assignments = self._assign_equations_3d(eq_by_type, i, j, k)
                    
                    # 各行の方程式から係数を行列に追加
                    for row_offset, eq in enumerate(assignments):
                        if eq is None:
                            raise ValueError(f"点 ({i}, {j}, {k}) の {row_offset} 行目に方程式が割り当てられていません")
                        
                        row = base_idx + row_offset
                        
                        # ステンシル係数を取得して行列に追加
                        stencil_coeffs = eq.get_stencil_coefficients(i=i, j=j, k=k)
                        for (di, dj, dk), coeffs in stencil_coeffs.items():
                            ni, nj, nk = i + di, j + dj, k + dk
                            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:  # グリッド範囲内チェック
                                col_base = ((nk * ny + nj) * nx + ni) * var_per_point
                                for coeff_idx, coeff in enumerate(coeffs):
                                    if coeff != 0.0:  # 非ゼロ要素のみ追加
                                        row_indices.append(row)
                                        col_indices.append(col_base + coeff_idx)
                                        # CuPy配列があればNumPyに変換
                                        data.append(float(self._to_numpy(coeff)))
        
        # SciPyを使用してCSR行列を構築
        A = sp_cpu.csr_matrix(
            (np.array(data), (np.array(row_indices), np.array(col_indices))), 
            shape=(system_size, system_size)
        )
        
        return A
    
    def _assign_equations_3d(self, eq_by_type: Dict[str, Any], i: int, j: int, k: int) -> List:
        """
        3D格子点における方程式の割り当てを決定
        
        Args:
            eq_by_type: 種類別に分類された方程式
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            各行に割り当てる方程式のリスト [0行目, 1行目, ..., 9行目]
        """
        governing = eq_by_type["governing"]
        dirichlet = eq_by_type["dirichlet"]
        neumann_x = eq_by_type["neumann_x"]
        neumann_y = eq_by_type["neumann_y"]
        neumann_z = eq_by_type["neumann_z"]
        auxiliary = eq_by_type["auxiliary"]
        
        # 方程式の不足時に使うデフォルト方程式
        fallback = governing
        
        # 割り当て結果
        assignments = [None] * 10
        
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
        
        # 7行目: z方向のディリクレ境界または補助方程式 (ψ_z)
        assign_with_fallback(7, dirichlet if (dirichlet and 
                                          dirichlet != assignments[1] and 
                                          dirichlet != assignments[4]) else None, True)
        
        # 8行目: z方向のノイマン境界または補助方程式 (ψ_zz)
        assign_with_fallback(8, neumann_z, True, dirichlet)
        
        # 9行目: 補助方程式 (ψ_zzz)
        assign_with_fallback(9, None, True, neumann_z or dirichlet)
        
        return assignments