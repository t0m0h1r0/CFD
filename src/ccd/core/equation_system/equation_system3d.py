"""
3次元方程式システムの定義モジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
3次元問題の格子点方程式を管理し、線形方程式系の行列を構築するための機能を提供します。
"""

import numpy as np
import scipy.sparse as sp_cpu
from typing import Dict, List, Any

from core.base.base_equation_system import BaseEquationSystem

class EquationSystem3D(BaseEquationSystem):
    """3次元方程式システムを管理するクラス"""

    def _initialize_equations(self):
        """3D用の方程式コレクションを初期化"""
        # 領域ごとの方程式コレクション
        self.equations = {
            'interior': [],       # 内部点
            
            # 面の領域
            'face_x_min': [],     # x最小面 (i=0)
            'face_x_max': [],     # x最大面 (i=nx-1)
            'face_y_min': [],     # y最小面 (j=0)
            'face_y_max': [],     # y最大面 (j=ny-1)
            'face_z_min': [],     # z最小面 (k=0)
            'face_z_max': [],     # z最大面 (k=nz-1)
            
            # 辺の領域（x方向辺）
            'edge_x_y_min_z_min': [],    # (0<i<nx-1, j=0, k=0)
            'edge_x_y_min_z_max': [],    # (0<i<nx-1, j=0, k=nz-1)
            'edge_x_y_max_z_min': [],    # (0<i<nx-1, j=ny-1, k=0)
            'edge_x_y_max_z_max': [],    # (0<i<nx-1, j=ny-1, k=nz-1)
            
            # 辺の領域（y方向辺）
            'edge_y_x_min_z_min': [],    # (i=0, 0<j<ny-1, k=0)
            'edge_y_x_min_z_max': [],    # (i=0, 0<j<ny-1, k=nz-1)
            'edge_y_x_max_z_min': [],    # (i=nx-1, 0<j<ny-1, k=0)
            'edge_y_x_max_z_max': [],    # (i=nx-1, 0<j<ny-1, k=nz-1)
            
            # 辺の領域（z方向辺）
            'edge_z_x_min_y_min': [],    # (i=0, j=0, 0<k<nz-1)
            'edge_z_x_min_y_max': [],    # (i=0, j=ny-1, 0<k<nz-1)
            'edge_z_x_max_y_min': [],    # (i=nx-1, j=0, 0<k<nz-1)
            'edge_z_x_max_y_max': [],    # (i=nx-1, j=ny-1, 0<k<nz-1)
            
            # 頂点領域
            'vertex_x_min_y_min_z_min': [],  # (i=0, j=0, k=0)
            'vertex_x_min_y_min_z_max': [],  # (i=0, j=0, k=nz-1)
            'vertex_x_min_y_max_z_min': [],  # (i=0, j=ny-1, k=0)
            'vertex_x_min_y_max_z_max': [],  # (i=0, j=ny-1, k=nz-1)
            'vertex_x_max_y_min_z_min': [],  # (i=nx-1, j=0, k=0)
            'vertex_x_max_y_min_z_max': [],  # (i=nx-1, j=0, k=nz-1)
            'vertex_x_max_y_max_z_min': [],  # (i=nx-1, j=ny-1, k=0)
            'vertex_x_max_y_max_z_max': [],  # (i=nx-1, j=ny-1, k=nz-1)
        }
    
    def _get_point_location(self, i: int, j: int = None, k: int = None) -> str:
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            位置を表す文字列
        """
        if j is None or k is None:
            raise ValueError("3D格子では j, k インデックスが必要です")
            
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        
        # 内部点
        if 0 < i < nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
            return 'interior'
        
        # 頂点点
        if i == 0 and j == 0 and k == 0:
            return 'vertex_x_min_y_min_z_min'
        elif i == 0 and j == 0 and k == nz - 1:
            return 'vertex_x_min_y_min_z_max'
        elif i == 0 and j == ny - 1 and k == 0:
            return 'vertex_x_min_y_max_z_min'
        elif i == 0 and j == ny - 1 and k == nz - 1:
            return 'vertex_x_min_y_max_z_max'
        elif i == nx - 1 and j == 0 and k == 0:
            return 'vertex_x_max_y_min_z_min'
        elif i == nx - 1 and j == 0 and k == nz - 1:
            return 'vertex_x_max_y_min_z_max'
        elif i == nx - 1 and j == ny - 1 and k == 0:
            return 'vertex_x_max_y_max_z_min'
        elif i == nx - 1 and j == ny - 1 and k == nz - 1:
            return 'vertex_x_max_y_max_z_max'
        
        # 辺領域（z方向辺）
        if i == 0 and j == 0 and 0 < k < nz - 1:
            return 'edge_z_x_min_y_min'
        elif i == 0 and j == ny - 1 and 0 < k < nz - 1:
            return 'edge_z_x_min_y_max'
        elif i == nx - 1 and j == 0 and 0 < k < nz - 1:
            return 'edge_z_x_max_y_min'
        elif i == nx - 1 and j == ny - 1 and 0 < k < nz - 1:
            return 'edge_z_x_max_y_max'
            
        # 辺領域（y方向辺）
        if i == 0 and 0 < j < ny - 1 and k == 0:
            return 'edge_y_x_min_z_min'
        elif i == 0 and 0 < j < ny - 1 and k == nz - 1:
            return 'edge_y_x_min_z_max'
        elif i == nx - 1 and 0 < j < ny - 1 and k == 0:
            return 'edge_y_x_max_z_min'
        elif i == nx - 1 and 0 < j < ny - 1 and k == nz - 1:
            return 'edge_y_x_max_z_max'
            
        # 辺領域（x方向辺）
        if 0 < i < nx - 1 and j == 0 and k == 0:
            return 'edge_x_y_min_z_min'
        elif 0 < i < nx - 1 and j == 0 and k == nz - 1:
            return 'edge_x_y_min_z_max'
        elif 0 < i < nx - 1 and j == ny - 1 and k == 0:
            return 'edge_x_y_max_z_min'
        elif 0 < i < nx - 1 and j == ny - 1 and k == nz - 1:
            return 'edge_x_y_max_z_max'
        
        # 面領域
        if i == 0 and 0 < j < ny - 1 and 0 < k < nz - 1:
            return 'face_x_min'
        elif i == nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1:
            return 'face_x_max'
        elif 0 < i < nx - 1 and j == 0 and 0 < k < nz - 1:
            return 'face_y_min'
        elif 0 < i < nx - 1 and j == ny - 1 and 0 < k < nz - 1:
            return 'face_y_max'
        elif 0 < i < nx - 1 and 0 < j < ny - 1 and k == 0:
            return 'face_z_min'
        elif 0 < i < nx - 1 and 0 < j < ny - 1 and k == nz - 1:
            return 'face_z_max'
        
        # 念のため
        return 'interior'
    
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
        assign_with_fallback(7, dirichlet if dirichlet and dirichlet != assignments[1] and dirichlet != assignments[4] else None, True)
        
        # 8行目: z方向のノイマン境界または補助方程式 (ψ_zz)
        assign_with_fallback(8, neumann_z, True, dirichlet)
        
        # 9行目: 補助方程式 (ψ_zzz)
        assign_with_fallback(9, None, True, neumann_z or dirichlet)
        
        return assignments
    
    def build_matrix_system(self):
        """
        3D用の行列システムを構築
        
        Returns:
            システム行列 (CPU CSR形式)
        """
        # 簡易的な検証
        self._validate_equations()
        
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
                    base_idx = (k * ny * nx + j * nx + i) * var_per_point
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
                    
                    # 各種方程式の存在確認
                    if not eq_by_type["governing"]:
                        raise ValueError(f"点 ({i}, {j}, {k}) に支配方程式が設定されていません")
                    
                    # 方程式の割り当て
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
                                col_base = (nk * ny * nx + nj * nx + ni) * var_per_point
                                for l, coeff in enumerate(coeffs):
                                    if coeff != 0.0:  # 非ゼロ要素のみ追加
                                        row_indices.append(row)
                                        col_indices.append(col_base + l)
                                        # CuPy配列があればNumPyに変換
                                        data.append(float(self._to_numpy(coeff)))
        
        # SciPyを使用してCSR行列を構築
        A = sp_cpu.csr_matrix(
            (np.array(data), (np.array(row_indices), np.array(col_indices))), 
            shape=(system_size, system_size)
        )
        
        return A
    
    def _identify_equation_type(self, equation, i: int, j: int, k: int) -> str:
        """
        方程式の種類を識別
        
        Args:
            equation: 対象の方程式
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            方程式の種類 ('governing', 'dirichlet', 等) または None (無効な場合)
        """
        # 方程式が有効かチェック
        if not equation.is_valid_at(i, j, k):
            return None
        
        # クラス名から方程式種別を判定 (importはローカルに行う)
        from equation.dim3.poisson import PoissonEquation3D
        from equation.dim3.original import OriginalEquation3D
        from equation.dim3.boundary import DirichletBoundaryEquation3D
        
        # 3D共通の方程式タイプ
        if isinstance(equation, (PoissonEquation3D, OriginalEquation3D)):
            return "governing"
        elif isinstance(equation, DirichletBoundaryEquation3D):
            return "dirichlet"
        
        # 3D固有のタイプまたは方向性のある方程式
        from equation.converter import DirectionalEquation3D
        if isinstance(equation, DirectionalEquation3D):
            # 内部の2D方程式が特定の種類かチェック
            if hasattr(equation, 'equation_2d'):
                # 方向に基づいて適切なノイマンタイプを返す
                if equation.direction == 'x':
                    return "neumann_x"
                elif equation.direction == 'y':
                    return "neumann_y"
                elif equation.direction == 'z':
                    return "neumann_z"
        
        # それ以外は補助方程式
        return "auxiliary"
