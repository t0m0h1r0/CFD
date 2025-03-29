"""
3次元方程式システムの定義モジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
3次元問題の格子点方程式を管理し、線形方程式系の行列を構築するための機能を提供します。
"""

import scipy.sparse as sp_cpu
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

from core.base.base_equation_system import BaseEquationSystem


class EquationSystem3D(BaseEquationSystem):
    """3次元方程式システムを管理するクラス"""
    
    # 3D格子の領域タイプ定義
    REGION_TYPES = {
        # 内部
        'interior': {'desc': '内部点'},
        
        # 面領域
        'face_x_min': {'desc': 'x最小面 (i=0)'},
        'face_x_max': {'desc': 'x最大面 (i=nx-1)'},
        'face_y_min': {'desc': 'y最小面 (j=0)'},
        'face_y_max': {'desc': 'y最大面 (j=ny-1)'},
        'face_z_min': {'desc': 'z最小面 (k=0)'},
        'face_z_max': {'desc': 'z最大面 (k=nz-1)'},
        
        # 辺領域（x方向辺）
        'edge_x_y_min_z_min': {'desc': '(0<i<nx-1, j=0, k=0)'},
        'edge_x_y_min_z_max': {'desc': '(0<i<nx-1, j=0, k=nz-1)'},
        'edge_x_y_max_z_min': {'desc': '(0<i<nx-1, j=ny-1, k=0)'},
        'edge_x_y_max_z_max': {'desc': '(0<i<nx-1, j=ny-1, k=nz-1)'},
        
        # 辺領域（y方向辺）
        'edge_y_x_min_z_min': {'desc': '(i=0, 0<j<ny-1, k=0)'},
        'edge_y_x_min_z_max': {'desc': '(i=0, 0<j<ny-1, k=nz-1)'},
        'edge_y_x_max_z_min': {'desc': '(i=nx-1, 0<j<ny-1, k=0)'},
        'edge_y_x_max_z_max': {'desc': '(i=nx-1, 0<j<ny-1, k=nz-1)'},
        
        # 辺領域（z方向辺）
        'edge_z_x_min_y_min': {'desc': '(i=0, j=0, 0<k<nz-1)'},
        'edge_z_x_min_y_max': {'desc': '(i=0, j=ny-1, 0<k<nz-1)'},
        'edge_z_x_max_y_min': {'desc': '(i=nx-1, j=0, 0<k<nz-1)'},
        'edge_z_x_max_y_max': {'desc': '(i=nx-1, j=ny-1, 0<k<nz-1)'},
        
        # 頂点領域
        'vertex_x_min_y_min_z_min': {'desc': '(i=0, j=0, k=0)'},
        'vertex_x_min_y_min_z_max': {'desc': '(i=0, j=0, k=nz-1)'},
        'vertex_x_min_y_max_z_min': {'desc': '(i=0, j=ny-1, k=0)'},
        'vertex_x_min_y_max_z_max': {'desc': '(i=0, j=ny-1, k=nz-1)'},
        'vertex_x_max_y_min_z_min': {'desc': '(i=nx-1, j=0, k=0)'},
        'vertex_x_max_y_min_z_max': {'desc': '(i=nx-1, j=0, k=nz-1)'},
        'vertex_x_max_y_max_z_min': {'desc': '(i=nx-1, j=ny-1, k=0)'},
        'vertex_x_max_y_max_z_max': {'desc': '(i=nx-1, j=ny-1, k=nz-1)'},
    }

    def _initialize_equations(self):
        """3D用の方程式コレクションを初期化"""
        # 領域ごとの方程式コレクション
        self.equations = {region_type: [] for region_type in self.REGION_TYPES}
        
        # 位置キャッシュ
        self._location_cache = {}
    
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
        
        # キャッシュをチェック
        cache_key = (i, j, k)
        if cache_key in self._location_cache:
            return self._location_cache[cache_key]
        
        # Grid3Dの境界タイプ判定を利用
        result = self.grid.get_boundary_type(i, j, k)
        self._location_cache[cache_key] = result
        return result
    
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
        auxiliary = eq_by_type["auxiliary"].copy()  # コピーして操作
        
        # デフォルト方程式
        fallback = governing
        
        # 割り当て結果
        assignments = [None] * 10
        
        # 方程式を割り当てるヘルパー関数
        def assign_with_fallback(idx, primary=None, use_auxiliary=False, tertiary=None):
            if primary:
                assignments[idx] = primary
            elif use_auxiliary and auxiliary:
                assignments[idx] = auxiliary.pop(0)
            elif tertiary:
                assignments[idx] = tertiary
            else:
                assignments[idx] = fallback
        
        # 0行目: 支配方程式 (ψ)
        assignments[0] = governing
        
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
        
        # LIL形式で行列を初期化
        A_lil = sp_cpu.lil_matrix((system_size, system_size))
        
        # 方程式の分類結果をキャッシュ
        equation_cache = {}
        
        # 非ゼロ要素カウント (デバッグ用)
        nnz_count = 0
        
        # 各格子点について処理
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    base_idx = (k * ny * nx + j * nx + i) * var_per_point
                    location = self._get_point_location(i, j, k)
                    location_equations = self.equations[location]
                    
                    # 方程式をタイプ別に分類（キャッシュ利用）
                    cache_key = (location, id(location_equations))
                    if cache_key in equation_cache:
                        eq_by_type = equation_cache[cache_key]
                    else:
                        eq_by_type = self._classify_equations_3d(location_equations, i, j, k)
                        equation_cache[cache_key] = eq_by_type
                    
                    # 各種方程式の存在確認
                    if not eq_by_type["governing"]:
                        raise ValueError(f"点 ({i}, {j}, {k}) に支配方程式が設定されていません")
                    
                    # 方程式の割り当て
                    assignments = self._assign_equations_3d(eq_by_type, i, j, k)
                    
                    # 行列構築
                    self._build_matrix_for_point(
                        A_lil, base_idx, assignments, i, j, k, nx, ny, nz, var_per_point
                    )
        
        # LIL行列をCSR形式に変換（計算効率向上）
        A = A_lil.tocsr()
        
        return A
    
    def _build_matrix_for_point(self, A_lil, base_idx, assignments, i, j, k, nx, ny, nz, var_per_point):
        """
        特定の格子点の行列要素を構築
        
        Args:
            A_lil: LIL形式の行列
            base_idx: 基準インデックス
            assignments: 割り当てられた方程式のリスト
            i, j, k: 格子点のインデックス
            nx, ny, nz: 格子点数
            var_per_point: 点あたりの変数数
        """
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
                    for offset, coeff in enumerate(coeffs):
                        if coeff != 0.0:  # 非ゼロ要素のみ追加
                            # LIL形式に直接値を設定
                            A_lil[row, col_base + offset] = float(self._to_numpy(coeff))
    
    def _classify_equations_3d(self, equations, i, j, k):
        """
        方程式を種類別に分類する内部メソッド
        
        Args:
            equations: 方程式のリスト
            i, j, k: 格子点のインデックス
            
        Returns:
            種類別に分類された方程式の辞書
        """
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
        
        # 方程式タイプ判定を効率化するため遅延インポート
        if not hasattr(self, '_equation_types_initialized'):
            # 3D固有の方程式タイプ判定のための準備
            from equation.dim3.poisson import PoissonEquation3D
            from equation.dim3.original import OriginalEquation3D
            from equation.dim3.boundary import DirichletBoundaryEquation3D
            from equation.converter import DirectionalEquation3D
            from equation.dim1.boundary import NeumannBoundaryEquation
            
            # タイプ判定関数を辞書として保持
            self._governing_types = (PoissonEquation3D, OriginalEquation3D)
            self._dirichlet_type = DirichletBoundaryEquation3D
            self._directional_type = DirectionalEquation3D
            self._neumann_type = NeumannBoundaryEquation
            
            self._equation_types_initialized = True
        
        # 3D共通の方程式タイプ判定
        if isinstance(equation, self._governing_types):
            return "governing"
        elif isinstance(equation, self._dirichlet_type):
            return "dirichlet"
        
        # 3D固有の方向性のある方程式の判定
        if isinstance(equation, self._directional_type):
            if hasattr(equation, 'equation_1d') and isinstance(equation.equation_1d, self._neumann_type):
                direction = equation.direction
                if direction == 'x':
                    return "neumann_x"
                elif direction == 'y':
                    return "neumann_y"
                elif direction == 'z':
                    return "neumann_z"
        
        # それ以外は補助方程式
        return "auxiliary"