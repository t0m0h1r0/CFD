"""
2次元方程式システムの定義モジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
2次元問題の格子点方程式を管理し、線形方程式系の行列を構築するための機能を提供します。
"""

import numpy as np
import scipy.sparse as sp_cpu
from typing import Dict, List, Any

from core.base.base_equation_system import BaseEquationSystem

class EquationSystem2D(BaseEquationSystem):
    """2次元方程式システムを管理するクラス"""

    def _initialize_equations(self):
        """2D用の方程式コレクションを初期化"""
        # 領域ごとの方程式コレクション
        self.equations = {
            'interior': [],       # 内部点
            'left': [],           # 左境界
            'right': [],          # 右境界
            'bottom': [],         # 下境界
            'top': [],            # 上境界
            'left_bottom': [],    # 左下角
            'right_bottom': [],   # 右下角
            'left_top': [],       # 左上角
            'right_top': []       # 右上角
        }
    
    def _get_point_location(self, i: int, j: int = None) -> str:
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            位置を表す文字列 ('interior', 'left', 'bottom', etc.)
        """
        if j is None:
            raise ValueError("2D格子では j インデックスが必要です")
            
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
    
    def build_matrix_system(self):
        """
        2D用の行列システムを構築
        
        Returns:
            システム行列 (CPU CSR形式)
        """
        # 簡易的な検証
        self._validate_equations()
        
        nx, ny = self.grid.nx_points, self.grid.ny_points
        var_per_point = 7  # 1点あたりの変数数 [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        system_size = nx * ny * var_per_point
        
        # LIL形式で行列を初期化（メモリ効率の良い構築）
        A_lil = sp_cpu.lil_matrix((system_size, system_size))
        
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
                                    # LIL形式に直接値を設定
                                    A_lil[row, col_base + k] = float(self._to_numpy(coeff))
        
        # LIL行列をCSR形式に変換（計算効率向上）
        A = A_lil.tocsr()
        
        return A