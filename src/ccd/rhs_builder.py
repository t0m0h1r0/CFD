"""
高精度コンパクト差分法（CCD）用の右辺ベクトル構築モジュール

このモジュールは、ポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築するクラス群を提供します。
"""

import numpy as np
from abc import ABC, abstractmethod


class RHSBuilder(ABC):
    """右辺ベクトルを構築する抽象基底クラス"""
    
    def __init__(self, system, grid, enable_dirichlet=True, enable_neumann=True):
        """
        初期化
        
        Args:
            system: 方程式システム
            grid: グリッドオブジェクト
            enable_dirichlet: ディリクレ境界条件を有効にするフラグ
            enable_neumann: ノイマン境界条件を有効にするフラグ
        """
        self.system = system
        self.grid = grid
        self.enable_dirichlet = enable_dirichlet
        self.enable_neumann = enable_neumann
    
    @abstractmethod
    def build_rhs_vector(self, f_values=None, **boundary_values):
        """
        右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値
            **boundary_values: 境界値の辞書
            
        Returns:
            右辺ベクトル（NumPy配列）
        """
        pass
    
    def _to_numpy(self, arr):
        """CuPy配列をNumPy配列に変換する (必要な場合のみ)"""
        if hasattr(arr, 'get'):
            return arr.get()
        return arr


class RHSBuilder1D(RHSBuilder):
    """1次元問題用の右辺ベクトル構築クラス"""
    
    def build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        left_neumann=None, right_neumann=None, **kwargs):
        """
        1D右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値（全格子点の配列）
            left_dirichlet, right_dirichlet: ディリクレ境界値
            left_neumann, right_neumann: ノイマン境界値
            
        Returns:
            右辺ベクトル (NumPy配列)
        """
        n = self.grid.n_points
        var_per_point = 4  # [ψ, ψ', ψ'', ψ''']
        b = np.zeros(n * var_per_point)
        
        # 入力値を NumPy に変換（必要な場合）
        if f_values is not None:
            f_values = self._to_numpy(f_values)
        
        # 境界条件に関する情報を出力
        boundary_info = []
        if self.enable_dirichlet:
            boundary_info.append(f"ディリクレ境界条件: 左={left_dirichlet}, 右={right_dirichlet}")
        if self.enable_neumann:
            boundary_info.append(f"ノイマン境界条件: 左={left_neumann}, 右={right_neumann}")
        if boundary_info:
            print("[1Dソルバー] " + "; ".join(boundary_info))
        
        # 各格子点に対して処理
        for i in range(n):
            base_idx = i * var_per_point
            location = self.system._get_point_location(i)
            location_equations = self.system.equations[location]
            
            # 方程式を種類別に分類
            eq_by_type = {"governing": None, "dirichlet": None, "neumann": None, "auxiliary": []}
            for eq in location_equations:
                eq_type = self.system._identify_equation_type(eq, i)
                if eq_type == "auxiliary":
                    eq_by_type["auxiliary"].append(eq)
                elif eq_type:  # Noneでない場合
                    eq_by_type[eq_type] = eq
            
            # ソース項の処理（支配方程式に対応する行）
            if eq_by_type["governing"] and f_values is not None:
                b[base_idx] = f_values[i]
            
            # 境界条件の処理
            if location == 'left':
                if self.enable_dirichlet and left_dirichlet is not None:
                    dirichlet_row = self._find_dirichlet_row(base_idx, location_equations, i)
                    if dirichlet_row is not None:
                        b[dirichlet_row] = left_dirichlet
                        
                if self.enable_neumann and left_neumann is not None:
                    neumann_row = self._find_neumann_row(base_idx, location_equations, i)
                    if neumann_row is not None:
                        b[neumann_row] = left_neumann
                        
            elif location == 'right':
                if self.enable_dirichlet and right_dirichlet is not None:
                    dirichlet_row = self._find_dirichlet_row(base_idx, location_equations, i)
                    if dirichlet_row is not None:
                        b[dirichlet_row] = right_dirichlet
                        
                if self.enable_neumann and right_neumann is not None:
                    neumann_row = self._find_neumann_row(base_idx, location_equations, i)
                    if neumann_row is not None:
                        b[neumann_row] = right_neumann
        
        return b
    
    def _find_dirichlet_row(self, base_idx, equations, i):
        """ディリクレ境界条件に対応する行インデックスを見つける"""
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "dirichlet":
                return base_idx + row_offset
        return None
    
    def _find_neumann_row(self, base_idx, equations, i):
        """ノイマン境界条件に対応する行インデックスを見つける"""
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "neumann":
                return base_idx + row_offset
        return None


class RHSBuilder2D(RHSBuilder):
    """2次元問題用の右辺ベクトル構築クラス"""
    
    def build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        bottom_dirichlet=None, top_dirichlet=None, left_neumann=None, 
                        right_neumann=None, bottom_neumann=None, top_neumann=None, **kwargs):
        """
        2D右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値 (nx×ny配列)
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: 境界値
            left_neumann, right_neumann, bottom_neumann, top_neumann: 境界導関数
            
        Returns:
            右辺ベクトル (NumPy配列)
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        var_per_point = 7  # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        b = np.zeros(nx * ny * var_per_point)
        
        # 入力値を NumPy に変換（必要な場合）
        if f_values is not None:
            f_values = self._to_numpy(f_values)
            
        # その他の境界値も同様に変換（必要な場合）
        boundary_values = {
            'left_dirichlet': self._to_numpy(left_dirichlet) if left_dirichlet is not None else None,
            'right_dirichlet': self._to_numpy(right_dirichlet) if right_dirichlet is not None else None,
            'bottom_dirichlet': self._to_numpy(bottom_dirichlet) if bottom_dirichlet is not None else None,
            'top_dirichlet': self._to_numpy(top_dirichlet) if top_dirichlet is not None else None,
            'left_neumann': self._to_numpy(left_neumann) if left_neumann is not None else None,
            'right_neumann': self._to_numpy(right_neumann) if right_neumann is not None else None,
            'bottom_neumann': self._to_numpy(bottom_neumann) if bottom_neumann is not None else None,
            'top_neumann': self._to_numpy(top_neumann) if top_neumann is not None else None
        }
        
        # 境界条件の状態を出力
        self._print_boundary_info(**boundary_values)
        
        # 各格子点に対して処理
        for j in range(ny):
            for i in range(nx):
                base_idx = (j * nx + i) * var_per_point
                location = self.system._get_point_location(i, j)
                location_equations = self.system.equations[location]
                
                # 方程式を種類別に分類
                eq_by_type = {
                    "governing": None, 
                    "dirichlet": None, 
                    "neumann_x": None, 
                    "neumann_y": None, 
                    "auxiliary": []
                }
                
                for eq in location_equations:
                    eq_type = self.system._identify_equation_type(eq, i, j)
                    if eq_type == "auxiliary":
                        eq_by_type["auxiliary"].append(eq)
                    elif eq_type:  # Noneでない場合
                        eq_by_type[eq_type] = eq
                
                # 各行に割り当てる方程式を決定
                assignments = self.system._assign_equations_2d(eq_by_type, i, j)
                
                # ソース項の処理（支配方程式に対応する行）
                governing_row = self._find_governing_row(assignments)
                if governing_row is not None and f_values is not None:
                    b[base_idx + governing_row] = f_values[i, j]
                
                # 境界条件の処理
                self._apply_boundary_values(
                    b, base_idx, location, assignments,
                    **boundary_values,
                    i=i, j=j
                )
        
        return b
    
    def _print_boundary_info(self, left_dirichlet=None, right_dirichlet=None, 
                           bottom_dirichlet=None, top_dirichlet=None,
                           left_neumann=None, right_neumann=None, 
                           bottom_neumann=None, top_neumann=None):
        """境界条件の情報を出力"""
        if self.enable_dirichlet:
            print(f"[2Dソルバー] ディリクレ境界条件: "
                  f"左={left_dirichlet is not None}, 右={right_dirichlet is not None}, "
                  f"下={bottom_dirichlet is not None}, 上={top_dirichlet is not None}")
        if self.enable_neumann:
            print(f"[2Dソルバー] ノイマン境界条件: "
                  f"左={left_neumann is not None}, 右={right_neumann is not None}, "
                  f"下={bottom_neumann is not None}, 上={top_neumann is not None}")
    
    def _find_governing_row(self, assignments):
        """支配方程式が割り当てられた行を見つける"""
        from equation.poisson import PoissonEquation, PoissonEquation2D
        from equation.original import OriginalEquation, OriginalEquation2D
        
        for row, eq in enumerate(assignments):
            if isinstance(eq, (PoissonEquation, PoissonEquation2D, OriginalEquation, OriginalEquation2D)):
                return row
        return None
    
    def _apply_boundary_values(self, b, base_idx, location, assignments,
                            left_dirichlet=None, right_dirichlet=None, 
                            bottom_dirichlet=None, top_dirichlet=None,
                            left_neumann=None, right_neumann=None, 
                            bottom_neumann=None, top_neumann=None,
                            i=None, j=None):
        """適切な場所に境界値を設定"""
        # インポートは関数内で行い、依存関係をローカルに限定
        from equation.boundary import (
            DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
        )
        
        # 境界値の取得
        def get_boundary_value(value, idx):
            if isinstance(value, (list, np.ndarray)) and idx < len(value):
                return value[idx]
            return value
        
        # 各行に割り当てられた方程式をチェック
        for row, eq in enumerate(assignments):
            # ディリクレ条件
            if isinstance(eq, DirichletBoundaryEquation2D) and self.enable_dirichlet:
                if 'left' in location and left_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(left_dirichlet, j)
                elif 'right' in location and right_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(right_dirichlet, j)
                elif 'bottom' in location and bottom_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(bottom_dirichlet, i)
                elif 'top' in location and top_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(top_dirichlet, i)
            
            # X方向ノイマン条件
            elif isinstance(eq, NeumannXBoundaryEquation2D) and self.enable_neumann:
                if 'left' in location and left_neumann is not None:
                    b[base_idx + row] = get_boundary_value(left_neumann, j)
                elif 'right' in location and right_neumann is not None:
                    b[base_idx + row] = get_boundary_value(right_neumann, j)
            
            # Y方向ノイマン条件
            elif isinstance(eq, NeumannYBoundaryEquation2D) and self.enable_neumann:
                if 'bottom' in location and bottom_neumann is not None:
                    b[base_idx + row] = get_boundary_value(bottom_neumann, i)
                elif 'top' in location and top_neumann is not None:
                    b[base_idx + row] = get_boundary_value(top_neumann, i)