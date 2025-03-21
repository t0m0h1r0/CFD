"""
高精度コンパクト差分法 (CCD) 用の2次元右辺ベクトル構築モジュール

このモジュールは、2次元のポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築するクラスを提供します。
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

from base_rhs_builder import RHSBuilder


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
            right-hand side vector (NumPy配列)
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        var_per_point = 7  # 1点あたりの変数数 [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        system_size = nx * ny * var_per_point
        b = np.zeros(system_size)
        
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
        """
        境界条件の情報を出力
        
        Args:
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: ディリクレ境界値
            left_neumann, right_neumann, bottom_neumann, top_neumann: ノイマン境界値
        """
        if self.enable_dirichlet:
            print(f"[2Dソルバー] ディリクレ境界条件: "
                  f"左={left_dirichlet is not None}, 右={right_dirichlet is not None}, "
                  f"下={bottom_dirichlet is not None}, 上={top_dirichlet is not None}")
        if self.enable_neumann:
            print(f"[2Dソルバー] ノイマン境界条件: "
                  f"左={left_neumann is not None}, 右={right_neumann is not None}, "
                  f"下={bottom_neumann is not None}, 上={top_neumann is not None}")
    
    def _find_governing_row(self, assignments):
        """
        支配方程式が割り当てられた行を見つける
        
        Args:
            assignments: 割り当てられた方程式のリスト
            
        Returns:
            支配方程式の行インデックス
        """
        # 必要なクラスを遅延インポート（依存関係を減らすため）
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
        """
        適切な場所に境界値を設定
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            location: 格子点の位置
            assignments: 割り当てられた方程式リスト
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: ディリクレ境界値
            left_neumann, right_neumann, bottom_neumann, top_neumann: ノイマン境界値
            i, j: 格子点インデックス
        """
        # 必要なクラスを遅延インポート
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
