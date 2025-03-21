"""
高精度コンパクト差分法 (CCD) 用の1次元右辺ベクトル構築モジュール

このモジュールは、1次元のポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築するクラスを提供します。
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List

from base_rhs_builder import RHSBuilder


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
            right-hand side vector (NumPy配列)
        """
        n = self.grid.n_points
        var_per_point = 4  # 1点あたりの変数数 [ψ, ψ', ψ'', ψ''']
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
        
        # 各格子点について処理
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
            
            # ソース項の処理（支配方程式に対応する行）
            if f_values is not None:
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
        """
        ディリクレ境界条件に対応する行インデックスを見つける
        
        Args:
            base_idx: 基準インデックス
            equations: 方程式リスト
            i: 格子点インデックス
            
        Returns:
            ディリクレ境界条件の行インデックス（見つからない場合はNone）
        """
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "dirichlet":
                return base_idx + row_offset
        return None
    
    def _find_neumann_row(self, base_idx, equations, i):
        """
        ノイマン境界条件に対応する行インデックスを見つける
        
        Args:
            base_idx: 基準インデックス
            equations: 方程式リスト
            i: 格子点インデックス
            
        Returns:
            ノイマン境界条件の行インデックス（見つからない場合はNone）
        """
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "neumann":
                return base_idx + row_offset
        return None
