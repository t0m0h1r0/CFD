"""
高精度コンパクト差分法 (CCD) 用の1次元右辺ベクトル構築モジュール

このモジュールは、1次元のポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築するクラスを提供します。
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

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
        
        # 境界条件の値を辞書にまとめる
        boundary_values = {
            'left_dirichlet': left_dirichlet,
            'right_dirichlet': right_dirichlet,
            'left_neumann': left_neumann,
            'right_neumann': right_neumann
        }
        
        # 境界条件の整合性を検証
        warnings = self._validate_boundary_conditions(boundary_values)
        if warnings:
            for warning in warnings:
                print(f"警告: {warning}")
        
        # 境界条件に関する情報を出力
        self._print_boundary_info(boundary_values)
        
        # 各格子点について処理
        for i in range(n):
            # 点の位置を特定 (内部/左境界/右境界)
            location = self._get_point_location(i)
            base_idx = i * var_per_point
            
            # その位置に対応する方程式群を取得
            location_equations = self.system.equations[location]
            
            # 右辺ベクトルの値をセット
            self._set_rhs_for_point(b, base_idx, location, location_equations, i, f_values, boundary_values)
        
        return b
    
    def _print_boundary_info(self, boundary_values: Dict[str, Any]):
        """
        境界条件に関する情報を出力
        
        Args:
            boundary_values: 境界条件の値の辞書
        """
        boundary_info = []
        if self.enable_dirichlet:
            left_dirichlet = boundary_values.get('left_dirichlet')
            right_dirichlet = boundary_values.get('right_dirichlet')
            boundary_info.append(f"ディリクレ境界条件: 左={left_dirichlet}, 右={right_dirichlet}")
        if self.enable_neumann:
            left_neumann = boundary_values.get('left_neumann')
            right_neumann = boundary_values.get('right_neumann')
            boundary_info.append(f"ノイマン境界条件: 左={left_neumann}, 右={right_neumann}")
        if boundary_info:
            print("[1Dソルバー] " + "; ".join(boundary_info))
    
    def _get_point_location(self, i: int) -> str:
        """
        格子点の位置を取得
        
        Args:
            i: 格子点のインデックス
            
        Returns:
            'interior', 'left', 'right' のいずれか
        """
        n = self.grid.n_points
        if i == 0:
            return 'left'
        elif i == n - 1:
            return 'right'
        else:
            return 'interior'
    
    def _set_rhs_for_point(self, b: np.ndarray, base_idx: int, location: str, 
                          equations: List, i: int, f_values: Optional[np.ndarray], 
                          boundary_values: Dict[str, Any]):
        """
        特定の格子点に対する右辺ベクトルを設定
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            location: 点の位置 ('interior', 'left', 'right')
            equations: その位置に対応する方程式のリスト
            i: 格子点のインデックス
            f_values: ソース項の値
            boundary_values: 境界条件の値の辞書
        """
        # 方程式の種類ごとに整理
        eq_by_type = self._classify_equations(equations, i)
        
        # 各方程式タイプに対応する行のインデックスを決定
        row_assignments = self._assign_equations_to_rows(eq_by_type)
        
        # ソース項の処理（支配方程式に対応する行）
        if f_values is not None:
            governing_row = base_idx + row_assignments.get('governing', 0)
            b[governing_row] = f_values[i]
        
        # 境界条件の処理
        self._apply_boundary_conditions(b, base_idx, location, row_assignments, boundary_values)
    
    def _classify_equations(self, equations: List, i: int) -> Dict[str, Any]:
        """
        方程式をタイプごとに分類
        
        Args:
            equations: 方程式のリスト
            i: 格子点のインデックス
            
        Returns:
            方程式の種類をキー、方程式（または方程式のリスト）を値とする辞書
        """
        eq_by_type = {
            "governing": None, 
            "dirichlet": None, 
            "neumann": None, 
            "auxiliary": []
        }
        
        for eq in equations:
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "auxiliary":
                eq_by_type["auxiliary"].append(eq)
            elif eq_type:  # Noneでない場合
                eq_by_type[eq_type] = eq
        
        return eq_by_type
    
    def _assign_equations_to_rows(self, eq_by_type: Dict[str, Any]) -> Dict[str, int]:
        """
        方程式の種類ごとに行を割り当て
        
        Args:
            eq_by_type: 方程式の種類をキー、方程式（または方程式のリスト）を値とする辞書
            
        Returns:
            方程式の種類をキー、行インデックスを値とする辞書
        """
        # 方程式タイプごとの行割り当て
        row_assignments = {}
        
        # 支配方程式は常に0行目
        if eq_by_type["governing"]:
            row_assignments["governing"] = 0
        
        # ディリクレ境界条件は通常1行目
        if eq_by_type["dirichlet"]:
            row_assignments["dirichlet"] = 1
        
        # ノイマン境界条件は通常2行目
        if eq_by_type["neumann"]:
            row_assignments["neumann"] = 2
        
        # 補助方程式は残りの行に割り当て
        aux_idx = 0
        for idx in range(4):  # 0～3行目
            if idx not in row_assignments.values():
                if aux_idx < len(eq_by_type["auxiliary"]):
                    row_assignments[f"auxiliary_{aux_idx}"] = idx
                    aux_idx += 1
        
        return row_assignments
    
    def _apply_boundary_conditions(self, b: np.ndarray, base_idx: int, location: str, 
                                 row_assignments: Dict[str, int], boundary_values: Dict[str, Any]):
        """
        境界条件を右辺ベクトルに適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            location: 点の位置 ('interior', 'left', 'right')
            row_assignments: 方程式の種類をキー、行インデックスを値とする辞書
            boundary_values: 境界条件の値の辞書
        """
        if location == 'left':
            if self.enable_dirichlet and 'dirichlet' in row_assignments and boundary_values.get('left_dirichlet') is not None:
                b[base_idx + row_assignments['dirichlet']] = boundary_values['left_dirichlet']
                
            if self.enable_neumann and 'neumann' in row_assignments and boundary_values.get('left_neumann') is not None:
                b[base_idx + row_assignments['neumann']] = boundary_values['left_neumann']
                
        elif location == 'right':
            if self.enable_dirichlet and 'dirichlet' in row_assignments and boundary_values.get('right_dirichlet') is not None:
                b[base_idx + row_assignments['dirichlet']] = boundary_values['right_dirichlet']
                
            if self.enable_neumann and 'neumann' in row_assignments and boundary_values.get('right_neumann') is not None:
                b[base_idx + row_assignments['neumann']] = boundary_values['right_neumann']