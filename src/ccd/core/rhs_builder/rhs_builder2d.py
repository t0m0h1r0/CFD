"""
高精度コンパクト差分法 (CCD) 用の2次元右辺ベクトル構築モジュール

このモジュールは、2次元のポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築するクラスを提供します。
"""

import numpy as np
from typing import Dict, Any, Optional, List

from core.base.base_rhs_builder import RHSBuilder


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
            
        # 境界条件値を辞書にまとめる
        boundary_values = {
            'left_dirichlet': self._to_numpy(left_dirichlet),
            'right_dirichlet': self._to_numpy(right_dirichlet),
            'bottom_dirichlet': self._to_numpy(bottom_dirichlet),
            'top_dirichlet': self._to_numpy(top_dirichlet),
            'left_neumann': self._to_numpy(left_neumann),
            'right_neumann': self._to_numpy(right_neumann),
            'bottom_neumann': self._to_numpy(bottom_neumann),
            'top_neumann': self._to_numpy(top_neumann)
        }
        
        # 境界条件の整合性を検証 (基底クラスの共通機能を使用)
        warnings = self._validate_boundary_conditions(boundary_values)
        if warnings:
            for warning in warnings:
                print(f"警告: {warning}")
        
        # 境界条件に関する情報を出力
        self._print_boundary_info(boundary_values)
        
        # 各格子点について処理
        for j in range(ny):
            for i in range(nx):
                base_idx = (j * nx + i) * var_per_point
                
                # 点の位置を特定
                location = self._get_point_location(i, j)
                
                # その位置に対応する方程式群を取得
                location_equations = self.system.equations[location]
                
                # 右辺ベクトルの値をセット
                self._set_rhs_for_point(b, base_idx, location, location_equations, i, j, f_values, boundary_values)
        
        return b
    
    def _print_boundary_info(self, boundary_values: Dict[str, Any]):
        """
        境界条件に関する情報を出力
        
        Args:
            boundary_values: 境界条件の値の辞書
        """
        if self.enable_dirichlet:
            left_dirichlet = boundary_values.get('left_dirichlet') is not None
            right_dirichlet = boundary_values.get('right_dirichlet') is not None
            bottom_dirichlet = boundary_values.get('bottom_dirichlet') is not None
            top_dirichlet = boundary_values.get('top_dirichlet') is not None
            print(f"[2Dソルバー] ディリクレ境界条件: "
                  f"左={left_dirichlet}, 右={right_dirichlet}, "
                  f"下={bottom_dirichlet}, 上={top_dirichlet}")
            
        if self.enable_neumann:
            left_neumann = boundary_values.get('left_neumann') is not None
            right_neumann = boundary_values.get('right_neumann') is not None
            bottom_neumann = boundary_values.get('bottom_neumann') is not None
            top_neumann = boundary_values.get('top_neumann') is not None
            print(f"[2Dソルバー] ノイマン境界条件: "
                  f"左={left_neumann}, 右={right_neumann}, "
                  f"下={bottom_neumann}, 上={top_neumann}")
    
    def _get_point_location(self, i: int, j: int) -> str:
        """
        格子点の位置を取得
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            位置を表す文字列 ('interior', 'left', 'right', 'bottom', 'top', 
                           'left_bottom', 'right_bottom', 'left_top', 'right_top')
        """
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
    
    def _set_rhs_for_point(self, b: np.ndarray, base_idx: int, location: str, 
                          equations: List, i: int, j: int, f_values: Optional[np.ndarray], 
                          boundary_values: Dict[str, Any]):
        """
        特定の格子点に対する右辺ベクトルを設定
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            location: 点の位置
            equations: その位置に対応する方程式のリスト
            i: x方向インデックス
            j: y方向インデックス
            f_values: ソース項の値
            boundary_values: 境界条件の値の辞書
        """
        # 方程式の種類ごとに整理 (基底クラスの共通機能を使用)
        eq_by_type = self._classify_equations(equations, i, j)
        
        # 方程式の割り当てを決定
        assignments = self.system._assign_equations_2d(eq_by_type, i, j)
        
        # ソース項の処理（支配方程式に対応する行）
        if f_values is not None:
            governing_row = self._find_governing_row(assignments)
            if governing_row is not None:
                b[base_idx + governing_row] = f_values[i, j]
        
        # 境界条件の処理
        self._apply_boundary_conditions(b, base_idx, location, assignments, i, j, boundary_values)
    
    def _find_governing_row(self, assignments: List) -> Optional[int]:
        """
        支配方程式が割り当てられた行を見つける
        
        Args:
            assignments: 割り当てられた方程式のリスト
            
        Returns:
            支配方程式の行インデックス、見つからない場合はNone
        """
        # 必要なクラスを遅延インポート（依存関係を減らすため）
        from equation.dim2.poisson import PoissonEquation2D
        from equation.dim2.original import OriginalEquation2D
        
        for row, eq in enumerate(assignments):
            if isinstance(eq, (PoissonEquation2D, OriginalEquation2D)):
                return row
        return None
    
    def _apply_boundary_conditions(self, b: np.ndarray, base_idx: int, location: str, 
                                  assignments: List, i: int, j: int, 
                                  boundary_values: Dict[str, Any]):
        """
        境界条件を右辺ベクトルに適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            location: 点の位置
            assignments: 割り当てられた方程式のリスト
            i: x方向インデックス
            j: y方向インデックス
            boundary_values: 境界条件の値の辞書
        """
        # 必要なクラスを遅延インポート
        from equation.dim2.boundary import DirichletBoundaryEquation2D
        from equation.converter import DirectionalEquation2D
        from equation.dim1.boundary import NeumannBoundaryEquation
        
        # 各行の方程式をチェック
        for row, eq in enumerate(assignments):
            # ディリクレ条件
            if isinstance(eq, DirichletBoundaryEquation2D) and self.enable_dirichlet:
                self._apply_dirichlet_condition(b, base_idx, row, location, i, j, boundary_values)
            
            # DirectionalEquation2Dを使用したノイマン条件
            elif isinstance(eq, DirectionalEquation2D) and self.enable_neumann:
                # 内部方程式がNeumannBoundaryEquationかチェック
                if hasattr(eq, 'equation_1d') and isinstance(eq.equation_1d, NeumannBoundaryEquation):
                    # 方向に応じた処理
                    if eq.direction == 'x':
                        self._apply_neumann_x_condition(b, base_idx, row, location, i, j, boundary_values)
                    elif eq.direction == 'y':
                        self._apply_neumann_y_condition(b, base_idx, row, location, i, j, boundary_values)
    
    def _apply_dirichlet_condition(self, b: np.ndarray, base_idx: int, row: int, 
                                  location: str, i: int, j: int, 
                                  boundary_values: Dict[str, Any]):
        """
        ディリクレ境界条件を適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            row: 行インデックス
            location: 点の位置
            i: x方向インデックス
            j: y方向インデックス
            boundary_values: 境界条件の値の辞書
        """
        # 位置に応じて適切な境界値を選択
        if 'left' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'left_dirichlet', j)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
        elif 'right' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'right_dirichlet', j)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
        elif 'bottom' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'bottom_dirichlet', i)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
        elif 'top' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'top_dirichlet', i)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
    
    def _apply_neumann_x_condition(self, b: np.ndarray, base_idx: int, row: int, 
                                  location: str, i: int, j: int, 
                                  boundary_values: Dict[str, Any]):
        """
        X方向ノイマン境界条件を適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            row: 行インデックス
            location: 点の位置
            i: x方向インデックス
            j: y方向インデックス
            boundary_values: 境界条件の値の辞書
        """
        if 'left' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'left_neumann', j)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
        elif 'right' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'right_neumann', j)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
    
    def _apply_neumann_y_condition(self, b: np.ndarray, base_idx: int, row: int, 
                                  location: str, i: int, j: int, 
                                  boundary_values: Dict[str, Any]):
        """
        Y方向ノイマン境界条件を適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            row: 行インデックス
            location: 点の位置
            i: x方向インデックス
            j: y方向インデックス
            boundary_values: 境界条件の値の辞書
        """
        if 'bottom' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'bottom_neumann', i)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
        elif 'top' in location:
            boundary_value = self._get_boundary_value(boundary_values, 'top_neumann', i)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value