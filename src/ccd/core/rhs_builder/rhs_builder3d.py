"""
高精度コンパクト差分法 (CCD) 用の3次元右辺ベクトル構築モジュール

このモジュールは、3次元のポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築するクラスを提供します。
"""

import numpy as np
from typing import Dict, Any, Optional, List

from core.base.base_rhs_builder import RHSBuilder
from equation.converter import DirectionalEquation3D
from equation.dim1.boundary import NeumannBoundaryEquation
from equation.dim3.boundary import DirichletBoundaryEquation3D


class RHSBuilder3D(RHSBuilder):
    """3次元問題用の右辺ベクトル構築クラス"""
    
    def build_rhs_vector(self, f_values=None, **boundary_values):
        """
        3D右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値（nx×ny×nz配列）
            **boundary_values: 境界値の辞書
                面の境界値:
                    face_x_min_dirichlet, face_x_max_dirichlet,
                    face_y_min_dirichlet, face_y_max_dirichlet,
                    face_z_min_dirichlet, face_z_max_dirichlet,
                    face_x_min_neumann, face_x_max_neumann, ...
                辺と頂点の境界値:
                    edge_*_dirichlet, edge_*_neumann
                    vertex_*_dirichlet
            
        Returns:
            right-hand side vector (NumPy配列)
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        var_per_point = 10  # 1点あたりの変数数 [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        system_size = nx * ny * nz * var_per_point
        b = np.zeros(system_size)
        
        # 入力値を NumPy に変換（必要な場合）
        if f_values is not None:
            f_values = self._to_numpy(f_values)
        
        # 境界条件の整合性を検証 (基底クラスの共通機能を使用)
        warnings = self._validate_boundary_conditions(boundary_values)
        if warnings:
            for warning in warnings:
                print(f"警告: {warning}")
        
        # 主要な面の境界条件情報を出力 (簡略化のため)
        self._print_boundary_info_summary(boundary_values)
        
        # 各格子点について処理
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # 点の位置を特定
                    location = self._get_point_location(i, j, k)
                    base_idx = (k * ny * nx + j * nx + i) * var_per_point
                    
                    # その位置に対応する方程式群を取得
                    location_equations = self.system.equations[location]
                    
                    # 右辺ベクトルの値をセット
                    self._set_rhs_for_point(b, base_idx, location, location_equations, i, j, k, f_values, boundary_values)
        
        return b
    
    def _print_boundary_info_summary(self, boundary_values: Dict[str, Any]):
        """
        主要な境界条件に関する情報を出力 (簡略化版)
        
        Args:
            boundary_values: 境界条件の値の辞書
        """
        faces = ["face_x_min", "face_x_max", "face_y_min", "face_y_max", "face_z_min", "face_z_max"]
        
        if self.enable_dirichlet:
            dirichlet_info = [f"{face}={face+'_dirichlet' in boundary_values}" for face in faces]
            print("[3Dソルバー] ディリクレ境界条件: " + ", ".join(dirichlet_info))
            
        if self.enable_neumann:
            neumann_info = [f"{face}={face+'_neumann' in boundary_values}" for face in faces]
            print("[3Dソルバー] ノイマン境界条件: " + ", ".join(neumann_info))
    
    def _get_point_location(self, i: int, j: int, k: int) -> str:
        """
        格子点の位置を取得
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            位置を表す文字列
        """
        # Grid3Dクラスの境界判定機能を利用
        return self.grid.get_boundary_type(i, j, k)
    
    def _set_rhs_for_point(self, b: np.ndarray, base_idx: int, location: str, 
                          equations: List, i: int, j: int, k: int, 
                          f_values: Optional[np.ndarray], 
                          boundary_values: Dict[str, Any]):
        """
        特定の格子点に対する右辺ベクトルを設定
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            location: 点の位置
            equations: その位置に対応する方程式のリスト
            i, j, k: 格子点のインデックス
            f_values: ソース項の値
            boundary_values: 境界条件の値の辞書
        """
        # 方程式の種類ごとに整理 (基底クラスの共通機能を使用)
        eq_by_type = self._classify_equations(equations, i, j, k)
        
        # 方程式の割り当てを決定
        assignments = self.system._assign_equations_3d(eq_by_type, i, j, k)
        
        # ソース項の処理（支配方程式に対応する行）
        if f_values is not None:
            governing_row = self._find_governing_row(assignments)
            if governing_row is not None:
                b[base_idx + governing_row] = f_values[i, j, k]
        
        # 境界条件の処理
        self._apply_boundary_conditions(b, base_idx, location, assignments, equations, i, j, k, boundary_values)
    
    def _find_governing_row(self, assignments: List) -> Optional[int]:
        """
        支配方程式が割り当てられた行を見つける
        
        Args:
            assignments: 割り当てられた方程式のリスト
            
        Returns:
            支配方程式の行インデックス、見つからない場合はNone
        """
        # 必要なクラスを遅延インポート（依存関係を減らすため）
        from equation.dim3.poisson import PoissonEquation3D
        from equation.dim3.original import OriginalEquation3D
        
        for row, eq in enumerate(assignments):
            if isinstance(eq, (PoissonEquation3D, OriginalEquation3D)):
                return row
        return None
    
    def _apply_boundary_conditions(self, b: np.ndarray, base_idx: int, location: str, 
                                  assignments: List, equations: List, i: int, j: int, k: int, 
                                  boundary_values: Dict[str, Any]):
        """
        境界条件を右辺ベクトルに適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            location: 点の位置
            assignments: 割り当てられた方程式のリスト
            equations: その位置に対応する方程式のリスト
            i, j, k: 格子点のインデックス
            boundary_values: 境界条件の値の辞書
        """
        # ディリクレ境界条件を適用
        for row, eq in enumerate(assignments):
            if isinstance(eq, DirichletBoundaryEquation3D) and self.enable_dirichlet:
                self._apply_dirichlet_condition(b, base_idx, row, location, i, j, k, boundary_values)
        
        # ノイマン境界条件を適用（変換元の1D方程式をチェック）
        for eq in equations:
            if isinstance(eq, DirectionalEquation3D) and self.enable_neumann:
                if hasattr(eq, 'equation_1d') and isinstance(eq.equation_1d, NeumannBoundaryEquation):
                    direction = eq.direction
                    # 方向に応じた行インデックスを決定
                    if direction == 'x':
                        row_idx = 1
                    elif direction == 'y':
                        row_idx = 4
                    elif direction == 'z':
                        row_idx = 7
                    else:
                        continue
                    
                    # ノイマン条件を適用
                    self._apply_neumann_condition(b, base_idx, row_idx, location, direction, i, j, k, boundary_values)
    
    def _apply_dirichlet_condition(self, b: np.ndarray, base_idx: int, row: int, 
                                  location: str, i: int, j: int, k: int,
                                  boundary_values: Dict[str, Any]):
        """
        ディリクレ境界条件を適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            row: 行インデックス
            location: 点の位置
            i, j, k: 格子点のインデックス
            boundary_values: 境界条件の値の辞書
        """
        # 境界条件の適用
        bc_key = f"{location}_dirichlet"
        if bc_key in boundary_values and boundary_values[bc_key] is not None:
            # 境界値は配列またはスカラー
            boundary_value = self._get_boundary_value_for_3d(boundary_values[bc_key], i, j, k, location)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
    
    def _apply_neumann_condition(self, b: np.ndarray, base_idx: int, row: int, 
                                location: str, direction: str, i: int, j: int, k: int,
                                boundary_values: Dict[str, Any]):
        """
        ノイマン境界条件を適用
        
        Args:
            b: 右辺ベクトル
            base_idx: 基準インデックス
            row: 行インデックス
            location: 点の位置
            direction: 微分方向 ('x', 'y', 'z')
            i, j, k: 格子点のインデックス
            boundary_values: 境界条件の値の辞書
        """
        # 正しいキーフォーマットを使用
        bc_key = f"{location}_neumann"
        if bc_key in boundary_values and boundary_values[bc_key] is not None:
            boundary_value = self._get_boundary_value_for_3d(boundary_values[bc_key], i, j, k, location)
            if boundary_value is not None:
                b[base_idx + row] = boundary_value
    
    def _get_boundary_value_for_3d(self, value, i: int, j: int, k: int, location: str) -> Optional[float]:
        """
        3D格子点に対応する境界値を取得
        
        Args:
            value: 境界値（スカラー、1D配列、2D配列、または3D配列）
            i, j, k: 格子点のインデックス
            location: 点の位置
            
        Returns:
            境界値、またはNone（適用できない場合）
        """
        if value is None:
            return None
            
        # スカラー値の場合
        if np.isscalar(value):
            return value
            
        # 配列の場合、位置に応じた添え字を決定
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        
        # 面の場合（2次元添え字）
        if 'face_x' in location:
            # x面: (j, k) が変数
            idx1, idx2 = j, k
            _max1, _max2 = ny, nz
        elif 'face_y' in location:
            # y面: (i, k) が変数
            idx1, idx2 = i, k
            _max1, _max2 = nx, nz
        elif 'face_z' in location:
            # z面: (i, j) が変数
            idx1, idx2 = i, j
            _max1, _max2 = nx, ny
        # 辺の場合（1次元添え字）
        elif 'edge_x' in location:
            # x辺: i が変数
            idx1, _max1 = i, nx
        elif 'edge_y' in location:
            # y辺: j が変数
            idx1, _max1 = j, ny
        elif 'edge_z' in location:
            # z辺: k が変数
            idx1, _max1 = k, nz
        # 頂点の場合（添え字は不要）
        else:
            return value if np.isscalar(value) else value.item()
            
        # 値の次元数に応じて適切に取得
        array_value = np.asarray(value)
        if array_value.ndim == 1:
            # 1次元配列
            if 'edge' in location:
                return array_value[idx1] if idx1 < array_value.shape[0] else array_value[-1]
            else:
                # 面には2次元配列が必要なのでエラー
                return None
        elif array_value.ndim == 2:
            # 2次元配列
            if 'face' in location:
                if idx1 < array_value.shape[0] and idx2 < array_value.shape[1]:
                    return array_value[idx1, idx2]
                else:
                    return array_value[-1, -1]
            else:
                return None
        elif array_value.ndim == 3:
            # 3次元配列
            if i < array_value.shape[0] and j < array_value.shape[1] and k < array_value.shape[2]:
                return array_value[i, j, k]
            else:
                return array_value[-1, -1, -1]
                
        return None