"""
高精度コンパクト差分法 (CCD) 用の右辺ベクトル構築の基底クラス

このモジュールは、ポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築する共通基底クラスを提供します。
次元ごとの実装を効率化するための共通機能が含まれています。
"""

from abc import ABC, abstractmethod
import numpy as np


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
        
        # 次元を判定
        self.dimension = self._detect_dimension()
    
    def _detect_dimension(self) -> int:
        """グリッドの次元を判定"""
        if hasattr(self.grid, 'is_3d') and self.grid.is_3d:
            return 3
        elif hasattr(self.grid, 'is_2d') and self.grid.is_2d:
            return 2
        else:
            return 1
    
    @abstractmethod
    def build_rhs_vector(self, f_values=None, **boundary_values):
        """
        右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値 (1Dの場合は長さnの配列、2Dの場合はn×m行列)
            **boundary_values: 境界値の辞書
                1Dの場合:
                    left_dirichlet: 左端のディリクレ境界値
                    right_dirichlet: 右端のディリクレ境界値
                    left_neumann: 左端のノイマン境界値
                    right_neumann: 右端のノイマン境界値
                2Dの場合:
                    left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet
                    left_neumann, right_neumann, bottom_neumann, top_neumann
                3Dの場合:
                    face_*_dirichlet, face_*_neumann
                    edge_*_dirichlet, edge_*_neumann
                    vertex_*_dirichlet
            
        Returns:
            右辺ベクトル（NumPy配列）
        """
        pass
    
    def _to_numpy(self, arr):
        """
        CuPy配列をNumPy配列に変換する (必要な場合のみ)
        
        Args:
            arr: 変換する配列
            
        Returns:
            NumPy配列またはスカラー
        """
        if arr is None:
            return None
            
        if hasattr(arr, 'get'):
            return arr.get()
        return arr
        
    def _validate_boundary_conditions(self, boundary_values):
        """
        与えられた境界条件の整合性を検証
        
        Args:
            boundary_values: 境界条件値の辞書
            
        Returns:
            警告メッセージのリスト（問題がない場合は空）
        """
        warnings = []
        
        # 次元に応じた境界条件の検証
        if self.dimension == 1:
            self._validate_1d_boundary_conditions(boundary_values, warnings)
        elif self.dimension == 2:
            self._validate_2d_boundary_conditions(boundary_values, warnings)
        else:  # 3次元
            self._validate_3d_boundary_conditions(boundary_values, warnings)
        
        return warnings
    
    def _validate_1d_boundary_conditions(self, boundary_values, warnings):
        """1D境界条件の検証"""
        # ディリクレ境界条件が有効なのに、値が指定されていない場合
        if self.enable_dirichlet:
            for key in ['left_dirichlet', 'right_dirichlet']:
                if key not in boundary_values or boundary_values[key] is None:
                    warnings.append(f"ディリクレ境界条件が有効ですが、{key}が指定されていません")
        
        # ノイマン境界条件が有効なのに、値が指定されていない場合
        if self.enable_neumann:
            for key in ['left_neumann', 'right_neumann']:
                if key not in boundary_values or boundary_values[key] is None:
                    warnings.append(f"ノイマン境界条件が有効ですが、{key}が指定されていません")
    
    def _validate_2d_boundary_conditions(self, boundary_values, warnings):
        """2D境界条件の検証"""
        # ディリクレ境界条件が有効なのに、値が指定されていない場合
        if self.enable_dirichlet:
            for key in ['left_dirichlet', 'right_dirichlet', 'bottom_dirichlet', 'top_dirichlet']:
                if key not in boundary_values or boundary_values[key] is None:
                    warnings.append(f"ディリクレ境界条件が有効ですが、{key}が指定されていません")
        
        # ノイマン境界条件が有効なのに、値が指定されていない場合
        if self.enable_neumann:
            for key in ['left_neumann', 'right_neumann', 'bottom_neumann', 'top_neumann']:
                if key not in boundary_values or boundary_values[key] is None:
                    warnings.append(f"ノイマン境界条件が有効ですが、{key}が指定されていません")
    
    def _validate_3d_boundary_conditions(self, boundary_values, warnings):
        """3D境界条件の検証"""
        # 簡略化のため主要な面のみをチェック
        faces = ["face_x_min", "face_x_max", "face_y_min", "face_y_max", "face_z_min", "face_z_max"]
        
        # ディリクレ境界条件が有効なのに、値が指定されていない場合
        if self.enable_dirichlet:
            for face in faces:
                key = f"{face}_dirichlet"
                if key not in boundary_values or boundary_values[key] is None:
                    warnings.append(f"ディリクレ境界条件が有効ですが、{key}が指定されていません")
        
        # ノイマン境界条件が有効なのに、値が指定されていない場合
        if self.enable_neumann:
            for face in faces:
                key = f"{face}_neumann"
                if key not in boundary_values or boundary_values[key] is None:
                    warnings.append(f"ノイマン境界条件が有効ですが、{key}が指定されていません")
    
    def _find_equation_by_type(self, location_equations, equation_type, i, j=None, k=None):
        """
        特定のタイプの方程式を検索
        
        Args:
            location_equations: 位置に対応する方程式のリスト
            equation_type: 方程式の種類 ("dirichlet", "neumann", etc.)
            i: x方向インデックス
            j: y方向インデックス (2D/3Dの場合のみ)
            k: z方向インデックス (3Dの場合のみ)
            
        Returns:
            (方程式, インデックス)のタプル、見つからない場合は(None, None)
        """
        for idx, eq in enumerate(location_equations):
            if self.dimension == 1:
                eq_type = self.system._identify_equation_type(eq, i)
            elif self.dimension == 2:
                eq_type = self.system._identify_equation_type(eq, i, j)
            else:  # 3次元
                eq_type = self.system._identify_equation_type(eq, i, j, k)
                
            if eq_type == equation_type:
                return eq, idx
        return None, None
    
    def _get_boundary_value(self, boundary_values, key, idx):
        """
        境界値を取得
        
        Args:
            boundary_values: 境界条件の値の辞書
            key: 境界条件の種類を示すキー
            idx: インデックス
            
        Returns:
            境界値、または境界値が指定されていない場合はNone
        """
        value = boundary_values.get(key)
        if value is None:
            return None
            
        # 配列の場合はインデックスを使用
        if isinstance(value, (list, np.ndarray)) and idx < len(value):
            return value[idx]
        # スカラーの場合はそのまま返す
        return value
    
    def _classify_equations(self, equations, i, j=None, k=None):
        """
        方程式をタイプごとに分類
        
        Args:
            equations: 方程式のリスト
            i, j, k: 格子点のインデックス (次元に応じて異なる)
            
        Returns:
            分類結果の辞書
        """
        # 次元に応じた分類を実行
        if self.dimension == 1:
            return self._classify_equations_1d(equations, i)
        elif self.dimension == 2:
            return self._classify_equations_2d(equations, i, j)
        else:  # 3次元
            return self._classify_equations_3d(equations, i, j, k)
    
    def _classify_equations_1d(self, equations, i):
        """1D方程式の分類"""
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
    
    def _classify_equations_2d(self, equations, i, j):
        """2D方程式の分類"""
        eq_by_type = {
            "governing": None, 
            "dirichlet": None, 
            "neumann_x": None, 
            "neumann_y": None, 
            "auxiliary": []
        }
        
        for eq in equations:
            eq_type = self.system._identify_equation_type(eq, i, j)
            if eq_type == "auxiliary":
                eq_by_type["auxiliary"].append(eq)
            elif eq_type:  # Noneでない場合
                eq_by_type[eq_type] = eq
        
        return eq_by_type
    
    def _classify_equations_3d(self, equations, i, j, k):
        """3D方程式の分類"""
        eq_by_type = {
            "governing": None, 
            "dirichlet": None, 
            "neumann_x": None, 
            "neumann_y": None, 
            "neumann_z": None,
            "auxiliary": []
        }
        
        for eq in equations:
            eq_type = self.system._identify_equation_type(eq, i, j, k)
            if eq_type == "auxiliary":
                eq_by_type["auxiliary"].append(eq)
            elif eq_type:  # Noneでない場合
                eq_by_type[eq_type] = eq
        
        return eq_by_type