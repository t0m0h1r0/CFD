"""
高精度コンパクト差分法 (CCD) 用の右辺ベクトル構築モジュール

このモジュールは、ポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築する基底クラスを提供します。
"""

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

    def _find_equation_by_type(self, location_equations, equation_type, i, j=None):
        """
        特定のタイプの方程式を検索
        
        Args:
            location_equations: 位置に対応する方程式のリスト
            equation_type: 方程式の種類 ("dirichlet", "neumann", "neumann_x", "neumann_y")
            i: x方向インデックス
            j: y方向インデックス (2Dの場合のみ)
            
        Returns:
            (方程式, インデックス)のタプル、見つからない場合は(None, None)
        """
        for idx, eq in enumerate(location_equations):
            eq_type = self.system._identify_equation_type(eq, i, j)
            if eq_type == equation_type:
                return eq, idx
        return None, None
    
    def _validate_boundary_conditions(self, boundary_values):
        """
        与えられた境界条件の整合性を検証
        
        Args:
            boundary_values: 境界条件値の辞書
            
        Returns:
            警告メッセージのリスト（問題がない場合は空）
        """
        warnings = []
        
        # ディリクレ境界条件が有効なのに、値が指定されていない場合
        if self.enable_dirichlet:
            if self.grid.is_2d:
                for key in ['left_dirichlet', 'right_dirichlet', 'bottom_dirichlet', 'top_dirichlet']:
                    if key not in boundary_values or boundary_values[key] is None:
                        warnings.append(f"ディリクレ境界条件が有効ですが、{key}が指定されていません")
            else:
                for key in ['left_dirichlet', 'right_dirichlet']:
                    if key not in boundary_values or boundary_values[key] is None:
                        warnings.append(f"ディリクレ境界条件が有効ですが、{key}が指定されていません")
        
        # ノイマン境界条件が有効なのに、値が指定されていない場合
        if self.enable_neumann:
            if self.grid.is_2d:
                for key in ['left_neumann', 'right_neumann', 'bottom_neumann', 'top_neumann']:
                    if key not in boundary_values or boundary_values[key] is None:
                        warnings.append(f"ノイマン境界条件が有効ですが、{key}が指定されていません")
            else:
                for key in ['left_neumann', 'right_neumann']:
                    if key not in boundary_values or boundary_values[key] is None:
                        warnings.append(f"ノイマン境界条件が有効ですが、{key}が指定されていません")
        
        return warnings