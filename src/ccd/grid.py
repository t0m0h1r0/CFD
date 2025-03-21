"""
高精度コンパクト差分法 (CCD) の計算格子モジュール

このモジュールは、CCDソルバーで使用する計算格子の後方互換性を提供します。
"""

from base_grid import BaseGrid
from grid1d import Grid1D
from grid2d import Grid2D


class Grid:
    """統合計算格子クラス (1Dと2Dの機能を統合、後方互換性用)"""
    
    def __init__(self, nx_points, ny_points=None, x_range=None, y_range=None):
        """
        計算格子を初期化 (1Dまたは2D)
        
        Args:
            nx_points: x方向の格子点数 (1Dの場合は総点数)
            ny_points: y方向の格子点数 (1Dの場合はNone)
            x_range: xの範囲 (x_min, x_max)
            y_range: yの範囲 (y_min, y_max) (1Dの場合はNone)
        """
        # 1Dか2Dかを判定
        self.is_2d = ny_points is not None
        
        # デフォルト範囲の設定
        if x_range is None:
            x_range = (-1.0, 1.0)
        
        if self.is_2d:
            # 2D格子の初期化
            if y_range is None:
                y_range = x_range  # デフォルトではx範囲と同じにする
                
            self._grid = Grid2D(nx_points, ny_points, x_range, y_range)
        else:
            # 1D格子の初期化
            self._grid = Grid1D(nx_points, x_range)
        
        # 全ての属性をグリッドオブジェクトから取得するようにする
        # __getattr__を使って自動的に処理
    
    def __getattr__(self, name):
        """
        属性アクセスを内部の1D/2Dグリッドオブジェクトに委譲
        
        Args:
            name: 属性名
            
        Returns:
            対応する属性値
        """
        if hasattr(self._grid, name):
            return getattr(self._grid, name)
        raise AttributeError(f"'Grid' オブジェクトには属性 '{name}' がありません")
    
    # 主要メソッドは明示的に委譲（補完のサポートのため）
    
    def get_point(self, i, j=None):
        """
        格子点の座標を取得
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (1Dの場合はNone)
            
        Returns:
            1D: x座標
            2D: (x, y)座標のタプル
        """
        return self._grid.get_point(i, j)
    
    def get_points(self):
        """
        全格子点を取得
        
        Returns:
            1D: x座標の配列
            2D: (X, Y)メッシュグリッドのタプル
        """
        return self._grid.get_points()
    
    def get_spacing(self):
        """
        格子間隔を取得
        
        Returns:
            1D: hスカラー
            2D: (hx, hy)タプル
        """
        return self._grid.get_spacing()
    
    def is_boundary_point(self, i, j=None):
        """
        境界点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            点が境界上にあるかどうかを示すブール値
        """
        return self._grid.is_boundary_point(i, j)
    
    def is_interior_point(self, i, j=None):
        """
        内部点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            点が内部にあるかどうかを示すブール値
        """
        return self._grid.is_interior_point(i, j)


# 後方互換性のためにエクスポート
__all__ = ["BaseGrid", "Grid1D", "Grid2D", "Grid"]