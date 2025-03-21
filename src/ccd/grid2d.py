"""
高精度コンパクト差分法 (CCD) の2次元計算格子

このモジュールは、CCDソルバーで使用する2次元計算格子のクラスを提供します。
"""

import numpy as np

from base_grid import BaseGrid


class Grid2D(BaseGrid):
    """2次元計算格子クラス"""
    
    def __init__(self, nx_points, ny_points, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)):
        """
        2D計算格子を初期化
        
        Args:
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数
            x_range: xの範囲 (x_min, x_max)
            y_range: yの範囲 (y_min, y_max)
        """
        super().__init__()
        self.is_2d = True
        
        # 2D格子の基本パラメータ
        self.nx_points = nx_points
        self.ny_points = ny_points
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        
        # 格子間隔の計算
        self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
        self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
        
        # 座標の生成
        self.x = np.linspace(self.x_min, self.x_max, self.nx_points)
        self.y = np.linspace(self.y_min, self.y_max, self.ny_points)
        
        # メッシュグリッド作成（ベクトル計算に便利）
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # 後方互換性のため
        self.n_points = max(nx_points, ny_points)
        self.h = self.hx
    
    def get_point(self, i, j=None):
        """
        格子点の座標を取得
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            (x, y)座標のタプル
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
        if i < 0 or i >= self.nx_points or j < 0 or j >= self.ny_points:
            raise IndexError(f"インデックス ({i}, {j}) は範囲外です")
            
        return self.x[i], self.y[j]
    
    def get_points(self):
        """
        全格子点を取得
        
        Returns:
            (X, Y)メッシュグリッドのタプル
        """
        return self.X, self.Y
    
    def get_spacing(self):
        """
        格子間隔を取得
        
        Returns:
            (hx, hy)格子間隔のタプル
        """
        return self.hx, self.hy
    
    def get_index(self, i, j=None):
        """
        2Dインデックスを平坦化された1Dインデックスに変換
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            1Dインデックス
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return i + j * self.nx_points
    
    def get_indices(self, flat_index):
        """
        平坦化された1Dインデックスを格子インデックスに変換
        
        Args:
            flat_index: 1Dインデックス
            
        Returns:
            (i, j)タプル
        """
        j = flat_index // self.nx_points
        i = flat_index % self.nx_points
        return i, j
    
    def is_boundary_point(self, i, j=None):
        """
        境界点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点が境界上にあるかどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return (i == 0 or i == self.nx_points - 1 or 
                j == 0 or j == self.ny_points - 1)
    
    def is_corner_point(self, i, j=None):
        """
        角点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点が角点かどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return ((i == 0 or i == self.nx_points - 1) and 
                (j == 0 or j == self.ny_points - 1))
    
    def is_interior_point(self, i, j=None):
        """
        内部点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点が内部にあるかどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return (0 < i < self.nx_points - 1 and 
                0 < j < self.ny_points - 1)
    
    def is_edge_point(self, i, j=None):
        """
        エッジ点（角を除く境界点）かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点がエッジ点かどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
        return self.is_boundary_point(i, j) and not self.is_corner_point(i, j)
    
    def is_left_boundary(self, i, j=None):
        """
        左境界上かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点が左境界上にあるかどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
        return i == 0 and 0 < j < self.ny_points - 1
    
    def is_right_boundary(self, i, j=None):
        """
        右境界上かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点が右境界上にあるかどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
        return i == self.nx_points - 1 and 0 < j < self.ny_points - 1
    
    def is_bottom_boundary(self, i, j=None):
        """
        下境界上かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点が下境界上にあるかどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
        return j == 0 and 0 < i < self.nx_points - 1
    
    def is_top_boundary(self, i, j=None):
        """
        上境界上かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            
        Returns:
            点が上境界上にあるかどうかを示すブール値
        """
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
        return j == self.ny_points - 1 and 0 < i < self.nx_points - 1
