"""
高精度コンパクト差分法 (CCD) の1次元計算格子

このモジュールは、CCDソルバーで使用する1次元計算格子のクラスを提供します。
"""

import numpy as np

from core.base.base_grid import BaseGrid


class Grid1D(BaseGrid):
    """1次元計算格子クラス"""
    
    def __init__(self, n_points, x_range=(-1.0, 1.0)):
        """
        1D計算格子を初期化
        
        Args:
            n_points: 格子点数
            x_range: xの範囲 (x_min, x_max)
        """
        super().__init__()
        self.is_2d = False
        self.n_points = n_points
        self.x_min, self.x_max = x_range
        self.h = (self.x_max - self.x_min) / (self.n_points - 1)
        
        # x座標の生成
        self.x = np.linspace(self.x_min, self.x_max, self.n_points)
        
        # 後方互換性のための属性
        self.nx_points = n_points
        self.ny_points = 1
        self.hx = self.h
        self.hy = 1.0
    
    def get_point(self, i, j=None):
        """
        格子点の座標を取得
        
        Args:
            i: x方向インデックス
            j: 使用されない (1Dでは無視)
            
        Returns:
            x座標
        """
        if i < 0 or i >= self.n_points:
            raise IndexError(f"インデックス {i} は範囲外です (0 <= i < {self.n_points})")
        return self.x[i]
    
    def get_points(self):
        """
        全格子点を取得
        
        Returns:
            x座標の配列
        """
        return self.x
    
    def get_spacing(self):
        """
        格子間隔を取得
        
        Returns:
            格子間隔 h
        """
        return self.h
    
    def is_boundary_point(self, i, j=None):
        """
        境界点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: 使用されない (1Dでは無視)
            
        Returns:
            点が境界上にあるかどうかを示すブール値
        """
        return i == 0 or i == self.n_points - 1
    
    def is_interior_point(self, i, j=None):
        """
        内部点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: 使用されない (1Dでは無視)
            
        Returns:
            点が内部にあるかどうかを示すブール値
        """
        return 0 < i < self.n_points - 1
    
    def is_left_boundary(self, i):
        """
        左境界点かどうかをチェック
        
        Args:
            i: x方向インデックス
            
        Returns:
            点が左境界上にあるかどうかを示すブール値
        """
        return i == 0
    
    def is_right_boundary(self, i):
        """
        右境界点かどうかをチェック
        
        Args:
            i: x方向インデックス
            
        Returns:
            点が右境界上にあるかどうかを示すブール値
        """
        return i == self.n_points - 1
