"""
2次元計算格子クラスモジュール

このモジュールは、2次元の計算格子を実装します。
"""

import numpy as np
from typing import Tuple, Union, Optional

from ..common.base_grid import BaseGrid

class Grid2D(BaseGrid):
    """
    2次元計算格子クラス
    
    x方向（水平方向）とy方向（奥行き方向）の2次元グリッドを管理します。
    """
    
    def __init__(self, 
                 nx_points: int, 
                 ny_points: int, 
                 x_range: Tuple[float, float] = (-1.0, 1.0),
                 y_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        2次元計算格子を初期化
        
        Args:
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数
            x_range: x方向の範囲 (x_min, x_max)
            y_range: y方向の範囲 (y_min, y_max)
        """
        super().__init__()
        self._dimension = 2
        
        # 基本パラメータ
        self.nx_points = nx_points
        self.ny_points = ny_points
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        
        # 格子間隔
        self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
        self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
        
        # 座標値
        self.x = np.linspace(self.x_min, self.x_max, self.nx_points)
        self.y = np.linspace(self.y_min, self.y_max, self.ny_points)
        
        # メッシュグリッド作成（ベクトル計算に便利）
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
    
    def get_point(self, i: int, j: int) -> Tuple[float, float]:
        """
        格子点(i,j)の座標値を返す
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            格子点の(x, y)座標のタプル
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
            
        return self.x[i], self.y[j]
    
    def get_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        全格子点の座標値を返す
        
        Returns:
            X, Yメッシュグリッドのタプル
        """
        return self.X, self.Y
    
    def get_spacing(self) -> Tuple[float, float]:
        """
        格子間隔を返す
        
        Returns:
            (hx, hy)のタプル
        """
        return self.hx, self.hy
    
    def is_boundary_point(self, i: int, j: int) -> bool:
        """
        境界点かどうかを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            境界上にあるかどうかのブール値
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
            
        return (i == 0 or i == self.nx_points - 1 or 
                j == 0 or j == self.ny_points - 1)
    
    def is_corner_point(self, i: int, j: int) -> bool:
        """
        角点かどうかを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            角点かどうかのブール値
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
            
        return ((i == 0 or i == self.nx_points - 1) and 
                (j == 0 or j == self.ny_points - 1))
    
    def is_edge_point(self, i: int, j: int) -> bool:
        """
        辺上の点（角を除く）かどうかを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            辺上の点かどうかのブール値
        """
        return self.is_boundary_point(i, j) and not self.is_corner_point(i, j)
    
    def get_index(self, i: int, j: int) -> int:
        """
        2次元インデックスを平坦化された1次元インデックスに変換
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            平坦化された1次元インデックス
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
            
        return i + j * self.nx_points
    
    def get_indices(self, flat_index: int) -> Tuple[int, int]:
        """
        平坦化された1次元インデックスを2次元インデックスに変換
        
        Args:
            flat_index: 平坦化された1次元インデックス
            
        Returns:
            (i, j)の2次元インデックスのタプル
        """
        if not (0 <= flat_index < self.nx_points * self.ny_points):
            raise IndexError(f"インデックス {flat_index} は範囲外です。有効範囲: 0-{self.nx_points * self.ny_points - 1}")
            
        j = flat_index // self.nx_points
        i = flat_index % self.nx_points
        return i, j
    
    def get_domain_size(self) -> Tuple[float, float]:
        """領域のサイズを返す"""
        return self.x_max - self.x_min, self.y_max - self.y_min
    
    def __str__(self) -> str:
        """文字列表現"""
        return (f"Grid2D(nx_points={self.nx_points}, ny_points={self.ny_points}, "
                f"x_range=({self.x_min}, {self.x_max}), y_range=({self.y_min}, {self.y_max}), "
                f"hx={self.hx}, hy={self.hy})")
    
    def __repr__(self) -> str:
        """Python表現"""
        return self.__str__()
