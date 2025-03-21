"""
1次元計算格子クラスモジュール

このモジュールは、1次元の計算格子を実装します。
"""

import numpy as np
from typing import Tuple, Union, Optional

from ..common.base_grid import BaseGrid

class Grid1D(BaseGrid):
    """
    1次元計算格子クラス
    
    x方向の1次元グリッドを管理します。
    """
    
    def __init__(self, n_points: int, x_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        1次元計算格子を初期化
        
        Args:
            n_points: 格子点数
            x_range: x方向の範囲 (x_min, x_max)
        """
        super().__init__()
        self._dimension = 1
        
        # 基本パラメータ
        self.n_points = n_points
        self.x_min, self.x_max = x_range
        
        # 格子間隔
        self.h = (self.x_max - self.x_min) / (self.n_points - 1)
        
        # 座標値
        self.x = np.linspace(self.x_min, self.x_max, self.n_points)
        
        # 後方互換性のため
        self.nx_points = n_points
    
    def get_point(self, i: int) -> float:
        """
        格子点iの座標値を返す
        
        Args:
            i: 格子点のインデックス
            
        Returns:
            格子点のx座標
        """
        if not 0 <= i < self.n_points:
            raise IndexError(f"インデックス {i} は範囲外です。有効範囲: 0-{self.n_points-1}")
            
        return self.x[i]
    
    def get_points(self) -> np.ndarray:
        """
        全格子点の座標値を返す
        
        Returns:
            xの座標値配列
        """
        return self.x
    
    def get_spacing(self) -> float:
        """
        格子間隔を返す
        
        Returns:
            格子間隔h
        """
        return self.h
    
    def is_boundary_point(self, i: int) -> bool:
        """
        境界点かどうかを判定
        
        Args:
            i: 格子点のインデックス
            
        Returns:
            境界上にあるかどうかのブール値
        """
        if not 0 <= i < self.n_points:
            raise IndexError(f"インデックス {i} は範囲外です。有効範囲: 0-{self.n_points-1}")
            
        return i == 0 or i == self.n_points - 1
    
    def get_left_boundary_index(self) -> int:
        """左境界のインデックスを返す"""
        return 0
    
    def get_right_boundary_index(self) -> int:
        """右境界のインデックスを返す"""
        return self.n_points - 1
    
    def get_domain_size(self) -> float:
        """領域のサイズを返す"""
        return self.x_max - self.x_min
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"Grid1D(n_points={self.n_points}, x_range=({self.x_min}, {self.x_max}), h={self.h})"
    
    def __repr__(self) -> str:
        """Python表現"""
        return self.__str__()
