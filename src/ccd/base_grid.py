"""
高精度コンパクト差分法 (CCD) の計算格子の基底クラス

このモジュールは、CCDソルバーで使用する計算格子の
基底クラスと共通機能を提供します。
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union, List, Optional


class BaseGrid(ABC):
    """計算格子の抽象基底クラス"""
    
    def __init__(self):
        """初期化"""
        self.is_2d = False  # サブクラスでオーバーライドされる
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_points(self):
        """
        全格子点を取得
        
        Returns:
            1D: x座標の配列
            2D: (X, Y)メッシュグリッドのタプル
        """
        pass
    
    @abstractmethod
    def get_spacing(self):
        """
        格子間隔を取得
        
        Returns:
            1D: hスカラー
            2D: (hx, hy)タプル
        """
        pass
    
    @abstractmethod
    def is_boundary_point(self, i, j=None):
        """
        境界点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            点が境界上にあるかどうかを示すブール値
        """
        pass
    
    @abstractmethod
    def is_interior_point(self, i, j=None):
        """
        内部点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            点が内部にあるかどうかを示すブール値
        """
        pass
