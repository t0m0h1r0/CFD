"""
グリッド基底クラスモジュール

このモジュールは、1D, 2D, 3Dの計算格子に共通の基底クラスを提供します。
各次元の具象クラスは、この基底クラスを継承して実装されます。
"""

import abc
import numpy as np
from typing import Tuple, Union, Optional, List

class BaseGrid(abc.ABC):
    """
    すべての計算格子の基底クラス
    
    共通のインターフェースとユーティリティメソッドを提供します。
    """
    
    def __init__(self):
        """ベースグリッドの初期化"""
        self._dimension = 0  # 派生クラスで設定される次元
    
    @property
    def dimension(self) -> int:
        """グリッドの次元を返す"""
        return self._dimension
    
    @property
    def is_1d(self) -> bool:
        """1次元グリッドかどうかを判定"""
        return self._dimension == 1
    
    @property
    def is_2d(self) -> bool:
        """2次元グリッドかどうかを判定"""
        return self._dimension == 2
    
    @property
    def is_3d(self) -> bool:
        """3次元グリッドかどうかを判定"""
        return self._dimension == 3
    
    @abc.abstractmethod
    def get_point(self, *indices) -> Union[float, Tuple[float, ...], np.ndarray]:
        """
        指定インデックスの座標値を返す
        
        Args:
            *indices: 各次元のインデックス
            
        Returns:
            座標値（1Dは値、2D以上はタプル）
        """
        pass
    
    @abc.abstractmethod
    def get_points(self) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        全格子点の座標値を返す
        
        Returns:
            座標値の配列またはメッシュグリッド
        """
        pass
    
    @abc.abstractmethod
    def get_spacing(self) -> Union[float, Tuple[float, ...]]:
        """
        格子間隔を返す
        
        Returns:
            格子間隔（1Dはスカラー、2D以上はタプル）
        """
        pass
    
    @abc.abstractmethod
    def is_boundary_point(self, *indices) -> bool:
        """
        境界点かどうかを判定
        
        Args:
            *indices: 各次元のインデックス
            
        Returns:
            境界上にあるかどうかのブール値
        """
        pass
    
    def is_interior_point(self, *indices) -> bool:
        """
        内部点かどうかを判定
        
        Args:
            *indices: 各次元のインデックス
            
        Returns:
            内部にあるかどうかのブール値
        """
        return not self.is_boundary_point(*indices)
    
    def _to_numpy(self, arr) -> np.ndarray:
        """
        配列をNumPy配列に変換（CuPy, JAX対応）
        
        Args:
            arr: 変換する配列
            
        Returns:
            NumPy配列
        """
        if hasattr(arr, 'get'):  # CuPy
            return arr.get()
        elif hasattr(arr, 'device'):  # JAX/PyTorch
            return np.array(arr)
        return arr  # Already NumPy
