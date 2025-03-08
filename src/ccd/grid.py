# grid.py
import cupy as cp  # NumPyではなくCuPyを使用
from typing import Tuple, List

class Grid:
    """計算格子を管理するクラス"""
    
    def __init__(self, n_points: int, x_range: Tuple[float, float]):
        """
        初期化
        
        Args:
            n_points: 格子点の数
            x_range: (x_min, x_max)の範囲
        """
        self.n_points = n_points
        self.x_min, self.x_max = x_range
        self.h = (self.x_max - self.x_min) / (self.n_points - 1)
        self.x = cp.linspace(self.x_min, self.x_max, self.n_points)
    
    def get_point(self, i: int) -> float:
        """i番目の格子点の座標を返す"""
        return self.x[i]
    
    def get_points(self) -> cp.ndarray:
        """全格子点の座標を返す"""
        return self.x
    
    def get_spacing(self) -> float:
        """格子間隔を返す"""
        return self.h