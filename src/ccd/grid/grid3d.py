"""
3次元計算格子クラスモジュール

このモジュールは、3次元の計算格子を実装します。
3次元座標系では、xは水平方向、yは奥行き方向、zは高さ方向とします。
"""

import numpy as np
from typing import Tuple, Union, Optional, List

from ..common.base_grid import BaseGrid

class Grid3D(BaseGrid):
    """
    3次元計算格子クラス
    
    x方向（水平方向）、y方向（奥行き方向）、z方向（高さ方向）の3次元グリッドを管理します。
    """
    
    def __init__(self, 
                 nx_points: int, 
                 ny_points: int, 
                 nz_points: int,
                 x_range: Tuple[float, float] = (-1.0, 1.0),
                 y_range: Tuple[float, float] = (-1.0, 1.0),
                 z_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        3次元計算格子を初期化
        
        Args:
            nx_points: x方向（水平方向）の格子点数
            ny_points: y方向（奥行き方向）の格子点数
            nz_points: z方向（高さ方向）の格子点数
            x_range: x方向の範囲 (x_min, x_max)
            y_range: y方向の範囲 (y_min, y_max)
            z_range: z方向の範囲 (z_min, z_max)
        """
        super().__init__()
        self._dimension = 3
        
        # 基本パラメータ
        self.nx_points = nx_points
        self.ny_points = ny_points
        self.nz_points = nz_points
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        
        # 格子間隔
        self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
        self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
        self.hz = (self.z_max - self.z_min) / (self.nz_points - 1)
        
        # 座標値
        self.x = np.linspace(self.x_min, self.x_max, self.nx_points)
        self.y = np.linspace(self.y_min, self.y_max, self.ny_points)
        self.z = np.linspace(self.z_min, self.z_max, self.nz_points)
        
        # メッシュグリッド作成（ベクトル計算に便利）
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def get_point(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """
        格子点(i,j,k)の座標値を返す
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            格子点の(x, y, z)座標のタプル
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
        if not (0 <= k < self.nz_points):
            raise IndexError(f"zインデックス {k} は範囲外です。有効範囲: 0-{self.nz_points-1}")
            
        return self.x[i], self.y[j], self.z[k]
    
    def get_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        全格子点の座標値を返す
        
        Returns:
            X, Y, Zメッシュグリッドのタプル
        """
        return self.X, self.Y, self.Z
    
    def get_spacing(self) -> Tuple[float, float, float]:
        """
        格子間隔を返す
        
        Returns:
            (hx, hy, hz)のタプル
        """
        return self.hx, self.hy, self.hz
    
    def is_boundary_point(self, i: int, j: int, k: int) -> bool:
        """
        境界点かどうかを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            境界上にあるかどうかのブール値
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
        if not (0 <= k < self.nz_points):
            raise IndexError(f"zインデックス {k} は範囲外です。有効範囲: 0-{self.nz_points-1}")
            
        return (i == 0 or i == self.nx_points - 1 or 
                j == 0 or j == self.ny_points - 1 or
                k == 0 or k == self.nz_points - 1)
    
    def is_corner_point(self, i: int, j: int, k: int) -> bool:
        """
        頂点（角）点かどうかを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            頂点点かどうかのブール値
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
        if not (0 <= k < self.nz_points):
            raise IndexError(f"zインデックス {k} は範囲外です。有効範囲: 0-{self.nz_points-1}")
            
        return ((i == 0 or i == self.nx_points - 1) and 
                (j == 0 or j == self.ny_points - 1) and
                (k == 0 or k == self.nz_points - 1))
    
    def is_edge_point(self, i: int, j: int, k: int) -> bool:
        """
        辺上の点（頂点を除く）かどうかを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            辺上の点かどうかのブール値
        """
        if not self.is_boundary_point(i, j, k) or self.is_corner_point(i, j, k):
            return False
        
        edge_conditions = [
            # x軸に沿った辺
            0 < i < self.nx_points - 1 and (j == 0 or j == self.ny_points - 1) and (k == 0 or k == self.nz_points - 1),
            # y軸に沿った辺
            (i == 0 or i == self.nx_points - 1) and 0 < j < self.ny_points - 1 and (k == 0 or k == self.nz_points - 1),
            # z軸に沿った辺
            (i == 0 or i == self.nx_points - 1) and (j == 0 or j == self.ny_points - 1) and 0 < k < self.nz_points - 1
        ]
        
        return any(edge_conditions)
    
    def is_face_point(self, i: int, j: int, k: int) -> bool:
        """
        面上の点（辺と頂点を除く）かどうかを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            面上の点かどうかのブール値
        """
        if not self.is_boundary_point(i, j, k) or self.is_edge_point(i, j, k) or self.is_corner_point(i, j, k):
            return False
        
        face_conditions = [
            # x方向の面 (i = 0 or i = nx_points-1)
            (i == 0 or i == self.nx_points - 1) and 0 < j < self.ny_points - 1 and 0 < k < self.nz_points - 1,
            # y方向の面 (j = 0 or j = ny_points-1)
            0 < i < self.nx_points - 1 and (j == 0 or j == self.ny_points - 1) and 0 < k < self.nz_points - 1,
            # z方向の面 (k = 0 or k = nz_points-1)
            0 < i < self.nx_points - 1 and 0 < j < self.ny_points - 1 and (k == 0 or k == self.nz_points - 1)
        ]
        
        return any(face_conditions)
    
    def get_index(self, i: int, j: int, k: int) -> int:
        """
        3次元インデックスを平坦化された1次元インデックスに変換
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            平坦化された1次元インデックス
        """
        if not (0 <= i < self.nx_points):
            raise IndexError(f"xインデックス {i} は範囲外です。有効範囲: 0-{self.nx_points-1}")
        if not (0 <= j < self.ny_points):
            raise IndexError(f"yインデックス {j} は範囲外です。有効範囲: 0-{self.ny_points-1}")
        if not (0 <= k < self.nz_points):
            raise IndexError(f"zインデックス {k} は範囲外です。有効範囲: 0-{self.nz_points-1}")
            
        return i + j * self.nx_points + k * self.nx_points * self.ny_points
    
    def get_indices(self, flat_index: int) -> Tuple[int, int, int]:
        """
        平坦化された1次元インデックスを3次元インデックスに変換
        
        Args:
            flat_index: 平坦化された1次元インデックス
            
        Returns:
            (i, j, k)の3次元インデックスのタプル
        """
        if not (0 <= flat_index < self.nx_points * self.ny_points * self.nz_points):
            max_idx = self.nx_points * self.ny_points * self.nz_points - 1
            raise IndexError(f"インデックス {flat_index} は範囲外です。有効範囲: 0-{max_idx}")
            
        k = flat_index // (self.nx_points * self.ny_points)
        remainder = flat_index % (self.nx_points * self.ny_points)
        j = remainder // self.nx_points
        i = remainder % self.nx_points
        return i, j, k
    
    def get_location_name(self, i: int, j: int, k: int) -> str:
        """
        格子点の位置名を返す（内部、面、辺、頂点）
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            位置を表す文字列
        """
        if self.is_corner_point(i, j, k):
            # 頂点の位置名を返す
            x_pos = "left" if i == 0 else "right"
            y_pos = "bottom" if j == 0 else "top"
            z_pos = "front" if k == 0 else "back"
            return f"{x_pos}_{y_pos}_{z_pos}"
        elif self.is_edge_point(i, j, k):
            # 辺の位置名を返す
            if 0 < i < self.nx_points - 1:  # x方向に内部
                y_pos = "bottom" if j == 0 else "top"
                z_pos = "front" if k == 0 else "back"
                return f"{y_pos}_{z_pos}"
            elif 0 < j < self.ny_points - 1:  # y方向に内部
                x_pos = "left" if i == 0 else "right"
                z_pos = "front" if k == 0 else "back"
                return f"{x_pos}_{z_pos}"
            else:  # z方向に内部
                x_pos = "left" if i == 0 else "right"
                y_pos = "bottom" if j == 0 else "top"
                return f"{x_pos}_{y_pos}"
        elif self.is_face_point(i, j, k):
            # 面の位置名を返す
            if i == 0:
                return "left"
            elif i == self.nx_points - 1:
                return "right"
            elif j == 0:
                return "bottom"
            elif j == self.ny_points - 1:
                return "top"
            elif k == 0:
                return "front"
            else:  # k == self.nz_points - 1
                return "back"
        else:
            return "interior"
    
    def get_domain_size(self) -> Tuple[float, float, float]:
        """領域のサイズを返す"""
        return self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min
    
    def __str__(self) -> str:
        """文字列表現"""
        return (f"Grid3D(nx_points={self.nx_points}, ny_points={self.ny_points}, nz_points={self.nz_points}, "
                f"x_range=({self.x_min}, {self.x_max}), y_range=({self.y_min}, {self.y_max}), "
                f"z_range=({self.z_min}, {self.z_max}), hx={self.hx}, hy={self.hy}, hz={self.hz})")
    
    def __repr__(self) -> str:
        """Python表現"""
        return self.__str__()
