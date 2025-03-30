"""
高精度コンパクト差分法 (CCD) の3次元計算格子

このモジュールは、CCDソルバーで使用する3次元計算格子のクラスを提供します。
"""

import numpy as np
from itertools import product
from core.base.base_grid import BaseGrid


class Grid3D(BaseGrid):
    """3次元計算格子クラス"""
    
    def __init__(self, nx_points, ny_points, nz_points, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-1.0, 1.0)):
        """
        3D計算格子を初期化
        
        Args:
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数
            nz_points: z方向の格子点数
            x_range: xの範囲 (x_min, x_max)
            y_range: yの範囲 (y_min, y_max)
            z_range: zの範囲 (z_min, z_max)
        """
        super().__init__()
        self.is_2d = False
        self.is_3d = True
        
        # 3D格子の基本パラメータ
        self.nx_points = nx_points
        self.ny_points = ny_points
        self.nz_points = nz_points
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        
        # 格子間隔の計算
        self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
        self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
        self.hz = (self.z_max - self.z_min) / (self.nz_points - 1)
        
        # 座標の生成
        self.x = np.linspace(self.x_min, self.x_max, self.nx_points)
        self.y = np.linspace(self.y_min, self.y_max, self.ny_points)
        self.z = np.linspace(self.z_min, self.z_max, self.nz_points)
        
        # メッシュグリッド作成（ベクトル計算に便利）
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # 後方互換性のため
        self.n_points = max(nx_points, ny_points, nz_points)
        self.h = self.hx
        
        # 境界識別用の定数
        self._MIN_BOUNDARY = 0
        self._MAX_BOUNDARY = 1
        self._INTERIOR = 2
        
        # 境界タイプ名のキャッシュ
        self._boundary_type_cache = {}
    
    def get_point(self, i, j=None, k=None):
        """
        格子点の座標を取得
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            (x, y, z)座標のタプル
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
            
        if (i < 0 or i >= self.nx_points or 
            j < 0 or j >= self.ny_points or 
            k < 0 or k >= self.nz_points):
            raise IndexError(f"インデックス ({i}, {j}, {k}) は範囲外です")
            
        return self.x[i], self.y[j], self.z[k]
    
    def get_points(self):
        """
        全格子点を取得
        
        Returns:
            (X, Y, Z)メッシュグリッドのタプル
        """
        return self.X, self.Y, self.Z
    
    def get_spacing(self):
        """
        格子間隔を取得
        
        Returns:
            (hx, hy, hz)格子間隔のタプル
        """
        return self.hx, self.hy, self.hz
    
    def get_index(self, i, j=None, k=None):
        """
        3Dインデックスを平坦化された1Dインデックスに変換
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            1Dインデックス
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        return i + j * self.nx_points + k * self.nx_points * self.ny_points
    
    def get_indices(self, flat_index):
        """
        平坦化された1Dインデックスを格子インデックスに変換
        
        Args:
            flat_index: 1Dインデックス
            
        Returns:
            (i, j, k)タプル
        """
        k = flat_index // (self.nx_points * self.ny_points)
        remainder = flat_index % (self.nx_points * self.ny_points)
        j = remainder // self.nx_points
        i = remainder % self.nx_points
        return i, j, k
    
    def _get_boundary_status(self, i, j, k):
        """
        点の境界ステータスを判定する内部メソッド
        
        Args:
            i, j, k: 格子インデックス
            
        Returns:
            各方向の境界ステータスを示す3要素のタプル (x_status, y_status, z_status)
            各ステータスは _MIN_BOUNDARY, _MAX_BOUNDARY, または _INTERIOR のいずれか
        """
        x_status = (self._MIN_BOUNDARY if i == 0 else 
                   (self._MAX_BOUNDARY if i == self.nx_points - 1 else 
                    self._INTERIOR))
        
        y_status = (self._MIN_BOUNDARY if j == 0 else 
                   (self._MAX_BOUNDARY if j == self.ny_points - 1 else 
                    self._INTERIOR))
        
        z_status = (self._MIN_BOUNDARY if k == 0 else 
                   (self._MAX_BOUNDARY if k == self.nz_points - 1 else 
                    self._INTERIOR))
        
        return (x_status, y_status, z_status)
    
    def is_boundary_point(self, i, j=None, k=None):
        """
        境界点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            点が境界上にあるかどうかを示すブール値
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        x_status, y_status, z_status = self._get_boundary_status(i, j, k)
        return x_status != self._INTERIOR or y_status != self._INTERIOR or z_status != self._INTERIOR
    
    def is_interior_point(self, i, j=None, k=None):
        """
        内部点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            点が内部にあるかどうかを示すブール値
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        x_status, y_status, z_status = self._get_boundary_status(i, j, k)
        return x_status == self._INTERIOR and y_status == self._INTERIOR and z_status == self._INTERIOR
    
    def is_face_point(self, i, j=None, k=None):
        """
        面上の点（辺や頂点を除く）かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            点が面上にあるかどうかを示すブール値
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        x_status, y_status, z_status = self._get_boundary_status(i, j, k)
        
        # 1つの方向のみが境界上（他の2方向は内部）である点が面点
        boundary_count = (x_status != self._INTERIOR) + (y_status != self._INTERIOR) + (z_status != self._INTERIOR)
        return boundary_count == 1
    
    def is_edge_point(self, i, j=None, k=None):
        """
        辺上の点（頂点を除く）かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            点が辺上にあるかどうかを示すブール値
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        x_status, y_status, z_status = self._get_boundary_status(i, j, k)
        
        # 2つの方向が境界上（残りの1方向は内部）である点が辺点
        boundary_count = (x_status != self._INTERIOR) + (y_status != self._INTERIOR) + (z_status != self._INTERIOR)
        return boundary_count == 2
    
    def is_vertex_point(self, i, j=None, k=None):
        """
        頂点かどうかをチェック
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            点が頂点かどうかを示すブール値
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        x_status, y_status, z_status = self._get_boundary_status(i, j, k)
        
        # 3つの方向全てが境界上である点が頂点
        return (x_status != self._INTERIOR and y_status != self._INTERIOR and z_status != self._INTERIOR)
    
    def get_boundary_type(self, i, j=None, k=None):
        """
        境界点のタイプを取得
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            境界タイプの文字列
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        # キャッシュをチェック
        cache_key = (i, j, k)
        if cache_key in self._boundary_type_cache:
            return self._boundary_type_cache[cache_key]
        
        # 内部点の場合
        if self.is_interior_point(i, j, k):
            self._boundary_type_cache[cache_key] = 'interior'
            return 'interior'
        
        # 境界ステータスを取得
        x_status, y_status, z_status = self._get_boundary_status(i, j, k)
        
        # 頂点の場合
        if self.is_vertex_point(i, j, k):
            x_part = 'x_min' if x_status == self._MIN_BOUNDARY else 'x_max'
            y_part = 'y_min' if y_status == self._MIN_BOUNDARY else 'y_max'
            z_part = 'z_min' if z_status == self._MIN_BOUNDARY else 'z_max'
            result = f'vertex_{x_part}_{y_part}_{z_part}'
            self._boundary_type_cache[cache_key] = result
            return result
        
        # 辺の場合
        if self.is_edge_point(i, j, k):
            # X方向の辺 (Y, Z方向が境界)
            if x_status == self._INTERIOR:
                y_part = 'y_min' if y_status == self._MIN_BOUNDARY else 'y_max'
                z_part = 'z_min' if z_status == self._MIN_BOUNDARY else 'z_max'
                result = f'edge_x_{y_part}_{z_part}'
            # Y方向の辺 (X, Z方向が境界)
            elif y_status == self._INTERIOR:
                x_part = 'x_min' if x_status == self._MIN_BOUNDARY else 'x_max'
                z_part = 'z_min' if z_status == self._MIN_BOUNDARY else 'z_max'
                result = f'edge_y_{x_part}_{z_part}'
            # Z方向の辺 (X, Y方向が境界)
            else:
                x_part = 'x_min' if x_status == self._MIN_BOUNDARY else 'x_max'
                y_part = 'y_min' if y_status == self._MIN_BOUNDARY else 'y_max'
                result = f'edge_z_{x_part}_{y_part}'
            
            self._boundary_type_cache[cache_key] = result
            return result
        
        # 面の場合
        if self.is_face_point(i, j, k):
            if x_status != self._INTERIOR:
                result = 'face_x_min' if x_status == self._MIN_BOUNDARY else 'face_x_max'
            elif y_status != self._INTERIOR:
                result = 'face_y_min' if y_status == self._MIN_BOUNDARY else 'face_y_max'
            else:
                result = 'face_z_min' if z_status == self._MIN_BOUNDARY else 'face_z_max'
            
            self._boundary_type_cache[cache_key] = result
            return result
        
        # 念のため
        self._boundary_type_cache[cache_key] = 'unknown'
        return 'unknown'