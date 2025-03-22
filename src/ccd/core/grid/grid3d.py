"""
高精度コンパクト差分法 (CCD) の3次元計算格子

このモジュールは、CCDソルバーで使用する3次元計算格子のクラスを提供します。
"""

import numpy as np

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
        
        return (i == 0 or i == self.nx_points - 1 or 
                j == 0 or j == self.ny_points - 1 or
                k == 0 or k == self.nz_points - 1)
    
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
        
        return (0 < i < self.nx_points - 1 and 
                0 < j < self.ny_points - 1 and
                0 < k < self.nz_points - 1)
    
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
            
        x_face = (i == 0 or i == self.nx_points - 1) and (0 < j < self.ny_points - 1) and (0 < k < self.nz_points - 1)
        y_face = (j == 0 or j == self.ny_points - 1) and (0 < i < self.nx_points - 1) and (0 < k < self.nz_points - 1)
        z_face = (k == 0 or k == self.nz_points - 1) and (0 < i < self.nx_points - 1) and (0 < j < self.ny_points - 1)
        
        return x_face or y_face or z_face
    
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
            
        x_edge = (0 < i < self.nx_points - 1) and ((j == 0 and k == 0) or 
                                               (j == 0 and k == self.nz_points - 1) or 
                                               (j == self.ny_points - 1 and k == 0) or 
                                               (j == self.ny_points - 1 and k == self.nz_points - 1))
                                                
        y_edge = (0 < j < self.ny_points - 1) and ((i == 0 and k == 0) or 
                                               (i == 0 and k == self.nz_points - 1) or 
                                               (i == self.nx_points - 1 and k == 0) or 
                                               (i == self.nx_points - 1 and k == self.nz_points - 1))
                                                
        z_edge = (0 < k < self.nz_points - 1) and ((i == 0 and j == 0) or 
                                               (i == 0 and j == self.ny_points - 1) or 
                                               (i == self.nx_points - 1 and j == 0) or 
                                               (i == self.nx_points - 1 and j == self.ny_points - 1))
                                                
        return x_edge or y_edge or z_edge
    
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
        
        return ((i == 0 or i == self.nx_points - 1) and 
                (j == 0 or j == self.ny_points - 1) and
                (k == 0 or k == self.nz_points - 1))
    
    def get_boundary_type(self, i, j=None, k=None):
        """
        境界点のタイプを取得
        
        Args:
            i: x方向インデックス
            j: y方向インデックス
            k: z方向インデックス
            
        Returns:
            境界タイプの文字列
            'interior': 内部点
            'face_x_min', 'face_x_max', 'face_y_min', 'face_y_max', 'face_z_min', 'face_z_max': 面
            'edge_x_*', 'edge_y_*', 'edge_z_*': 辺 (主要方向_面1_面2)
            'vertex_x_min_y_min_z_min' など: 頂点
        """
        if j is None or k is None:
            raise ValueError("3D格子ではj, kインデックスを指定する必要があります")
        
        # 内部点
        if self.is_interior_point(i, j, k):
            return 'interior'
        
        # 頂点
        if self.is_vertex_point(i, j, k):
            x_suffix = '_x_min' if i == 0 else '_x_max'
            y_suffix = '_y_min' if j == 0 else '_y_max'
            z_suffix = '_z_min' if k == 0 else '_z_max'
            return 'vertex' + x_suffix + y_suffix + z_suffix
        
        # 辺 - 新しい命名規則に従ってフォーマット
        if self.is_edge_point(i, j, k):
            # エッジの方向を特定
            if 0 < i < self.nx_points - 1:  # x方向に沿ったエッジ
                prefix = 'edge_x'
                y_suffix = '_y_min' if j == 0 else '_y_max'
                z_suffix = '_z_min' if k == 0 else '_z_max'
                return prefix + y_suffix + z_suffix
            elif 0 < j < self.ny_points - 1:  # y方向に沿ったエッジ
                prefix = 'edge_y'
                x_suffix = '_x_min' if i == 0 else '_x_max'
                z_suffix = '_z_min' if k == 0 else '_z_max'
                return prefix + x_suffix + z_suffix
            elif 0 < k < self.nz_points - 1:  # z方向に沿ったエッジ
                prefix = 'edge_z'
                x_suffix = '_x_min' if i == 0 else '_x_max'
                y_suffix = '_y_min' if j == 0 else '_y_max'
                return prefix + x_suffix + y_suffix
        
        # 面
        if i == 0:
            return 'face_x_min'
        elif i == self.nx_points - 1:
            return 'face_x_max'
        elif j == 0:
            return 'face_y_min'
        elif j == self.ny_points - 1:
            return 'face_y_max'
        elif k == 0:
            return 'face_z_min'
        elif k == self.nz_points - 1:
            return 'face_z_max'
        
        # 念のため
        return 'unknown'