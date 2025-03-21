import numpy as np

class Grid:
    """統合計算格子クラス (1D, 2D, 3Dの機能を統合、CPU最適化版)"""
    
    def __init__(self, nx_points, ny_points=None, nz_points=None, x_range=None, y_range=None, z_range=None):
        """
        計算格子を初期化 (1D, 2D, または 3D)
        
        Args:
            nx_points: x方向の格子点数 (1Dの場合は総点数)
            ny_points: y方向の格子点数 (1Dの場合はNone)
            nz_points: z方向の格子点数 (1D/2Dの場合はNone)
            x_range: xの範囲 (x_min, x_max)
            y_range: yの範囲 (y_min, y_max) (1Dの場合はNone)
            z_range: zの範囲 (z_min, z_max) (1D/2Dの場合はNone)
        """
        # 1D, 2D, 3Dかを判定
        self.is_2d = ny_points is not None
        self.is_3d = nz_points is not None
        
        if self.is_3d:
            # 3D格子の初期化
            self.nx_points = nx_points
            self.ny_points = ny_points
            self.nz_points = nz_points
            self.x_min, self.x_max = x_range
            self.y_min, self.y_max = y_range
            self.z_min, self.z_max = z_range
            
            self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
            self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
            self.hz = (self.z_max - self.z_min) / (self.nz_points - 1)
            
            self.x = np.linspace(self.x_min, self.x_max, self.nx_points)
            self.y = np.linspace(self.y_min, self.y_max, self.ny_points)
            self.z = np.linspace(self.z_min, self.z_max, self.nz_points)
            
            # メッシュグリッド作成（ベクトル計算に便利）
            self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
            
            # 後方互換性のため
            self.n_points = max(nx_points, ny_points, nz_points)
            self.h = self.hx
        elif self.is_2d:
            # 2D格子の初期化
            self.nx_points = nx_points
            self.ny_points = ny_points
            self.x_min, self.x_max = x_range
            self.y_min, self.y_max = y_range
            
            self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
            self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
            
            self.x = np.linspace(self.x_min, self.x_max, self.nx_points)
            self.y = np.linspace(self.y_min, self.y_max, self.ny_points)
            
            # メッシュグリッド作成（ベクトル計算に便利）
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
            
            # 後方互換性のため
            self.n_points = max(nx_points, ny_points)
            self.h = self.hx
            
            # 3D互換性のため
            self.nz_points = 1
            self.hz = 1.0
            self.z = np.array([0.0])
            self.Z = None
        else:
            # 1D格子の初期化
            self.n_points = nx_points
            if x_range is None:
                raise ValueError("1D格子にはx_rangeを指定する必要があります")
            self.x_min, self.x_max = x_range
            self.h = (self.x_max - self.x_min) / (self.n_points - 1)
            self.x = np.linspace(self.x_min, self.x_max, self.n_points)
            
            # 後方互換性のため
            self.nx_points = nx_points
            self.ny_points = 1
            self.nz_points = 1
            self.hx = self.h
            self.hy = 1.0
            self.hz = 1.0
            self.y = np.array([0.0])
            self.z = np.array([0.0])
            self.Y = None
            self.Z = None
    
    def get_point(self, i, j=None, k=None):
        """
        格子点の座標を取得
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (1Dの場合はNone)
            k: z方向/層インデックス (1D/2Dの場合はNone)
            
        Returns:
            1D: x座標
            2D: (x, y)座標のタプル
            3D: (x, y, z)座標のタプル
        """
        if self.is_3d:
            if j is None or k is None:
                raise ValueError("3D格子ではjとkインデックスを指定する必要があります")
            return self.x[i], self.y[j], self.z[k]
        elif self.is_2d:
            if j is None:
                raise ValueError("2D格子ではjインデックスを指定する必要があります")
            return self.x[i], self.y[j]
        else:
            return self.x[i]
    
    def get_points(self):
        """
        全格子点を取得
        
        Returns:
            1D: x座標の配列
            2D: (X, Y)メッシュグリッドのタプル
            3D: (X, Y, Z)メッシュグリッドのタプル
        """
        if self.is_3d:
            return self.X, self.Y, self.Z
        elif self.is_2d:
            return self.X, self.Y
        else:
            return self.x
    
    def get_spacing(self):
        """
        格子間隔を取得
        
        Returns:
            1D: hスカラー
            2D: (hx, hy)タプル
            3D: (hx, hy, hz)タプル
        """
        if self.is_3d:
            return self.hx, self.hy, self.hz
        elif self.is_2d:
            return self.hx, self.hy
        else:
            return self.h
    
    # 多次元特有のメソッド（1D互換性チェック付き）
    def get_index(self, i, j=None, k=None):
        """
        多次元インデックスを平坦化された1Dインデックスに変換
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (1D以外は必須)
            k: z方向/層インデックス (3Dの場合は必須)
            
        Returns:
            1Dインデックス
        """
        if not self.is_2d and not self.is_3d:
            return i
        
        if self.is_3d:
            if j is None or k is None:
                raise ValueError("3D格子ではjとkインデックスを指定する必要があります")
            return i + j * self.nx_points + k * self.nx_points * self.ny_points
        
        # 2Dの場合
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return i + j * self.nx_points
    
    def get_indices(self, flat_index):
        """
        平坦化された1Dインデックスを格子インデックスに変換
        
        Args:
            flat_index: 1Dインデックス
            
        Returns:
            1D: iインデックス
            2D: (i, j)タプル
            3D: (i, j, k)タプル
        """
        if not self.is_2d and not self.is_3d:
            return flat_index
        
        if self.is_3d:
            k = flat_index // (self.nx_points * self.ny_points)
            remainder = flat_index % (self.nx_points * self.ny_points)
            j = remainder // self.nx_points
            i = remainder % self.nx_points
            return i, j, k
        
        # 2Dの場合
        j = flat_index // self.nx_points
        i = flat_index % self.nx_points
        return i, j
    
    def is_boundary_point(self, i, j=None, k=None):
        """
        境界点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (1D以外は必須)
            k: z方向/層インデックス (3Dの場合は必須)
            
        Returns:
            点が境界上にあるかどうかを示すブール値
        """
        if self.is_3d:
            if j is None or k is None:
                raise ValueError("3D格子ではjとkインデックスを指定する必要があります")
            
            return (i == 0 or i == self.nx_points - 1 or 
                    j == 0 or j == self.ny_points - 1 or
                    k == 0 or k == self.nz_points - 1)
        elif self.is_2d:
            if j is None:
                raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
            return (i == 0 or i == self.nx_points - 1 or 
                    j == 0 or j == self.ny_points - 1)
        else:
            return i == 0 or i == self.n_points - 1
    
    def is_edge_point(self, i, j=None, k=None):
        """
        エッジ点かどうかをチェック (2D, 3Dのみ)
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (必須)
            k: z方向/層インデックス (3Dの場合は必須)
            
        Returns:
            点がエッジ上にあるかどうかを示すブール値
        """
        if not self.is_2d and not self.is_3d:
            return self.is_boundary_point(i)
            
        if self.is_3d:
            if j is None or k is None:
                raise ValueError("3D格子ではjとkインデックスを指定する必要があります")
                
            # 頂点以外のエッジを判定
            return ((i == 0 or i == self.nx_points - 1) and 
                    (j == 0 or j == self.ny_points - 1) and
                    0 < k < self.nz_points - 1) or \
                   ((i == 0 or i == self.nx_points - 1) and 
                    0 < j < self.ny_points - 1 and
                    (k == 0 or k == self.nz_points - 1)) or \
                   (0 < i < self.nx_points - 1 and 
                    (j == 0 or j == self.ny_points - 1) and
                    (k == 0 or k == self.nz_points - 1))
        else:  # 2D
            if j is None:
                raise ValueError("2D格子ではjインデックスを指定する必要があります")
                
            # 角点以外の境界点を判定
            return (((i == 0 or i == self.nx_points - 1) and 0 < j < self.ny_points - 1) or
                    (0 < i < self.nx_points - 1 and (j == 0 or j == self.ny_points - 1)))
    
    def is_corner_point(self, i, j=None, k=None):
        """
        角点/頂点かどうかをチェック (2D/3Dのみ)
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (1D以外は必須)
            k: z方向/層インデックス (3Dの場合は必須)
            
        Returns:
            点が角点/頂点かどうかを示すブール値
        """
        if not self.is_2d and not self.is_3d:
            return i == 0 or i == self.n_points - 1
        
        if self.is_3d:
            if j is None or k is None:
                raise ValueError("3D格子ではjとkインデックスを指定する必要があります")
            
            return ((i == 0 or i == self.nx_points - 1) and 
                    (j == 0 or j == self.ny_points - 1) and 
                    (k == 0 or k == self.nz_points - 1))
        else:  # 2D
            if j is None:
                raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
            return ((i == 0 or i == self.nx_points - 1) and 
                    (j == 0 or j == self.ny_points - 1))
    
    def is_face_point(self, i, j=None, k=None):
        """
        面上の点かどうかをチェック (3Dのみ)
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス
            k: z方向/層インデックス
            
        Returns:
            点が面上にあるかどうかを示すブール値
        """
        if not self.is_3d:
            # 3D以外では意味がないのでFalseを返す
            return False
            
        if j is None or k is None:
            raise ValueError("3D格子ではjとkインデックスを指定する必要があります")
            
        # エッジや頂点を除く境界点
        is_x_face = ((i == 0 or i == self.nx_points - 1) and 
                      0 < j < self.ny_points - 1 and 
                      0 < k < self.nz_points - 1)
        is_y_face = (0 < i < self.nx_points - 1 and 
                     (j == 0 or j == self.ny_points - 1) and 
                     0 < k < self.nz_points - 1)
        is_z_face = (0 < i < self.nx_points - 1 and 
                     0 < j < self.ny_points - 1 and 
                     (k == 0 or k == self.nz_points - 1))
        
        return is_x_face or is_y_face or is_z_face
    
    def is_interior_point(self, i, j=None, k=None):
        """
        内部点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (1D以外は必須)
            k: z方向/層インデックス (3Dの場合は必須)
            
        Returns:
            点が内部にあるかどうかを示すブール値
        """
        if self.is_3d:
            if j is None or k is None:
                raise ValueError("3D格子ではjとkインデックスを指定する必要があります")
            
            return (0 < i < self.nx_points - 1 and 
                    0 < j < self.ny_points - 1 and 
                    0 < k < self.nz_points - 1)
        elif self.is_2d:
            if j is None:
                raise ValueError("2D格子ではjインデックスを指定する必要があります")
            
            return (0 < i < self.nx_points - 1 and 
                    0 < j < self.ny_points - 1)
        else:
            return 0 < i < self.n_points - 1