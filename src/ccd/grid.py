import cupy as cp

class Grid:
    """統合計算格子クラス (1Dと2Dの機能を統合)"""
    
    def __init__(self, nx_points, ny_points=None, x_range=None, y_range=None):
        """
        計算格子を初期化 (1Dまたは2D)
        
        Args:
            nx_points: x方向の格子点数 (1Dの場合は総点数)
            ny_points: y方向の格子点数 (1Dの場合はNone)
            x_range: xの範囲 (x_min, x_max)
            y_range: yの範囲 (y_min, y_max) (1Dの場合はNone)
        """
        # 1Dか2Dかを判定
        self.is_2d = ny_points is not None
        
        if self.is_2d:
            # 2D格子の初期化
            self.nx_points = nx_points
            self.ny_points = ny_points
            self.x_min, self.x_max = x_range
            self.y_min, self.y_max = y_range
            
            self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
            self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
            
            self.x = cp.linspace(self.x_min, self.x_max, self.nx_points)
            self.y = cp.linspace(self.y_min, self.y_max, self.ny_points)
            
            # メッシュグリッド作成（ベクトル計算に便利）
            self.X, self.Y = cp.meshgrid(self.x, self.y, indexing='ij')
            
            # 後方互換性のため
            self.n_points = max(nx_points, ny_points)
            self.h = self.hx
        else:
            # 1D格子の初期化
            self.n_points = nx_points
            if x_range is None:
                raise ValueError("1D格子にはx_rangeを指定する必要があります")
            self.x_min, self.x_max = x_range
            self.h = (self.x_max - self.x_min) / (self.n_points - 1)
            self.x = cp.linspace(self.x_min, self.x_max, self.n_points)
            
            # 後方互換性のため
            self.nx_points = nx_points
            self.ny_points = 1
            self.hx = self.h
            self.hy = 1.0
    
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
        if self.is_2d:
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
        """
        if self.is_2d:
            return self.X, self.Y
        else:
            return self.x
    
    def get_spacing(self):
        """
        格子間隔を取得
        
        Returns:
            1D: hスカラー
            2D: (hx, hy)タプル
        """
        if self.is_2d:
            return self.hx, self.hy
        else:
            return self.h
    
    # 2D特有のメソッド（1D互換性チェック付き）
    def get_index(self, i, j=None):
        """
        2Dインデックスを平坦化された1Dインデックスに変換
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            1Dインデックス
        """
        if not self.is_2d:
            return i
        
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
        """
        if not self.is_2d:
            return flat_index
        
        j = flat_index // self.nx_points
        i = flat_index % self.nx_points
        return i, j
    
    def is_boundary_point(self, i, j=None):
        """
        境界点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            点が境界上にあるかどうかを示すブール値
        """
        if not self.is_2d:
            return i == 0 or i == self.n_points - 1
        
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return (i == 0 or i == self.nx_points - 1 or 
                j == 0 or j == self.ny_points - 1)
    
    def is_corner_point(self, i, j=None):
        """
        角点かどうかをチェック (2Dのみ)
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            点が角点かどうかを示すブール値
        """
        if not self.is_2d:
            return i == 0 or i == self.n_points - 1
        
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return ((i == 0 or i == self.nx_points - 1) and 
                (j == 0 or j == self.ny_points - 1))
    
    def is_interior_point(self, i, j=None):
        """
        内部点かどうかをチェック
        
        Args:
            i: x方向/行インデックス
            j: y方向/列インデックス (2Dの場合は必須)
            
        Returns:
            点が内部にあるかどうかを示すブール値
        """
        if not self.is_2d:
            return 0 < i < self.n_points - 1
        
        if j is None:
            raise ValueError("2D格子ではjインデックスを指定する必要があります")
        
        return (0 < i < self.nx_points - 1 and 
                0 < j < self.ny_points - 1)