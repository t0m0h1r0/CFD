import numpy as np
from equation.base3d import Equation3D

class DirichletBoundaryEquation3D(Equation3D):
    """
    3次元ディリクレ境界条件: ψ(x,y,z) = value
    """
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        格子点(i,j,k)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # ψ (インデックス0) に制約を設定
        coeffs = {(0, 0, 0): np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
        return coeffs
        
    def is_valid_at(self, i=None, j=None, k=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        # 3Dグリッドの境界上にあるかどうかをチェック
        return self.grid.is_boundary_point(i, j, k)


class NeumannBoundaryEquation3D(Equation3D):
    """
    3次元ノイマン境界条件（法線方向微分）: dψ/dn(x,y,z) = value
    """
    def __init__(self, direction='x', grid=None):
        """
        初期化
        
        Args:
            direction: 微分方向 ('x', 'y', 'z')
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
        self.direction = direction
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        格子点(i,j,k)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # 方向に応じた微分に制約を設定
        coeffs = np.zeros(10)
        
        if self.direction == 'x':
            # ψ_x (インデックス1) に制約を設定
            coeffs[1] = 1.0
        elif self.direction == 'y':
            # ψ_y (インデックス4) に制約を設定
            coeffs[4] = 1.0
        else:  # self.direction == 'z'
            # ψ_z (インデックス7) に制約を設定
            coeffs[7] = 1.0
            
        return {(0, 0, 0): coeffs}
        
    def is_valid_at(self, i=None, j=None, k=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        # 指定方向の境界上にあるかどうかをチェック
        if self.direction == 'x':
            return i == 0 or i == self.grid.nx_points - 1
        elif self.direction == 'y':
            return j == 0 or j == self.grid.ny_points - 1
        else:  # self.direction == 'z'
            return k == 0 or k == self.grid.nz_points - 1


class DirectionalNeumannBoundaryEquation3D(Equation3D):
    """
    3次元ノイマン境界条件（特定方向微分）: dψ/d{direction}(x,y,z) = value
    """
    def __init__(self, direction='x', grid=None):
        """
        初期化
        
        Args:
            direction: 微分方向 ('x', 'y', 'z')
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
        self.direction = direction
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        格子点(i,j,k)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # 方向に応じた微分に制約を設定
        coeffs = np.zeros(10)
        
        if self.direction == 'x':
            # ψ_x (インデックス1) に制約を設定
            coeffs[1] = 1.0
        elif self.direction == 'y':
            # ψ_y (インデックス4) に制約を設定
            coeffs[4] = 1.0
        else:  # self.direction == 'z'
            # ψ_z (インデックス7) に制約を設定
            coeffs[7] = 1.0
            
        return {(0, 0, 0): coeffs}
        
    def is_valid_at(self, i=None, j=None, k=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        # 任意境界点で有効（一般的に他の境界条件と組み合わせて使用）
        return self.grid.is_boundary_point(i, j, k)
