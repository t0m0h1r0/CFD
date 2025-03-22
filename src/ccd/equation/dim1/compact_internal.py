import numpy as np
from .base import Equation

class Internal1stDerivativeEquation(Equation):
    """内部点での1階導関数関係式"""

    def __init__(self, grid=None):
        """
        内部点での1階導関数関係式を初期化
        
        Args:
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None):
        """
        ステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        h = self.grid.get_spacing()
        coeffs = {
            -1: np.array([35/32, 19/32, 1/8, 1/96]) * np.array([h**-1, h**0, h**1, h**2]),
            0: np.array([0, 1, 0, 0]) * np.array([h**-1, h**0, h**1, h**2]),
            1: np.array([-35/32, 19/32, -1/8, 1/96]) * np.array([h**-1, h**0, h**1, h**2])
        }
        return coeffs

    def is_valid_at(self, i=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        n = self.grid.n_points
        return 0 < i < n - 1


class Internal2ndDerivativeEquation(Equation):
    """内部点での2階導関数関係式"""

    def __init__(self, grid=None):
        """
        内部点での2階導関数関係式を初期化
        
        Args:
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None):
        """
        ステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        h = self.grid.get_spacing()
        coeffs = {
            -1: np.array([-4, -29/16, -5/16, -1/48]) * np.array([h**-2, h**-1, h**0, h**1]),
            0: np.array([8, 0, 1, 0]) * np.array([h**-2, h**-1, h**0, h**1]),
            1: np.array([-4, 29/16, -5/16, 1/48]) * np.array([h**-2, h**-1, h**0, h**1])
        }
        return coeffs

    def get_rhs(self, i=None):
        """
        右辺値を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, i=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        n = self.grid.n_points
        return 0 < i < n - 1


class Internal3rdDerivativeEquation(Equation):
    """内部点での3階導関数関係式"""

    def __init__(self, grid=None):
        """
        内部点での3階導関数関係式を初期化
        
        Args:
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None):
        """
        ステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        h = self.grid.get_spacing()
        coeffs = {
            -1: np.array([-105/16, -105/16, -15/8, -3/16]) * np.array([h**-3, h**-2, h**-1, h**0]),
            0: np.array([0, 0, 0, 1]) * np.array([h**-3, h**-2, h**-1, h**0]),
            1: np.array([105/16, -105/16, 15/8, -3/16]) * np.array([h**-3, h**-2, h**-1, h**0])
        }
        return coeffs

    def get_rhs(self, i=None):
        """
        右辺値を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, i=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        n = self.grid.n_points
        return 0 < i < n - 1