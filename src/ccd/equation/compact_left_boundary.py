import cupy as cp
from .base1d import Equation

class LeftBoundary1stDerivativeEquation(Equation):
    """左境界点での1階導関数関係式"""

    def __init__(self, grid=None):
        """
        左境界点での1階導関数関係式を初期化
        
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
            0: cp.array([9/2, 1, 0, 0]) * cp.array([h**-1, h**0, h**1, h**2]),
            1: cp.array([-4, 4, -1, 1/3]) * cp.array([h**-1, h**0, h**1, h**2]),
            2: cp.array([-(1/2), 0, 0, 0]) * cp.array([h**-1, h**0, h**1, h**2])
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
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        return i == 0


class LeftBoundary2ndDerivativeEquation(Equation):
    """左境界点での2階導関数関係式"""

    def __init__(self, grid=None):
        """
        左境界点での2階導関数関係式を初期化
        
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
            0: cp.array([-16, 0, 1, 0]) * cp.array([h**-2, h**-1, h**0, h**1]),
            1: cp.array([12, -20, 5, -(7/3)]) * cp.array([h**-2, h**-1, h**0, h**1]),
            2: cp.array([4, 0, 0, 0]) * cp.array([h**-2, h**-1, h**0, h**1])
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
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        return i == 0


class LeftBoundary3rdDerivativeEquation(Equation):
    """左境界点での3階導関数関係式"""

    def __init__(self, grid=None):
        """
        左境界点での3階導関数関係式を初期化
        
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
            0: cp.array([42, 0, 0, 1]) * cp.array([h**-3, h**-2, h**-1, h**0]),
            1: cp.array([-24, 60, -12, 9]) * cp.array([h**-3, h**-2, h**-1, h**0]),
            2: cp.array([-18, 0, 0, 0]) * cp.array([h**-3, h**-2, h**-1, h**0])
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
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        return i == 0