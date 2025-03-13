import cupy as cp
from .base1d import Equation
from .base2d import Equation2D

class OriginalEquation(Equation):
    """元の関数をそのまま使用する方程式"""
    
    def __init__(self, grid=None):
        """
        元の関数を使用する方程式を初期化
        
        Args:
            f_func: 関数 f(x)（Noneの場合はsolve時に設定）
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
        return {0: cp.array([1, 0, 0, 0])}

    def is_valid_at(self, i=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # すべての点で有効
        return True
    
    
class OriginalEquation2D(Equation2D):    
    """2次元の元の関数をそのまま使用する方程式"""
    
    def __init__(self, grid=None):
        """
        2次元の元の関数を使用する方程式を初期化
        
        Args:
            f_func: 関数 f(x, y)（Noneの場合はsolve時に設定）
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        ステンシル係数を返す
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = {
            (0, 0): cp.array([1, 0, 0, 0, 0, 0, 0])
        }
        return coeffs
    
    def is_valid_at(self, i=None, j=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
            
        return True