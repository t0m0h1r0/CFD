import cupy as cp
from .base1d import Equation
from .base2d import Equation2D

class OriginalEquation(Equation):
    """元の関数をそのまま使用する方程式"""
    
    def __init__(self, f_func, grid=None):
        """
        元の関数を使用する方程式を初期化
        
        Args:
            f_func: 関数 f(x)
            grid: 計算格子オブジェクト（オプション）
        """
        super().__init__(grid)
        self.f_func = f_func

    def get_stencil_coefficients(self, grid=None, i=None):
        """
        ステンシル係数を返す
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        return {0: cp.array([1, 0, 0, 0])}

    def get_rhs(self, grid=None, i=None):
        """
        右辺値を返す
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: グリッド点のインデックス
            
        Returns:
            右辺の値
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
            
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        x = using_grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, grid=None, i=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # すべての点で有効
        return True
    
    
class OriginalEquation2D(Equation2D):    
    """2次元の元の関数をそのまま使用する方程式"""
    
    def __init__(self, f_func, grid=None):
        """
        2次元の元の関数を使用する方程式を初期化
        
        Args:
            f_func: 関数 f(x, y)
            grid: 計算格子オブジェクト（オプション）
        """
        super().__init__(grid)
        self.f_func = f_func
    
    def get_stencil_coefficients(self, grid=None, i=None, j=None):
        """
        ステンシル係数を返す
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = {
            (0, 0): cp.array([1, 0, 0, 0, 0, 0, 0])
        }
        return coeffs
    
    def get_rhs(self, grid=None, i=None, j=None):
        """
        右辺値を返す
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            右辺の値
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
            
        x, y = using_grid.get_point(i, j)
        return self.f_func(x, y)
    
    def is_valid_at(self, grid=None, i=None, j=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
            
        return using_grid.is_interior_point(i, j)