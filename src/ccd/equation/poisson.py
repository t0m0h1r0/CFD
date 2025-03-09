import cupy as cp
from .base1d import Equation

class PoissonEquation(Equation):
    """ポアソン方程式: psi''(x) = f(x)"""

    def __init__(self, f_func, grid=None):
        """ポアソン方程式を初期化
        
        Args:
            f_func: 右辺の関数 f(x)
            grid: 計算格子オブジェクト（オプション）
        """
        super().__init__(grid)
        self.f_func = f_func

    def get_stencil_coefficients(self, grid=None, i=None):
        """ステンシル係数を返す
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # ポアソン方程式の場合、ステンシル係数は位置に依存しないため
        # gridとiは実際には使用しない
        return {0: cp.array([0, 0, 1, 0])}

    def get_rhs(self, grid=None, i=None):
        """右辺関数f(x)の値
        
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
        """方程式が適用可能かどうかを判定
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # ポアソン方程式は全ての点で有効
        return True
    
from .base2d import Equation2D
class PoissonEquation2D(Equation2D):
    """2D Poisson equation: ψ_xx + ψ_yy = f(x,y)"""
    
    def __init__(self, f_func, grid=None):
        """
        Initialize with right-hand side function
        
        Args:
            f_func: Function f(x,y) for right-hand side
            grid: 計算格子オブジェクト（オプション）
        """
        super().__init__(grid)
        self.f_func = f_func
    
    def get_stencil_coefficients(self, grid=None, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        For 2D Poisson equation, we need coefficients for ψ_xx and ψ_yy
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
        
        Returns:
            Dictionary with stencil coefficients
        """
        # Indices in the unknown vector:
        # 0: ψ, 1: ψ_x, 2: ψ_xx, 3: ψ_xxx, 4: ψ_y, 5: ψ_yy, 6: ψ_yyy
        
        # Set coefficient 1.0 for ψ_xx (index 2) and 1.0 for ψ_yy (index 5)
        coeffs = {
            (0, 0): cp.array([0, 0, 1, 0, 0, 1, 0])
        }
        return coeffs
    
    def get_rhs(self, grid=None, i=None, j=None):
        """
        Get right-hand side value f(x,y) at point (i,j)
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            Right-hand side value
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
        Check if equation is valid at grid point (i,j)
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            Boolean indicating if equation is valid
        """
        # Poisson equation is valid at all points
        return True