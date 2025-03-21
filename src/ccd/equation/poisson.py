import numpy as np
from .base1d import Equation
from .base2d import Equation2D
from .base3d import Equation3D

class PoissonEquation(Equation):
    """ポアソン方程式: psi''(x) = f(x)"""

    def __init__(self, grid=None):
        """ポアソン方程式を初期化
        
        Args:
            f_func: 右辺の関数 f(x)（Noneの場合はsolve時に設定）
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)

    def get_stencil_coefficients(self, i=None):
        """ステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # ポアソン方程式の場合、ステンシル係数は位置に依存しないため
        # iは実際には使用しない
        return {0: np.array([0, 0, 1, 0])}

    def is_valid_at(self, i=None):
        """方程式が適用可能かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # ポアソン方程式は全ての点で有効
        return True


class PoissonEquation2D(Equation2D):
    """2D Poisson equation: ψ_xx + ψ_yy = f(x,y)"""
    
    def __init__(self, grid=None):
        """
        Initialize with right-hand side function
        
        Args:
            f_func: Function f(x,y) for right-hand side（Noneの場合はsolve時に設定）
            grid: Grid2D object
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        Get stencil coefficients at grid point (i,j)
        
        For 2D Poisson equation, we need coefficients for ψ_xx and ψ_yy
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
        
        Returns:
            Dictionary with stencil coefficients
        """
        # Indices in the unknown vector:
        # 0: ψ, 1: ψ_x, 2: ψ_xx, 3: ψ_xxx, 4: ψ_y, 5: ψ_yy, 6: ψ_yyy
        
        # Set coefficient 1.0 for ψ_xx (index 2) and 1.0 for ψ_yy (index 5)
        coeffs = {
            (0, 0): np.array([0, 0, 1, 0, 0, 1, 0])
        }
        return coeffs
        
    def is_valid_at(self, i=None, j=None):
        """
        Check if equation is valid at grid point (i,j)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            
        Returns:
            Boolean indicating if equation is valid
        """
        # Poisson equation is valid at all points
        return True

class PoissonEquation3D(Equation3D):
    """3D Poisson equation: ψ_xx + ψ_yy + ψ_zz = f(x,y,z)"""
    
    def __init__(self, grid=None):
        """
        Initialize with right-hand side function
        
        Args:
            f_func: Function f(x,y,z) for right-hand side（Noneの場合はsolve時に設定）
            grid: Grid object with 3D support
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        Get stencil coefficients at grid point (i,j,k)
        
        For 3D Poisson equation, we need coefficients for ψ_xx, ψ_yy, and ψ_zz
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
        
        Returns:
            Dictionary with stencil coefficients
        """
        # Indices in the unknown vector:
        # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        #  0   1    2     3      4    5     6      7    8     9
        
        # Set coefficient 1.0 for ψ_xx (index 2), ψ_yy (index 5), and ψ_zz (index 8)
        coeffs = {
            (0, 0, 0): np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        }
        return coeffs
        
    def is_valid_at(self, i=None, j=None, k=None):
        """
        Check if equation is valid at grid point (i,j,k)
        
        Args:
            i: x-direction grid index
            j: y-direction grid index
            k: z-direction grid index
            
        Returns:
            Boolean indicating if equation is valid
        """
        # Poisson equation is valid at all points
        return True