import cupy as cp
from .base1d import Equation
from .base2d import Equation2D

class DirichletBoundaryEquation(Equation):
    """ディリクレ境界条件: psi(x) = value"""

    def __init__(self, grid=None):
        """
        ディリクレ境界条件を初期化
        
        Args:
            value: 境界値
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
        境界条件が有効かどうかを判定
        
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
        return i == 0 or i == n - 1


class NeumannBoundaryEquation(Equation):
    """ノイマン境界条件: psi'(x) = value"""

    def __init__(self, grid=None):
        """
        ノイマン境界条件を初期化
        
        Args:
            value: 境界値
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
        return {0: cp.array([0, 1, 0, 0])}

    def is_valid_at(self, i=None):
        """
        境界条件が有効かどうかを判定
        
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
        return i == 0 or i == n - 1


class DirichletXBoundaryEquation2D(Equation2D):
    """
    X方向（左右境界）のディリクレ境界条件: ψ(x,y) = value
    """
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        格子点(i,j)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = {(0, 0): cp.array([1, 0, 0, 0, 0, 0, 0])}
        return coeffs
        
    def is_valid_at(self, i=None, j=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None:
            raise ValueError("x方向のグリッド点インデックスiを指定する必要があります。")
            
        return i == 0 or i == self.grid.nx_points - 1


class DirichletYBoundaryEquation2D(Equation2D):
    """
    Y方向（下上境界）のディリクレ境界条件: ψ(x,y) = value
    """
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        格子点(i,j)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = {(0, 0): cp.array([1, 0, 0, 0, 0, 0, 0])}
        return coeffs
        
    def is_valid_at(self, i=None, j=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if j is None:
            raise ValueError("y方向のグリッド点インデックスjを指定する必要があります。")
            
        return j == 0 or j == self.grid.ny_points - 1

class NeumannXBoundaryEquation2D(Equation2D):
    """
    X方向（左右境界）のノイマン境界条件: ∂ψ/∂x(x,y) = value
    """
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        格子点(i,j)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = {(0, 0): cp.array([0, 1, 0, 0, 0, 0, 0])}
        return coeffs
        
    def is_valid_at(self, i=None, j=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None:
            raise ValueError("x方向のグリッド点インデックスiを指定する必要があります。")
            
        return i == 0 or i == self.grid.nx_points - 1


class NeumannYBoundaryEquation2D(Equation2D):
    """
    Y方向（下上境界）のノイマン境界条件: ∂ψ/∂y(x,y) = value
    """
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        格子点(i,j)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = {(0, 0): cp.array([0, 0, 0, 0, 1, 0, 0])}
        return coeffs
        
    def is_valid_at(self, i=None, j=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if j is None:
            raise ValueError("y方向のグリッド点インデックスjを指定する必要があります。")
            
        return j == 0 or j == self.grid.ny_points - 1