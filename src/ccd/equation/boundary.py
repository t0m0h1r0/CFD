import cupy as cp
from .base1d import Equation

class DirichletBoundaryEquation(Equation):
    """ディリクレ境界条件: psi(x) = value"""

    def __init__(self, value: float):
        self.value = value

    def get_stencil_coefficients(self, grid, i):
        return {0: cp.array([1, 0, 0, 0])}

    def get_rhs(self, grid, i):
        return self.value

    def is_valid_at(self, grid, i):
        n = grid.n_points
        return i == 0 or i == n - 1


class NeumannBoundaryEquation(Equation):
    """ノイマン境界条件: psi'(x) = value"""

    def __init__(self, value: float):
        self.value = value

    def get_stencil_coefficients(self, grid, i):
        return {0: cp.array([0, 1, 0, 0])}

    def get_rhs(self, grid, i):
        return self.value

    def is_valid_at(self, grid, i):
        n = grid.n_points
        return i == 0 or i == n - 1
    
import cupy as cp
from equation.base2d import Equation2D

class DirichletXBoundaryEquation2D(Equation2D):
    """
    X方向（左右境界）のディリクレ境界条件: ψ(x,y) = value
    """
    def __init__(self, value: cp.array):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
        """
        self.value = value
    
    def get_stencil_coefficients(self, grid, i, j):
        """
        格子点(i,j)におけるステンシル係数を取得
        """
        coeffs = {(0, 0): cp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        return coeffs
    
    def get_rhs(self, grid, i, j):
        """
        境界値を取得
        """
        if j < len(self.value):
            return self.value[j]
        return 0.0
    
    def is_valid_at(self, grid, i, j):
        """
        この境界条件が有効かどうかをチェック
        """
        return i == 0 or i == grid.nx_points - 1


class DirichletYBoundaryEquation2D(Equation2D):
    """
    Y方向（下上境界）のディリクレ境界条件: ψ(x,y) = value
    """
    def __init__(self, value: cp.array):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
        """
        self.value = value
    
    def get_stencil_coefficients(self, grid, i, j):
        """
        格子点(i,j)におけるステンシル係数を取得
        """
        coeffs = {(0, 0): cp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        return coeffs
    
    def get_rhs(self, grid, i, j):
        """
        境界値を取得
        """
        if i < len(self.value):
            return self.value[i]
        return 0.0
    
    def is_valid_at(self, grid, i, j):
        """
        この境界条件が有効かどうかをチェック
        """
        return j == 0 or j == grid.ny_points - 1

class NeumannXBoundaryEquation2D(Equation2D):
    """
    X方向（左右境界）のディリクレ境界条件: ψ(x,y) = value
    """
    def __init__(self, value: cp.array):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
        """
        self.value = value
    
    def get_stencil_coefficients(self, grid, i, j):
        """
        格子点(i,j)におけるステンシル係数を取得
        """
        coeffs = {(0, 0): cp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        return coeffs
    
    def get_rhs(self, grid, i, j):
        """
        境界値を取得
        """
        if j < len(self.value):
            return self.value[j]
        return 0.0
    
    def is_valid_at(self, grid, i, j):
        """
        この境界条件が有効かどうかをチェック
        """
        return i == 0 or i == grid.nx_points - 1


class NeumannYBoundaryEquation2D(Equation2D):
    """
    Y方向（下上境界）のディリクレ境界条件: ψ(x,y) = value
    """
    def __init__(self, value: cp.array):
        """
        初期化
        
        Args:
            value: 境界値を格納したCuPy配列
        """
        self.value = value
    
    def get_stencil_coefficients(self, grid, i, j):
        """
        格子点(i,j)におけるステンシル係数を取得
        """
        coeffs = {(0, 0): cp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])}
        return coeffs
    
    def get_rhs(self, grid, i, j):
        """
        境界値を取得
        """
        if i < len(self.value):
            return self.value[i]
        return 0.0
    
    def is_valid_at(self, grid, i, j):
        """
        この境界条件が有効かどうかをチェック
        """
        return j == 0 or j == grid.ny_points - 1
