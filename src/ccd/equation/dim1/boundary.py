import numpy as np
from .base import Equation

class DirichletBoundaryEquation(Equation):
    """ディリクレ境界条件: psi(x) = value"""

    def __init__(self, grid=None):
        """
        ディリクレ境界条件を初期化
        
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
        return {0: np.array([1, 0, 0, 0])}

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
        
    def get_equation_type(self):
        """
        ディリクレ境界条件の種類を返す

        Returns:
            str: "dirichlet"
        """
        return "dirichlet"


class NeumannBoundaryEquation(Equation):
    """ノイマン境界条件: psi'(x) = value"""

    def __init__(self, grid=None):
        """
        ノイマン境界条件を初期化
        
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
        return {0: np.array([0, 1, 0, 0])}

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
        
    def get_equation_type(self):
        """
        ノイマン境界条件の種類を返す

        Returns:
            str: "neumann"
        """
        return "neumann"