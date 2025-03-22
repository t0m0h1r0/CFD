import numpy as np
from .base import Equation2D

class DirichletBoundaryEquation2D(Equation2D):
    """
    2次元ディリクレ境界条件: ψ(x,y) = value
    方向を指定可能な統一された境界条件クラス
    """
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
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
        # ψ (インデックス0) に制約を設定
        coeffs = {(0, 0): np.array([1, 0, 0, 0, 0, 0, 0])}
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
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        return (i == 0 or i == self.grid.nx_points - 1 or 
                j == 0 or j == self.grid.ny_points - 1)