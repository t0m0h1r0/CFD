import numpy as np
from .base import Equation3D

class DirichletBoundaryEquation3D(Equation3D):
    """
    3次元ディリクレ境界条件: ψ(x,y,z) = value
    """
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        格子点(i,j,k)におけるステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # ψ (インデックス0) に制約を設定
        coeffs = {(0, 0, 0): np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
        return coeffs
        
    def is_valid_at(self, i=None, j=None, k=None):
        """
        この境界条件が有効かどうかをチェック
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        # 3Dグリッドの境界上にあるかどうかをチェック
        return self.grid.is_boundary_point(i, j, k)
        
    def get_equation_type(self):
        """
        3Dディリクレ境界条件の種類を返す

        Returns:
            str: "dirichlet"
        """
        return "dirichlet"