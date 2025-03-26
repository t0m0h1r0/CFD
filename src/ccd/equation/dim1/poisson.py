import numpy as np
from .base import Equation

class PoissonEquation(Equation):
    """ポアソン方程式: psi''(x) = f(x)"""

    def __init__(self, grid=None):
        """ポアソン方程式を初期化
        
        Args:
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
        
    def get_equation_type(self):
        """
        ポアソン方程式の種類を返す - 支配方程式

        Returns:
            str: "governing"
        """
        return "governing"