import numpy as cp
from .base1d import Equation

class EssentialEquation(Equation):
    """特定の未知数に対する基本方程式"""

    def __init__(self, k, grid=None):
        """
        基本方程式を初期化
        
        Args:
            k: 未知数のインデックス (0:ψ, 1:ψ', 2:ψ'', 3:ψ''')
            f_func: 右辺の関数 f(x)
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
        if k not in [0, 1, 2, 3]:
            k = 0
        self.k = k

    def get_stencil_coefficients(self, i=None):
        """
        ステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = cp.zeros(4)
        coeffs[self.k] = 1.0
        return {0: coeffs}

    def is_valid_at(self, i=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # 全ての点で有効
        return True