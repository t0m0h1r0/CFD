import numpy as np
from .base import Equation

class OriginalEquation(Equation):
    """元の関数をそのまま使用する方程式"""
    
    def __init__(self, grid=None):
        """
        元の関数を使用する方程式を初期化
        
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
        方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # すべての点で有効
        return True