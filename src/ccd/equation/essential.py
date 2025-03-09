import cupy as cp
from .base1d import Equation

class EssentialEquation(Equation):
    """特定の未知数に対する基本方程式"""

    def __init__(self, k, f_func, grid=None):
        """
        基本方程式を初期化
        
        Args:
            k: 未知数のインデックス (0:ψ, 1:ψ', 2:ψ'', 3:ψ''')
            f_func: 右辺の関数 f(x)
            grid: 計算格子オブジェクト（オプション）
        """
        super().__init__(grid)
        if k not in [0, 1, 2, 3]:
            k = 0
        self.k = k
        self.f_func = f_func

    def get_stencil_coefficients(self, grid=None, i=None):
        """
        ステンシル係数を返す
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        coeffs = cp.zeros(4)
        coeffs[self.k] = 1.0
        return {0: coeffs}

    def get_rhs(self, grid=None, i=None):
        """
        右辺値を返す
        
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
        """
        方程式が有効かどうかを判定
        
        Args:
            grid: 計算格子（Noneの場合はself.gridを使用）
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # 全ての点で有効
        return True