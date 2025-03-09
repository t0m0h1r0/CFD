# equation/custom.py
import cupy as cp
from typing import Dict, Callable, List
from grid1d import Grid
from .base1d import Equation


class CustomEquation(Equation):
    """カスタム方程式: ユーザー定義の係数による方程式"""

    def __init__(self, f_func: Callable[[float], float], coeff=[1,0,0,0], grid=None):
        """
        カスタム方程式の初期化

        Args:
            f_func: 右辺の関数 f(x)
            coeff: 未知変数の係数 [c0, c1, c2, c3] (それぞれψ, ψ', ψ'', ψ'''に対応)
            grid: 計算格子オブジェクト
        """
        super().__init__(grid)
        self.f_func = f_func
        self.coeff = coeff

    def get_stencil_coefficients(self, i=None):
        """
        方程式のステンシル係数を返す

        Args:
            i: グリッド点のインデックス

        Returns:
            ステンシル係数の辞書
        """
        # ステンシル係数（位置に依存しない）
        coeffs = {
            0: cp.array(self.coeff),
        }

        return coeffs

    def get_rhs(self, i=None):
        """
        右辺関数f(x)の値

        Args:
            i: グリッド点のインデックス

        Returns:
            右辺の値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
            
        # グリッド点の座標値を取得
        x = self.grid.get_point(i)
        return self.f_func(x)

    def is_valid_at(self, i=None):
        """
        方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        # カスタム方程式は全ての点で有効
        return True