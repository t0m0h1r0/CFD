# equation/compact_internal.py
import cupy as cp
from typing import Dict
from grid import Grid
from .base import Equation


class Internal1stDerivativeEquation(Equation):
    """内部点での1階導関数関係式"""

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        1階導関数関係式のステンシル係数を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # グリッド間隔を取得
        h = grid.get_spacing()
        
        # 既存コードから流用した内部点の係数
        coeffs = {
            -1: cp.array([35 / 32, 19 / 32, 1 / 8, 1 / 96]),  # 左隣接点
            0: cp.array([0, 1, 0, 0]),  # 中央点
            1: cp.array([-35 / 32, 19 / 32, -1 / 8, 1 / 96]),  # 右隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * cp.array([h**-1, h**0, h**1, h**2])

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        1階導関数関係式の右辺を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        内部点でのみ有効
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            内部点の場合True
        """
        n = grid.n_points
        return 0 < i < n - 1


class Internal2ndDerivativeEquation(Equation):
    """内部点での2階導関数関係式"""

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        2階導関数関係式のステンシル係数を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # グリッド間隔を取得
        h = grid.get_spacing()
        
        # 既存コードから流用した内部点の係数
        coeffs = {
            -1: cp.array([-4, -29 / 16, -5 / 16, -1 / 48]),  # 左隣接点
            0: cp.array([8, 0, 1, 0]),  # 中央点
            1: cp.array([-4, 29 / 16, -5 / 16, 1 / 48]),  # 右隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * cp.array([h**-2, h**-1, h**0, h**1])

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        2階導関数関係式の右辺を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        内部点でのみ有効
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            内部点の場合True
        """
        n = grid.n_points
        return 0 < i < n - 1


class Internal3rdDerivativeEquation(Equation):
    """内部点での3階導関数関係式"""

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        3階導関数関係式のステンシル係数を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # グリッド間隔を取得
        h = grid.get_spacing()
        
        # 既存コードから流用した内部点の係数
        coeffs = {
            -1: cp.array([-105 / 16, -105 / 16, -15 / 8, -3 / 16]),  # 左隣接点
            0: cp.array([0, 0, 0, 1]),  # 中央点
            1: cp.array([105 / 16, -105 / 16, 15 / 8, -3 / 16]),  # 右隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * cp.array([h**-3, h**-2, h**-1, h**0])

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        3階導関数関係式の右辺を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        内部点でのみ有効
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            内部点の場合True
        """
        n = grid.n_points
        return 0 < i < n - 1