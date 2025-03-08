# equation/compact_left_boundary.py
import cupy as cp
from typing import Dict
from grid import Grid
from .base import Equation


class LeftBoundary1stDerivativeEquation(Equation):
    """左境界点での1階導関数関係式"""

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        左境界点の1階導関数関係式のステンシル係数を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            ステンシル係数の辞書
        """
        # グリッド間隔を取得
        h = grid.get_spacing()

        # 既存コードから流用した左境界点の係数
        coeffs = {
            0: cp.array([9/2,1,0,0]),  # 境界点
            1: cp.array([-4,4,-1,1/3]),  # 第1隣接点
            2: cp.array([-(1/2),0,0,0]),  # 第2隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * cp.array([h**-1, h**0, h**1, h**2])

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        左境界点の1階導関数関係式の右辺を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        左境界でのみ有効かを判定

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            左境界の場合True
        """
        return i == 0


class LeftBoundary2ndDerivativeEquation(Equation):
    """左境界点での2階導関数関係式"""

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        左境界点の2階導関数関係式のステンシル係数を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            ステンシル係数の辞書
        """
        # グリッド間隔を取得
        h = grid.get_spacing()

        # 既存コードから流用した左境界点の係数
        coeffs = {
            0: cp.array([-16,0,1,0]),  # 境界点
            1: cp.array([12,-20,5,-(7/3)]),  # 第1隣接点
            2: cp.array([4,0,0,0]),  # 第2隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * cp.array([h**-2, h**-1, h**0, h**1])

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        左境界点の2階導関数関係式の右辺を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        左境界でのみ有効かを判定

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            左境界の場合True
        """
        return i == 0


class LeftBoundary3rdDerivativeEquation(Equation):
    """左境界点での3階導関数関係式"""

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        左境界点の3階導関数関係式のステンシル係数を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            ステンシル係数の辞書
        """
        # グリッド間隔を取得
        h = grid.get_spacing()

        # 既存コードから流用した左境界点の係数
        coeffs = {
            0: cp.array([42,0,0,1]),  # 境界点
            1: cp.array([-24,60,-12,9]),  # 第1隣接点
            2: cp.array([-18,0,0,0]),  # 第2隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * cp.array([h**-3, h**-2, h**-1, h**0])

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        左境界点の3階導関数関係式の右辺を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            右辺の値
        """
        return 0.0

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        左境界でのみ有効かを判定

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            左境界の場合True
        """
        return i == 0
