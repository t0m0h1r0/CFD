# equation/compact.py
import cupy as np
from typing import Tuple
from .base import Equation


class InternalCompactDifferenceEquation(Equation):
    """内部点でのコンパクト差分式"""

    def get_coefficients(
        self, i: int, n: int, h: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        # 既存コードから流用した内部点の係数
        a = np.array([35 / 32, 19 / 32, 1 / 8, 1 / 96])  # 左隣接点
        b = np.array([-1, 0, 1, 0])  # 中央点
        c = np.array([-35 / 32, 19 / 32, -1 / 8, 1 / 96])  # 右隣接点

        # スケール調整（グリッド幅に応じて調整）
        a_scaled = a * np.array([1, h**-1, h**-2, h**-3])
        b_scaled = b * np.array([1, h**-1, h**-2, h**-3])
        c_scaled = c * np.array([1, h**-1, h**-2, h**-3])

        F = 0.0  # 右辺は常に0

        return a_scaled, b_scaled, c_scaled, F

    def is_valid_at(self, i: int, n: int) -> bool:
        """内部点でのみ有効"""
        return 0 < i < n - 1


class LeftBoundaryCompactDifferenceEquation(Equation):
    """左境界でのコンパクト差分式"""

    def get_coefficients(
        self, i: int, n: int, h: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        # 既存コードから流用した左境界点の係数
        b = np.array([11 / 2, 1, 0, 0])  # 境界点
        c1 = np.array([24, 24, 4, 4 / 3])  # 隣接点1
        c2 = np.array([-59 / 2, 10, -1, 0])  # 隣接点2

        # スケール調整
        b_scaled = b * np.array([1, h**-1, h**-2, h**-3])
        c1_scaled = c1 * np.array([1, h**-1, h**-2, h**-3])
        c2_scaled = c2 * np.array([1, h**-1, h**-2, h**-3])

        a = np.zeros(4)  # 左境界なのでa = 0
        c = c1_scaled  # 簡略化のため、c1のみ使用

        F = 0.0

        return a, b_scaled, c, F

    def is_valid_at(self, i: int, n: int) -> bool:
        """左境界でのみ有効"""
        return i == 0


class RightBoundaryCompactDifferenceEquation(Equation):
    """右境界でのコンパクト差分式"""

    def get_coefficients(
        self, i: int, n: int, h: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        # 既存コードから流用した右境界点の係数
        a1 = np.array([-24, 24, -4, 4 / 3])  # 隣接点-1
        a2 = np.array([59 / 2, 10, 1, 0])  # 隣接点-2
        b = np.array([-11 / 2, 1, 0, 0])  # 境界点

        # スケール調整
        a1_scaled = a1 * np.array([1, h**-1, h**-2, h**-3])
        a2_scaled = a2 * np.array([1, h**-1, h**-2, h**-3])
        b_scaled = b * np.array([1, h**-1, h**-2, h**-3])

        a = a1_scaled  # 簡略化のため、a1のみ使用
        c = np.zeros(4)  # 右境界なのでc = 0

        F = 0.0

        return a, b_scaled, c, F

    def is_valid_at(self, i: int, n: int) -> bool:
        """右境界でのみ有効"""
        return i == n - 1
