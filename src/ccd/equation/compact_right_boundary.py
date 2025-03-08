# equation/compact_right_boundary.py
import cupy as np
from typing import Dict
from .base import Equation


class RightBoundaryFunctionEquation(Equation):
    """右境界点での関数値関係式"""

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        coeffs = {
            0: np.array([1, 0, 0, 0]),  # 境界点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * np.array([1, h**-1, h**-2, h**-3])

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        return 0.0

    def is_valid_at(self, i: int, n: int) -> bool:
        """右境界でのみ有効"""
        return i == n - 1


class RightBoundary1stDerivativeEquation(Equation):
    """右境界点での1階導関数関係式"""

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        # 既存コードから流用した右境界点の係数
        coeffs = {
            -2: np.array([59 / 2, 10, 1, 0]),  # i-2点
            -1: np.array([-24, 24, -4, 4 / 3]),  # i-1点
            0: np.array([-11 / 2, 1, 0, 0]),  # i点（境界点）
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * np.array([1, h**-1, h**-2, h**-3])

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        return 0.0

    def is_valid_at(self, i: int, n: int) -> bool:
        """右境界でのみ有効"""
        return i == n - 1


class RightBoundary2ndDerivativeEquation(Equation):
    """右境界点での2階導関数関係式"""

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        # 既存コードから流用した右境界点の係数
        coeffs = {
            -2: np.array([579 / 2, -99, 10, 0]),  # i-2点
            -1: np.array([-264, 216, -44, 12]),  # i-1点
            0: np.array([-51 / 2, 0, 1, 0]),  # i点（境界点）
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * np.array([1, h**-1, h**-2, h**-3])

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        return 0.0

    def is_valid_at(self, i: int, n: int) -> bool:
        """右境界でのみ有効"""
        return i == n - 1


class RightBoundary3rdDerivativeEquation(Equation):
    """右境界点での3階導関数関係式"""

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        # 既存コードから流用した右境界点の係数
        coeffs = {
            -2: np.array([-6963 / 4, -1203 / 2, -123 / 2, 0]),  # i-2点
            -1: np.array([1644, -1236, 282, -66]),  # i-1点
            0: np.array([-387 / 4, 0, 0, 1]),  # i点（境界点）
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * np.array([1, h**-1, h**-2, h**-3])

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        return 0.0

    def is_valid_at(self, i: int, n: int) -> bool:
        """右境界でのみ有効"""
        return i == n - 1
