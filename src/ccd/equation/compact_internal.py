# equation/compact_internal.py
import cupy as np
from typing import Dict
from .base import Equation

class Internal1stDerivativeEquation(Equation):
    """内部点での1階導関数関係式"""

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        # 既存コードから流用した内部点の係数
        coeffs = {
            -1: np.array([35 / 32, 19 / 32, 1 / 8, 1 / 96]),  # 左隣接点
            0: np.array([0, 1, 0, 0]),  # 中央点
            1: np.array([-35 / 32, 19 / 32, -1 / 8, 1 / 96]),  # 右隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * np.array([h**-1, h**0, h**1, h**2])

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        return 0.0

    def is_valid_at(self, i: int, n: int) -> bool:
        """内部点でのみ有効"""
        return 0 < i < n - 1


class Internal2ndDerivativeEquation(Equation):
    """内部点での2階導関数関係式"""

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        # 既存コードから流用した内部点の係数
        coeffs = {
            -1: np.array([-4, -29 / 16, -5 / 16, -1 / 48]),  # 左隣接点
            0: np.array([0, 0, 1, 0]),  # 中央点
            1: np.array([-4, 29 / 16, -5 / 16, 1 / 48]),  # 右隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * np.array([h**-2, h**-1, h**0, h**1])

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        return 0.0

    def is_valid_at(self, i: int, n: int) -> bool:
        """内部点でのみ有効"""
        return 0 < i < n - 1


class Internal3rdDerivativeEquation(Equation):
    """内部点での3階導関数関係式"""

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        # 既存コードから流用した内部点の係数
        coeffs = {
            -1: np.array([-105 / 16, -105 / 16, -15 / 8, -3 / 16]),  # 左隣接点
            0: np.array([0, 0, 0, 1]),  # 中央点
            1: np.array([105 / 16, -105 / 16, 15 / 8, -3 / 16]),  # 右隣接点
        }

        # スケール調整
        for offset, coef in coeffs.items():
            coeffs[offset] = coef * np.array([h**-3, h**-2, h**-1, h**0])

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        return 0.0

    def is_valid_at(self, i: int, n: int) -> bool:
        """内部点でのみ有効"""
        return 0 < i < n - 1
