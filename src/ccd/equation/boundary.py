# equation/boundary.py
import cupy as np
from typing import Dict
from .base import Equation


class DirichletBoundaryEquation(Equation):
    """ディリクレ境界条件: psi(x) = alpha"""

    def __init__(self, alpha: float, is_left: bool = True):
        """
        ディリクレ境界条件の初期化

        Args:
            alpha: 境界値
            is_left: 左境界ならTrue、右境界ならFalse
        """
        self.alpha = alpha
        self.is_left = is_left

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        """ディリクレ境界条件の係数を返す"""
        coeffs = {
            0: np.array([1, 0, 0, 0]),  # psiに対応する係数を1に設定
        }

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        """指定された境界値"""
        return self.alpha

    def is_valid_at(self, i: int, n: int) -> bool:
        """境界点でのみ有効"""
        if self.is_left:
            return i == 0
        else:
            return i == n - 1


class NeumannBoundaryEquation(Equation):
    """ノイマン境界条件: psi'(x) = beta"""

    def __init__(self, beta: float, is_left: bool = True):
        """
        ノイマン境界条件の初期化

        Args:
            beta: 境界での微分値
            is_left: 左境界ならTrue、右境界ならFalse
        """
        self.beta = beta
        self.is_left = is_left

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        """ノイマン境界条件の係数を返す"""
        coeffs = {
            0: np.array([0, 1, 0, 0]),  # psi'に対応する係数を1に設定
        }

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        """指定された微分値"""
        return self.beta

    def is_valid_at(self, i: int, n: int) -> bool:
        """境界点でのみ有効"""
        if self.is_left:
            return i == 0
        else:
            return i == n - 1
