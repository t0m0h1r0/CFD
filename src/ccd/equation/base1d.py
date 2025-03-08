from abc import ABC, abstractmethod
import cupy as cp

class Equation(ABC):
    """差分方程式の基底クラス"""

    @abstractmethod
    def get_stencil_coefficients(self, grid, i):
        """グリッド点iにおけるステンシル係数を返す"""
        pass

    @abstractmethod
    def get_rhs(self, grid, i):
        """方程式の右辺を返す"""
        pass

    @abstractmethod
    def is_valid_at(self, grid, i):
        """方程式がグリッド点iに適用可能かを判定"""
        pass

    def __add__(self, other):
        return CombinedEquation(self, other, "+")

    def __sub__(self, other):
        return CombinedEquation(self, other, "-")

    def __mul__(self, scalar):
        return ScaledEquation(self, scalar)

    __rmul__ = __mul__


class CombinedEquation(Equation):
    """二つの方程式を組み合わせた方程式"""

    def __init__(self, eq1, eq2, operation="+"):
        self.eq1 = eq1
        self.eq2 = eq2
        self.operation = operation

    def get_stencil_coefficients(self, grid, i):
        coeffs1 = self.eq1.get_stencil_coefficients(grid, i)
        coeffs2 = self.eq2.get_stencil_coefficients(grid, i)
        combined_coeffs = {}
        all_offsets = set(list(coeffs1.keys()) + list(coeffs2.keys()))

        for offset in all_offsets:
            coeff1 = coeffs1.get(offset, cp.zeros(4))
            coeff2 = coeffs2.get(offset, cp.zeros(4))
            if self.operation == "+":
                combined_coeffs[offset] = coeff1 + coeff2
            else:
                combined_coeffs[offset] = coeff1 - coeff2

        return combined_coeffs

    def get_rhs(self, grid, i):
        rhs1 = self.eq1.get_rhs(grid, i)
        rhs2 = self.eq2.get_rhs(grid, i)
        return rhs1 + rhs2 if self.operation == "+" else rhs1 - rhs2

    def is_valid_at(self, grid, i):
        return self.eq1.is_valid_at(grid, i) and self.eq2.is_valid_at(grid, i)


class ScaledEquation(Equation):
    """スカラー倍された方程式"""

    def __init__(self, equation, scalar):
        self.equation = equation
        self.scalar = scalar

    def get_stencil_coefficients(self, grid, i):
        coeffs = self.equation.get_stencil_coefficients(grid, i)
        return {offset: self.scalar * coeff for offset, coeff in coeffs.items()}

    def get_rhs(self, grid, i):
        return self.scalar * self.equation.get_rhs(grid, i)

    def is_valid_at(self, grid, i):
        return self.equation.is_valid_at(grid, i)