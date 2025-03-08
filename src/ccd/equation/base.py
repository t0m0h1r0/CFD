# equation/base.py
from abc import ABC, abstractmethod
import cupy as cp
from typing import Dict, TypeVar
from grid import Grid

# Type variable for self-referencing
T = TypeVar("T", bound="Equation")


class Equation(ABC):
    """差分方程式の基底クラス"""

    @abstractmethod
    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        グリッド点iにおけるステンシル係数を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            {offset: coeffs, ...} の辞書
            offsetはグリッド点iからの相対位置（i+offset）
            coeffsは [psi, psi', psi'', psi'''] に対応する4成分ベクトル
        """
        pass

    @abstractmethod
    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        方程式の右辺を返す

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            右辺の値
        """
        pass

    @abstractmethod
    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        方程式がグリッド点iに適用可能かを判定

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            方程式が適用可能な場合True
        """
        pass

    def __add__(self: T, other: T) -> "CombinedEquation":
        """
        二つの方程式を加算する演算子のオーバーロード

        Args:
            other: 加算する方程式

        Returns:
            二つの方程式を組み合わせた新しい方程式
        """
        return CombinedEquation(self, other, operation="+")

    def __radd__(self: T, other: T) -> "CombinedEquation":
        """
        右側からの加算をサポート

        Args:
            other: 加算する方程式

        Returns:
            二つの方程式を組み合わせた新しい方程式
        """
        return CombinedEquation(other, self, operation="+")

    def __sub__(self: T, other: T) -> "CombinedEquation":
        """
        二つの方程式を減算する演算子のオーバーロード

        Args:
            other: 減算する方程式

        Returns:
            二つの方程式の差分を表す新しい方程式
        """
        return CombinedEquation(self, other, operation="-")

    def __rsub__(self: T, other: T) -> "CombinedEquation":
        """
        右側からの減算をサポート

        Args:
            other: 減算される方程式

        Returns:
            二つの方程式の差分を表す新しい方程式
        """
        return CombinedEquation(other, self, operation="-")

    def __mul__(self: T, scalar: float) -> "ScaledEquation":
        """
        方程式をスカラー倍する演算子のオーバーロード

        Args:
            scalar: 係数

        Returns:
            スカラー倍された方程式
        """
        return ScaledEquation(self, scalar)

    def __rmul__(self: T, scalar: float) -> "ScaledEquation":
        """
        右側からのスカラー倍をサポート

        Args:
            scalar: 係数

        Returns:
            スカラー倍された方程式
        """
        return ScaledEquation(self, scalar)


class CombinedEquation(Equation):
    """二つの方程式を組み合わせた方程式"""

    def __init__(self, eq1: Equation, eq2: Equation, operation: str = "+"):
        """
        初期化

        Args:
            eq1: 一つ目の方程式
            eq2: 二つ目の方程式
            operation: 演算子（'+'または'-'）
        """
        self.eq1 = eq1
        self.eq2 = eq2
        self.operation = operation

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        組み合わせた方程式のステンシル係数を計算

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            組み合わせた係数の辞書
        """
        # 両方の方程式からステンシル係数を取得
        coeffs1 = self.eq1.get_stencil_coefficients(grid, i)
        coeffs2 = self.eq2.get_stencil_coefficients(grid, i)

        # 組み合わせた係数を格納する辞書
        combined_coeffs = {}

        # 両方の辞書のキー（オフセット）の集合を取得
        all_offsets = set(list(coeffs1.keys()) + list(coeffs2.keys()))

        # 各オフセットについて係数を計算
        for offset in all_offsets:
            # デフォルト値を0のベクトルとして設定
            coeff1 = coeffs1.get(offset, cp.zeros(4))
            coeff2 = coeffs2.get(offset, cp.zeros(4))

            # 演算に従って係数を組み合わせる
            if self.operation == "+":
                combined_coeffs[offset] = coeff1 + coeff2
            else:  # self.operation == '-'
                combined_coeffs[offset] = coeff1 - coeff2

        return combined_coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        組み合わせた方程式の右辺を計算

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            組み合わせた右辺の値
        """
        rhs1 = self.eq1.get_rhs(grid, i)
        rhs2 = self.eq2.get_rhs(grid, i)

        if self.operation == "+":
            return rhs1 + rhs2
        else:  # self.operation == '-'
            return rhs1 - rhs2

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        組み合わせた方程式が適用可能かを判定

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            両方の方程式が適用可能な場合にTrue
        """
        # 両方の方程式が適用可能である必要がある
        return self.eq1.is_valid_at(grid, i) and self.eq2.is_valid_at(grid, i)


class ScaledEquation(Equation):
    """スカラー倍された方程式"""

    def __init__(self, equation: Equation, scalar: float):
        """
        初期化

        Args:
            equation: 元の方程式
            scalar: 係数
        """
        self.equation = equation
        self.scalar = scalar

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        スケールされた方程式のステンシル係数を計算

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            スケールされた係数の辞書
        """
        coeffs = self.equation.get_stencil_coefficients(grid, i)

        # 各係数をスカラー倍
        scaled_coeffs = {
            offset: self.scalar * coeff for offset, coeff in coeffs.items()
        }

        return scaled_coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        スケールされた方程式の右辺を計算

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            スケールされた右辺の値
        """
        return self.scalar * self.equation.get_rhs(grid, i)

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        スケールされた方程式が適用可能かを判定

        Args:
            grid: 計算格子
            i: グリッド点のインデックス

        Returns:
            元の方程式が適用可能な場合にTrue
        """
        return self.equation.is_valid_at(grid, i)
