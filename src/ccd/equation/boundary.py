# equation/boundary.py
import cupy as cp
from typing import Dict
from grid import Grid
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

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        ディリクレ境界条件のステンシル係数を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # psiに対応する係数を1に設定
        coeffs = {
            0: cp.array([1, 0, 0, 0]),
        }

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        指定された境界値を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            境界値
        """
        return self.alpha

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        境界点でのみ有効かを判定
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            境界点の場合True
        """
        n = grid.n_points
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

    def get_stencil_coefficients(self, grid: Grid, i: int) -> Dict[int, cp.ndarray]:
        """
        ノイマン境界条件のステンシル係数を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        # psi'に対応する係数を1に設定
        coeffs = {
            0: cp.array([0, 1, 0, 0]),
        }

        return coeffs

    def get_rhs(self, grid: Grid, i: int) -> float:
        """
        指定された微分値を返す
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            境界での微分値
        """
        return self.beta

    def is_valid_at(self, grid: Grid, i: int) -> bool:
        """
        境界点でのみ有効かを判定
        
        Args:
            grid: 計算格子
            i: グリッド点のインデックス
            
        Returns:
            境界点の場合True
        """
        n = grid.n_points
        if self.is_left:
            return i == 0
        else:
            return i == n - 1