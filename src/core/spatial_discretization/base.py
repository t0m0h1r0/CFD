from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Protocol, Union, runtime_checkable

import jax.numpy as jnp
from jax.typing import ArrayLike

from ..common.grid import GridManager
from ..common.types import BoundaryCondition, BCType


@runtime_checkable
class DerivativeStrategy(Protocol):
    """空間微分のための戦略インターフェース"""

    def compute_first_derivative(self, field: ArrayLike, dx: float) -> ArrayLike:
        """一階微分の計算"""
        ...

    def compute_second_derivative(self, field: ArrayLike, dx: float) -> ArrayLike:
        """二階微分の計算"""
        ...


class BoundaryConditionApplicator:
    """境界条件適用のための汎用クラス"""

    def __init__(self, grid_manager: GridManager):
        """
        境界条件適用器の初期化

        Args:
            grid_manager: グリッド管理オブジェクト
        """
        self.grid_manager = grid_manager

    def apply_boundary_condition(
        self,
        field: ArrayLike,
        boundary_conditions: Dict[str, BoundaryCondition],
        direction: str,
    ) -> ArrayLike:
        """
        指定された方向の境界条件を適用

        Args:
            field: 入力フィールド
            boundary_conditions: 境界条件の辞書
            direction: 適用する方向 ('x', 'y', 'z')

        Returns:
            境界条件適用後のフィールド
        """
        # 境界マッピング
        boundary_map = {
            "x": ("left", "right"),
            "y": ("bottom", "top"),
            "z": ("front", "back"),
        }

        try:
            left_key, right_key = boundary_map[direction]
        except KeyError:
            raise ValueError(f"Invalid direction: {direction}")

        # デフォルト境界条件
        default_bc = BoundaryCondition(
            type=BCType.DIRICHLET, value=0.0, location="default"
        )

        # 左境界条件の取得
        left_bc = boundary_conditions.get(left_key, default_bc)
        right_bc = boundary_conditions.get(right_key, default_bc)

        # 境界条件の適用
        modified_field = field.copy()

        if left_bc.type == BCType.DIRICHLET:
            modified_field = modified_field.at[0].set(left_bc.value)
        elif left_bc.type == BCType.PERIODIC:
            modified_field = modified_field.at[0].set(modified_field[-1])

        if right_bc.type == BCType.DIRICHLET:
            modified_field = modified_field.at[-1].set(right_bc.value)
        elif right_bc.type == BCType.PERIODIC:
            modified_field = modified_field.at[-1].set(modified_field[0])

        return modified_field


class SpatialDiscretizationBase(ABC):
    """空間離散化の抽象基底クラス"""

    def __init__(
        self,
        grid_manager: GridManager,
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        derivative_strategy: Optional[DerivativeStrategy] = None,
    ):
        """
        空間離散化の初期化

        Args:
            grid_manager: グリッド管理オブジェクト
            boundary_conditions: 境界条件の辞書（オプション）
            derivative_strategy: 微分戦略（オプション）
        """
        self.grid_manager = grid_manager
        self.boundary_conditions = boundary_conditions or {}
        self.derivative_strategy = derivative_strategy
        self.boundary_condition_applicator = BoundaryConditionApplicator(grid_manager)

    @abstractmethod
    def discretize(
        self, field: ArrayLike, direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        空間微分の計算

        Args:
            field: 入力フィールド
            direction: 微分方向 ('x', 'y', 'z')

        Returns:
            一階微分と二階微分のタプル
        """
        raise NotImplementedError(
            "Discretization method must be implemented by subclasses"
        )

    def apply_boundary_conditions(
        self, field: ArrayLike, derivatives: Tuple[ArrayLike, ArrayLike], direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        デフォルトの境界条件適用メソッド

        Args:
            field: 入力フィールド
            derivatives: 微分値のタプル（一階微分、二階微分）
            direction: 適用する方向

        Returns:
            境界条件適用後の微分値
        """
        # フィールドに境界条件を適用
        modified_field = self.boundary_condition_applicator.apply_boundary_condition(
            field, self.boundary_conditions, direction
        )

        # 派生クラスで実装される具体的な境界条件処理
        first_deriv, second_deriv = derivatives

        return (
            first_deriv,  # 暫定的に元の微分値を返す
            second_deriv,
        )

    def set_derivative_strategy(self, strategy: DerivativeStrategy) -> None:
        """
        微分戦略の動的設定

        Args:
            strategy: 新しい微分戦略
        """
        self.derivative_strategy = strategy


class CompactDifferenceBase(SpatialDiscretizationBase):
    """コンパクト差分法の基底クラス"""

    def __init__(
        self,
        grid_manager: GridManager,
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        coefficients: Optional[Dict] = None,
    ):
        """
        コンパクト差分法の初期化

        Args:
            grid_manager: グリッド管理オブジェクト
            boundary_conditions: 境界条件の辞書
            coefficients: 差分係数の辞書
        """
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = coefficients or {}

    def build_coefficient_matrices(self, direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        係数行列の構築

        Args:
            direction: 微分方向

        Returns:
            左辺行列と右辺行列のタプル
        """
        dx = self.grid_manager.get_grid_spacing(direction)
        n_points = self.grid_manager.get_grid_points(direction)

        # 行列の初期化
        lhs = jnp.zeros((2 * n_points, 2 * n_points))
        rhs = jnp.zeros((2 * n_points, n_points))

        return lhs, rhs

    @abstractmethod
    def solve_system(
        self, lhs: ArrayLike, rhs: ArrayLike, field: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        連立方程式の解法

        Args:
            lhs: 左辺行列
            rhs: 右辺行列
            field: 入力フィールド

        Returns:
            一階微分と二階微分
        """
        raise NotImplementedError(
            "System solving method must be implemented by subclasses"
        )


# クラスごとのユニークな型を定義
DiscretizationTypes = Union[SpatialDiscretizationBase, CompactDifferenceBase]
