from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import SpatialDiscretizationBase
from ...common.types import BoundaryCondition, BCType
from ...common.grid import GridManager

class CentralDifferenceDiscretization(SpatialDiscretizationBase):
    """2次精度中心差分による空間微分スキーム"""
    
    def __init__(
        self, 
        grid_manager: GridManager,
        boundary_conditions: Optional[dict[str, BoundaryCondition]] = None
    ):
        """
        中心差分スキームの初期化
        
        Args:
            grid_manager: グリッド管理オブジェクト
            boundary_conditions: 境界条件の辞書
        """
        super().__init__(grid_manager, boundary_conditions)
    
    def _first_derivative_x(self, field: ArrayLike, dx: float) -> ArrayLike:
        """
        x方向の一階微分（中心差分）
        
        Args:
            field: 入力フィールド
            dx: x方向のグリッド間隔
        
        Returns:
            x方向の一階微分
        """
        first_deriv = jnp.zeros_like(field)
        
        # 内部点の中心差分
        first_deriv = first_deriv.at[1:-1, :].set(
            (field[2:, :] - field[:-2, :]) / (2 * dx)
        )
        
        # 境界点の前方/後方差分
        first_deriv = first_deriv.at[0, :].set(
            (field[1, :] - field[0, :]) / dx
        )
        first_deriv = first_deriv.at[-1, :].set(
            (field[-1, :] - field[-2, :]) / dx
        )
        
        return first_deriv
    
    def _first_derivative_y(self, field: ArrayLike, dy: float) -> ArrayLike:
        """
        y方向の一階微分（中心差分）
        
        Args:
            field: 入力フィールド
            dy: y方向のグリッド間隔
        
        Returns:
            y方向の一階微分
        """
        first_deriv = jnp.zeros_like(field)
        
        # 内部点の中心差分
        first_deriv = first_deriv.at[:, 1:-1].set(
            (field[:, 2:] - field[:, :-2]) / (2 * dy)
        )
        
        # 境界点の前方/後方差分
        first_deriv = first_deriv.at[:, 0].set(
            (field[:, 1] - field[:, 0]) / dy
        )
        first_deriv = first_deriv.at[:, -1].set(
            (field[:, -1] - field[:, -2]) / dy
        )
        
        return first_deriv
    
    def _second_derivative_x(self, field: ArrayLike, dx: float) -> ArrayLike:
        """
        x方向の二階微分（中心差分）
        
        Args:
            field: 入力フィールド
            dx: x方向のグリッド間隔
        
        Returns:
            x方向の二階微分
        """
        second_deriv = jnp.zeros_like(field)
        
        # 内部点の中心差分
        second_deriv = second_deriv.at[1:-1, :].set(
            (field[2:, :] - 2 * field[1:-1, :] + field[:-2, :]) / (dx**2)
        )
        
        # 境界点の1次近似
        second_deriv = second_deriv.at[0, :].set(
            (field[1, :] - 2 * field[0, :]) / (dx**2)
        )
        second_deriv = second_deriv.at[-1, :].set(
            (field[-1, :] - 2 * field[-2, :]) / (dx**2)
        )
        
        return second_deriv
    
    def _second_derivative_y(self, field: ArrayLike, dy: float) -> ArrayLike:
        """
        y方向の二階微分（中心差分）
        
        Args:
            field: 入力フィールド
            dy: y方向のグリッド間隔
        
        Returns:
            y方向の二階微分
        """
        second_deriv = jnp.zeros_like(field)
        
        # 内部点の中心差分
        second_deriv = second_deriv.at[:, 1:-1].set(
            (field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2]) / (dy**2)
        )
        
        # 境界点の1次近似
        second_deriv = second_deriv.at[:, 0].set(
            (field[:, 1] - 2 * field[:, 0]) / (dy**2)
        )
        second_deriv = second_deriv.at[:, -1].set(
            (field[:, -1] - 2 * field[:, -2]) / (dy**2)
        )
        
        return second_deriv
    
    def discretize(
        self, 
        field: ArrayLike, 
        direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        空間微分の計算
        
        Args:
            field: 微分する入力フィールド
            direction: 微分の方向
        
        Returns:
            (一階微分, 二階微分)のタプル
        """
        # 境界条件の適用
        field = self.apply_boundary_conditions(field, direction)
        
        # グリッド間隔の取得
        dx = self.grid_manager.get_grid_spacing(direction)[0]
        
        # 方向に応じた微分の計算
        if direction == 'x':
            first_deriv = self._first_derivative_x(field, dx)
            second_deriv = self._second_derivative_x(field, dx)
        elif direction == 'y':
            first_deriv = self._first_derivative_y(field, dx)
            second_deriv = self._second_derivative_y(field, dx)
        else:
            raise ValueError(f"サポートされていない方向: {direction}")
        
        return first_deriv, second_deriv
    
    def apply_boundary_conditions(
        self, 
        field: ArrayLike, 
        direction: str
    ) -> ArrayLike:
        """
        境界条件の適用
        
        Args:
            field: 入力フィールド
            direction: 境界条件を適用する方向
        
        Returns:
            境界条件を適用したフィールド
        """
        # 境界条件が定義されていない場合はそのまま返す
        if not self.boundary_conditions or direction not in self.boundary_conditions:
            return field
        
        bc = self.boundary_conditions[direction]
        
        # 座標の取得
        x, y, _ = self.grid_manager.get_coordinates()
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        # 境界条件の種類に応じた処理
        if bc.type == BCType.DIRICHLET:
            # Dirichlet境界条件
            if callable(bc.value):
                # 関数型境界条件
                if direction == 'x':
                    field = field.at[0, :].set(bc.value(X[0, :], Y[0, :]))
                    field = field.at[-1, :].set(bc.value(X[-1, :], Y[-1, :]))
                elif direction == 'y':
                    field = field.at[:, 0].set(bc.value(X[:, 0], Y[:, 0]))
                    field = field.at[:, -1].set(bc.value(X[:, -1], Y[:, -1]))
            else:
                # 定数境界条件
                if direction == 'x':
                    field = field.at[0, :].set(bc.value)
                    field = field.at[-1, :].set(bc.value)
                elif direction == 'y':
                    field = field.at[:, 0].set(bc.value)
                    field = field.at[:, -1].set(bc.value)
        
        elif bc.type == BCType.NEUMANN:
            # Neumann境界条件（勾配を固定）
            if direction == 'x':
                field = field.at[0, :].set(field[1, :])
                field = field.at[-1, :].set(field[-2, :])
            elif direction == 'y':
                field = field.at[:, 0].set(field[:, 1])
                field = field.at[:, -1].set(field[:, -2])
        
        elif bc.type == BCType.PERIODIC:
            # 周期境界条件
            if direction == 'x':
                field = field.at[0, :].set(field[-2, :])
                field = field.at[-1, :].set(field[1, :])
            elif direction == 'y':
                field = field.at[:, 0].set(field[:, -2])
                field = field.at[:, -1].set(field[:, 1])
        
        return field