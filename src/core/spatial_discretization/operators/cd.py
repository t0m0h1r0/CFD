from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import SpatialDiscretizationBase
from ...common.types import BoundaryCondition, BCType
from ...common.grid import GridManager

class CentralDifferenceDiscretization(SpatialDiscretizationBase):
    """中心差分による空間微分スキーム"""
    
    def __init__(self, 
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None):
        """
        中心差分スキームの初期化
        
        Args:
            grid_manager: グリッド管理オブジェクト
            boundary_conditions: 境界条件の辞書
        """
        super().__init__(grid_manager, boundary_conditions)
    
    def _central_first_derivative(self, 
                                 field: ArrayLike, 
                                 direction: str) -> ArrayLike:
        """
        中心差分による一階微分の計算
        
        Args:
            field: 微分する入力フィールド
            direction: 微分の方向 ('x' または 'y')
        
        Returns:
            一階微分の結果
        """
        # グリッドスペーシングの取得
        dx = self.grid_manager.get_grid_spacing(direction)
        
        # 方向に応じた微分の計算
        if direction == 'x':
            # x方向の微分
            first_deriv = jnp.zeros_like(field)
            first_deriv = first_deriv.at[1:-1, :].set(
                (field[2:, :] - field[:-2, :]) / (2 * dx)
            )
            
            # 前方差分による境界点の処理
            first_deriv = first_deriv.at[0, :].set(
                (field[1, :] - field[0, :]) / dx
            )
            first_deriv = first_deriv.at[-1, :].set(
                (field[-1, :] - field[-2, :]) / dx
            )
        
        elif direction == 'y':
            # y方向の微分
            first_deriv = jnp.zeros_like(field)
            first_deriv = first_deriv.at[:, 1:-1].set(
                (field[:, 2:] - field[:, :-2]) / (2 * dx)
            )
            
            # 前方差分による境界点の処理
            first_deriv = first_deriv.at[:, 0].set(
                (field[:, 1] - field[:, 0]) / dx
            )
            first_deriv = first_deriv.at[:, -1].set(
                (field[:, -1] - field[:, -2]) / dx
            )
        
        else:
            raise ValueError(f"無効な微分方向: {direction}")
        
        return first_deriv
    
    def _central_second_derivative(self, 
                                  field: ArrayLike, 
                                  direction: str) -> ArrayLike:
        """
        中心差分による二階微分の計算
        
        Args:
            field: 微分する入力フィールド
            direction: 微分の方向 ('x' または 'y')
        
        Returns:
            二階微分の結果
        """
        # グリッドスペーシングの取得
        dx = self.grid_manager.get_grid_spacing(direction)
        
        # 方向に応じた二階微分の計算
        if direction == 'x':
            # x方向の二階微分
            second_deriv = jnp.zeros_like(field)
            second_deriv = second_deriv.at[1:-1, :].set(
                (field[2:, :] - 2 * field[1:-1, :] + field[:-2, :]) / (dx**2)
            )
            
            # 境界点の処理（1次近似）
            second_deriv = second_deriv.at[0, :].set(
                (field[1, :] - 2 * field[0, :]) / (dx**2)
            )
            second_deriv = second_deriv.at[-1, :].set(
                (field[-1, :] - 2 * field[-2, :]) / (dx**2)
            )
        
        elif direction == 'y':
            # y方向の二階微分
            second_deriv = jnp.zeros_like(field)
            second_deriv = second_deriv.at[:, 1:-1].set(
                (field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2]) / (dx**2)
            )
            
            # 境界点の処理（1次近似）
            second_deriv = second_deriv.at[:, 0].set(
                (field[:, 1] - 2 * field[:, 0]) / (dx**2)
            )
            second_deriv = second_deriv.at[:, -1].set(
                (field[:, -1] - 2 * field[:, -2]) / (dx**2)
            )
        
        else:
            raise ValueError(f"無効な微分方向: {direction}")
        
        return second_deriv
    
    def discretize(self, 
                   field: ArrayLike, 
                   direction: str) -> Tuple[ArrayLike, ArrayLike]:
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
        
        # 一階微分と二階微分の計算
        first_deriv = self._central_first_derivative(field, direction)
        second_deriv = self._central_second_derivative(field, direction)
        
        return first_deriv, second_deriv
    
    def apply_boundary_conditions(self, 
                                  field: ArrayLike, 
                                  direction: str) -> ArrayLike:
        """
        境界条件の適用
        
        Args:
            field: 入力フィールド
            direction: 境界条件を適用する方向
        
        Returns:
            境界条件を適用したフィールド
        """
        # 境界条件が定義されていない場合は、そのまま返す
        if not self.boundary_conditions or direction not in self.boundary_conditions:
            return field
        
        bc = self.boundary_conditions[direction]
        
        # 境界条件の種類に応じた処理
        if bc.type == BCType.DIRICHLET:
            # Dirichlet境界条件（定数値または関数）
            if callable(bc.value):
                # 関数の場合は、座標に応じた値を計算
                x, y, _ = self.grid_manager.get_coordinates()
                X, Y = jnp.meshgrid(x, y, indexing='ij')
                
                if direction == 'x':
                    field = field.at[0, :].set(bc.value(X[0, :], Y[0, :]))
                    field = field.at[-1, :].set(bc.value(X[-1, :], Y[-1, :]))
                elif direction == 'y':
                    field = field.at[:, 0].set(bc.value(X[:, 0], Y[:, 0]))
                    field = field.at[:, -1].set(bc.value(X[:, -1], Y[:, -1]))
            else:
                # 定数値の場合
                if direction == 'x':
                    field = field.at[0, :].set(bc.value)
                    field = field.at[-1, :].set(bc.value)
                elif direction == 'y':
                    field = field.at[:, 0].set(bc.value)
                    field = field.at[:, -1].set(bc.value)
        
        elif bc.type == BCType.NEUMANN:
            # Neumannの境界条件（勾配を一定に保つ）
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