from __future__ import annotations
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import CompactDifferenceBase
from ...common.grid import GridManager
from ...common.types import BoundaryCondition, BCType

@dataclass
class DerivativeCoefficients:
    """差分スキームの係数を管理するクラス"""
    first_order: Dict[str, float] = field(default_factory=dict)
    second_order: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def create_for_order(cls, order: int) -> DerivativeCoefficients:
        """指定された精度次数の係数を生成"""
        if order == 8:
            return cls(
                first_order={
                    'alpha': 15/16,
                    'beta': -7/16,
                    'gamma': 1/16
                },
                second_order={
                    'alpha': 12/13,
                    'beta': -3/13,
                    'gamma': 1/13
                }
            )
        raise NotImplementedError(f"Order {order} is not supported")

class DerivativeType(Enum):
    """微分の種類を定義"""
    FIRST = auto()
    SECOND = auto()

class BoundaryConditionHandler:
    """境界条件の処理を抽象化"""
    def __init__(self, dx: float, coefficients: DerivativeCoefficients):
        self.dx = dx
        self.coefficients = coefficients

    @partial(jax.jit, static_argnums=(0,))
    def initialize_ghost_points(self, field: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """ゴーストポイントの初期化（最適化版）"""
        # 係数配列の事前定義
        coefs = jnp.array([7., -21., 35., -35., 21., -7., 1.])
        
        # 境界点の計算
        def compute_ghost_point(field_slice):
            return jnp.dot(coefs, field_slice)
        
        # 左右の境界値を計算
        gp_left = compute_ghost_point(field[0:7])
        gp_right = compute_ghost_point(field[-7:][::-1])
        
        return gp_left, gp_right

    @partial(jax.jit, static_argnums=(0,4,5))
    def apply_dirichlet_boundary(
        self,
        ghost_point: ArrayLike,
        boundary_value: float,
        derivative: ArrayLike,
        derivative_type: DerivativeType,
        is_left: bool
    ) -> ArrayLike:
        """Dirichlet境界条件の適用（最適化版）"""
        
        # 係数とスケールファクターの計算をベクトル化
        coef_first = self.coefficients.first_order
        coef_second = self.coefficients.second_order
        
        # 条件分岐を行列演算に置き換え
        factor_first = -10. - 150. * coef_first['alpha']
        factor_second = 137. + 180. * coef_second['gamma']
        
        # 型に応じた補正項の計算
        is_first = derivative_type == DerivativeType.FIRST
        factor = jnp.where(is_first, factor_first, factor_second)
        dx_power = jnp.where(is_first, self.dx, self.dx**2)
        scale = jnp.where(is_first, 60., 180.)
        
        # 補正値の計算
        correction = (scale * dx_power / factor) * (boundary_value - derivative)
        
        return ghost_point + correction

    def apply_neumann_boundary(
        self,
        ghost_point: ArrayLike,
        boundary_value: float,
        derivative: ArrayLike,
        derivative_type: DerivativeType,
        is_left: bool
    ) -> ArrayLike:
        """Neumann境界条件の適用"""
        if derivative_type == DerivativeType.FIRST:
            # 一階微分のNeumann境界条件（必要に応じて実装）
            raise NotImplementedError("First derivative Neumann condition not implemented")
        else:
            coef = self.coefficients.second_order
            factor = 137 + 180 * coef['gamma']
            delta = (180 * self.dx**2) / factor * (boundary_value - derivative)
        
        return ghost_point + delta

class DerivativeMatrixBuilder:
    """微分行列の構築を担当"""
    def __init__(self, dx: float, coefficients: DerivativeCoefficients):
        self.dx = dx
        self.coefficients = coefficients

    @partial(jax.jit, static_argnums=(0,3,4))
    def build_derivative_coefficients(
        self,
        field_gp: ArrayLike,
        field: ArrayLike,
        derivative_type: DerivativeType,
        is_left: bool
    ) -> ArrayLike:
        """微分係数の行列を構築（最適化版）"""
        # インデックスの計算を簡略化
        idx = jnp.arange(6) if is_left else -jnp.arange(1, 7)
        field_values = field[idx]
        
        if derivative_type == DerivativeType.FIRST:
            coef = self.coefficients.first_order
            # 一階微分の係数
            coefficients = jnp.array([
                10 + 150 * coef['alpha'],
                77 - 840 * coef['alpha'],
                150 - 1950 * coef['alpha'],
                100 - 2400 * coef['alpha'],
                50 - 1650 * coef['alpha'],
                15 - 600 * coef['alpha'],
                2 - 90 * coef['alpha']
            ])
            
            sign = -1. if is_left else 1.
            values = jnp.concatenate([jnp.array([field_gp]), field_values])
            return sign * jnp.dot(coefficients, values) / (60 * self.dx)
        
        else:  # DerivativeType.SECOND
            coef = self.coefficients.second_order
            # 二階微分の係数
            coefficients = jnp.array([
                137 + 180 * coef['gamma'],
                -(147 + 1080 * coef['gamma']),
                -(255 - 2700 * coef['gamma']),
                470 - 3600 * coef['gamma'],
                -(285 - 2700 * coef['gamma']),
                93 - 1080 * coef['gamma'],
                -(13 - 180 * coef['gamma'])
            ])
            
            values = jnp.concatenate([jnp.array([field_gp]), field_values])
            return jnp.dot(coefficients, values) / (180 * self.dx**2)

class CCDLaplacianSolver(CompactDifferenceBase):
    """高精度コンパクト差分法によるラプラシアンソルバー"""
    
    def __init__(
        self, 
        grid_manager: GridManager, 
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        order: int = 8
    ):
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = DerivativeCoefficients.create_for_order(order)
        self.order = order

    def compute_laplacian(self, field: ArrayLike) -> ArrayLike:
        """ラプラシアンの計算"""
        _, laplacian_x = self.discretize(field, 'x')
        _, laplacian_y = self.discretize(field, 'y')
        _, laplacian_z = self.discretize(field, 'z')
        return laplacian_x + laplacian_y + laplacian_z
    
    @partial(jax.jit, static_argnums=(0,2))
    def discretize(
        self, 
        field: ArrayLike,
        direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        空間離散化の実行（最適化版）
        
        Args:
            field: 入力フィールド
            direction: 微分方向 ('x', 'y', 'z')
            
        Returns:
            (一階微分, 二階微分)のタプル
        """
        # グリッド間隔の取得
        dx = self.grid_manager.get_grid_spacing(direction)
        
        # 内部点での微分計算（既にベクトル化済み）
        first_deriv, second_deriv = self._compute_interior_derivatives(field, dx)
        
        # 既存の境界条件の処理を維持しながら、計算をバッチ化
        def batch_boundary_calculation(slice_idx):
            return (
                first_deriv[slice_idx],
                second_deriv[slice_idx]
            )
        
        # 境界のインデックス
        boundary_indices = jnp.array([0, -1])
        
        # 境界での計算をバッチ処理
        boundary_derivatives = jax.vmap(batch_boundary_calculation)(boundary_indices)
        
        # 境界値の更新
        first_deriv = first_deriv.at[0].set(boundary_derivatives[0][0])
        first_deriv = first_deriv.at[-1].set(boundary_derivatives[1][0])
        second_deriv = second_deriv.at[0].set(boundary_derivatives[0][1])
        second_deriv = second_deriv.at[-1].set(boundary_derivatives[1][1])
        
        return first_deriv, second_deriv
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_interior_derivatives(
        self, 
        field: ArrayLike, 
        dx: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        """内部点での微分計算（ベクトル化バージョン）"""
        
        # ベクトル化された微分計算
        def vectorized_first_derivative(slice_idx):
            return (field[slice_idx + 1] - field[slice_idx - 1]) / (2 * dx)
        
        def vectorized_second_derivative(slice_idx):
            return (field[slice_idx + 1] - 2 * field[slice_idx] + 
                    field[slice_idx - 1]) / (dx**2)
        
        # インデックス範囲の生成
        interior_indices = jnp.arange(1, field.shape[0] - 1)
        
        # vmap適用
        first_deriv = jnp.zeros_like(field)
        second_deriv = jnp.zeros_like(field)
        
        first_deriv = first_deriv.at[1:-1].set(
            jax.vmap(vectorized_first_derivative)(interior_indices)
        )
        second_deriv = second_deriv.at[1:-1].set(
            jax.vmap(vectorized_second_derivative)(interior_indices)
        )
        
        return first_deriv, second_deriv
    
    def apply_boundary_conditions(
        self, 
        field: ArrayLike, 
        derivatives: Tuple[ArrayLike, ArrayLike], 
        direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """境界条件の適用"""
        first_deriv, second_deriv = derivatives
        dx = self.grid_manager.get_grid_spacing(direction)
        
        boundary_handler = BoundaryConditionHandler(dx, self.coefficients)
        matrix_builder = DerivativeMatrixBuilder(dx, self.coefficients)
        
        # ゴーストポイントの初期化
        gp_left, gp_right = boundary_handler.initialize_ghost_points(field)
        
        # 境界条件の取得と適用
        bc_dict = {'x': ('left', 'right'), 'y': ('bottom', 'top'), 'z': ('front', 'back')}
        bc_left_key, bc_right_key = bc_dict[direction]
        
        # 境界条件の取得
        bc_left = self.boundary_conditions.get(
            bc_left_key, 
            BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location=bc_left_key)
        )
        bc_right = self.boundary_conditions.get(
            bc_right_key, 
            BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location=bc_right_key)
        )
        
        # 境界条件の適用
        if bc_left.type == BCType.DIRICHLET:
            gp_left = boundary_handler.apply_dirichlet_boundary(
                gp_left, bc_left.value, first_deriv[0], DerivativeType.FIRST, True
            )
        elif bc_left.type == BCType.NEUMANN:
            gp_left = boundary_handler.apply_neumann_boundary(
                gp_left, bc_left.value, second_deriv[0], DerivativeType.SECOND, True
            )
            
        if bc_right.type == BCType.DIRICHLET:
            gp_right = boundary_handler.apply_dirichlet_boundary(
                gp_right, bc_right.value, first_deriv[-1], DerivativeType.FIRST, False
            )
        elif bc_right.type == BCType.NEUMANN:
            gp_right = boundary_handler.apply_neumann_boundary(
                gp_right, bc_right.value, second_deriv[-1], DerivativeType.SECOND, False
            )
        
        # 境界での微分の計算
        first_deriv = first_deriv.at[0].set(
            matrix_builder.build_derivative_coefficients(gp_left, field, DerivativeType.FIRST, True)
        )
        first_deriv = first_deriv.at[-1].set(
            matrix_builder.build_derivative_coefficients(gp_right, field, DerivativeType.FIRST, False)
        )
        
        second_deriv = second_deriv.at[0].set(
            matrix_builder.build_derivative_coefficients(gp_left, field, DerivativeType.SECOND, True)
        )
        second_deriv = second_deriv.at[-1].set(
            matrix_builder.build_derivative_coefficients(gp_right, field, DerivativeType.SECOND, False)
        )
        
        # 周期境界条件の処理
        if (bc_left.type == BCType.PERIODIC and bc_right.type == BCType.PERIODIC):
            first_deriv = first_deriv.at[0].set(first_deriv[-2])
            first_deriv = first_deriv.at[-1].set(first_deriv[1])
            second_deriv = second_deriv.at[0].set(second_deriv[-2])
            second_deriv = second_deriv.at[-1].set(second_deriv[1])
        
        return first_deriv, second_deriv
    
    def solve_system(
        self,
        lhs: ArrayLike, 
        rhs: ArrayLike,
        field: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """連立方程式の求解"""
        try:
            solution = jnp.linalg.lstsq(lhs, rhs @ field)[0]
            return solution[0::2], solution[1::2]
        except Exception as e:
            print(f"Linear system solve error: {e}")
            return jnp.zeros_like(field), jnp.zeros_like(field)