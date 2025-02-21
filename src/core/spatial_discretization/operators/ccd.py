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
        """CCDの理論的な係数値を生成"""
        if order == 8:
            return cls(
                first_order={
                    'alpha': 17/12,
                    'beta': 1/24,
                    'gamma': -1/12
                },
                second_order={
                    'alpha': 43/8,
                    'beta': -3/20,
                    'gamma': 1/120
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
        """ゴーストポイントの初期化 - 6次精度の補間を使用"""
        # 境界補間のための係数 (6点ステンシル)
        coefs = jnp.array([37/60, -2/15, 1/60, -1/60, 2/15, -37/60])
        
        # 境界近傍のポイントを取得
        left_points = field[:6]  # Shape: (6, ny, nz)
        right_points = field[-6:][::-1]  # Shape: (6, ny, nz)
        
        # coefs を適切な形状に拡張 (6, 1, 1) -> (6, ny, nz) にブロードキャスト
        coefs = coefs.reshape(-1, 1, 1)
        
        # ゴーストポイントの計算
        gp_left = jnp.sum(coefs * left_points, axis=0)  # Shape: (ny, nz)
        gp_right = jnp.sum(coefs * right_points, axis=0)  # Shape: (ny, nz)
        
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
        """Dirichlet境界条件の改善版実装"""
        if derivative_type == DerivativeType.FIRST:
            # 6次精度の境界条件
            if is_left:
                coef = jnp.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]).reshape(-1, 1, 1)
                stencil = jnp.concatenate([ghost_point[None, ...], 
                                        jnp.broadcast_to(boundary_value, ghost_point.shape)[None, ...].repeat(6, axis=0)])
            else:
                coef = jnp.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])[::-1].reshape(-1, 1, 1)
                stencil = jnp.concatenate([jnp.broadcast_to(boundary_value, ghost_point.shape)[None, ...].repeat(6, axis=0),
                                        ghost_point[None, ...]])
                
            return jnp.sum(coef * stencil, axis=0)
        else:
            # 6次精度の2階微分境界条件
            if is_left:
                coef = jnp.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]).reshape(-1, 1, 1)
                stencil = jnp.concatenate([ghost_point[None, ...], 
                                        jnp.broadcast_to(boundary_value, ghost_point.shape)[None, ...].repeat(6, axis=0)])
            else:
                coef = jnp.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])[::-1].reshape(-1, 1, 1)
                stencil = jnp.concatenate([jnp.broadcast_to(boundary_value, ghost_point.shape)[None, ...].repeat(6, axis=0),
                                        ghost_point[None, ...]])
                
            return jnp.sum(coef * stencil, axis=0)

    def apply_neumann_boundary(
        self,
        ghost_point: ArrayLike,
        boundary_value: float,
        derivative: ArrayLike,
        derivative_type: DerivativeType,
        is_left: bool
    ) -> ArrayLike:
        """Neumann境界条件の実装"""
        if derivative_type == DerivativeType.FIRST:
            raise NotImplementedError("First derivative Neumann condition not implemented")
        
        # 6次精度のNeumann境界条件
        coef = jnp.array([1/60, -2/15, 37/60, -37/60, 2/15, -1/60])
        sign = 1.0 if is_left else -1.0
        
        return ghost_point + sign * self.dx * boundary_value * jnp.sum(coef)

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
        """微分係数の行列を構築 - 3次元対応版"""
        coef = (self.coefficients.first_order if derivative_type == DerivativeType.FIRST 
                else self.coefficients.second_order)
        
        # 係数を3次元用に整形
        if derivative_type == DerivativeType.FIRST:
            base_coef = jnp.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
        else:
            base_coef = jnp.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
            
        if not is_left:
            base_coef = base_coef[::-1]
        
        # 係数を3次元に拡張 (7,1,1)
        coefficients = base_coef.reshape(-1, 1, 1)
        
        # 境界点でのステンシル構築
        if is_left:
            stencil = jnp.concatenate([
                field_gp[None, ...],
                field[:6] if is_left else field[-6:]
            ])
        else:
            stencil = jnp.concatenate([
                field[:6] if is_left else field[-6:],
                field_gp[None, ...]
            ])
        
        # スケーリング係数
        scale = 1.0 / (self.dx if derivative_type == DerivativeType.FIRST else self.dx**2)
        
        # 3次元での計算
        return scale * jnp.sum(coefficients * stencil, axis=0)

class CombinedCompactDifference(CompactDifferenceBase):
    """高精度コンパクト差分法の実装"""
    
    def __init__(
        self, 
        grid_manager: GridManager,
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        order: int = 8
    ):
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = DerivativeCoefficients.create_for_order(order)
        self.order = order

    @partial(jax.jit, static_argnums=(0,2))
    def discretize(
        self,
        field: ArrayLike,
        direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        指定方向の空間微分を計算
        
        Args:
            field: 入力フィールド
            direction: 微分方向 ('x', 'y', 'z')
            
        Returns:
            (一階微分, 二階微分)のタプル
        """
        dx = self.grid_manager.get_grid_spacing(direction)
        
        # 内部点での微分計算
        first_deriv, second_deriv = self._compute_interior_derivatives(field, dx)
        
        # 境界条件の設定を取得
        bc_map = {'x': ('left', 'right'), 'y': ('bottom', 'top'), 'z': ('front', 'back')}
        bc_left_key, bc_right_key = bc_map[direction]
        
        # 境界条件の準備
        bc_left = self.boundary_conditions.get(
            bc_left_key,
            BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location=bc_left_key)
        )
        bc_right = self.boundary_conditions.get(
            bc_right_key,
            BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location=bc_right_key)
        )
        
        # 境界条件の適用
        boundary_handler = BoundaryConditionHandler(dx, self.coefficients)
        matrix_builder = DerivativeMatrixBuilder(dx, self.coefficients)
        
        # ゴーストポイントの初期化と更新
        gp_left, gp_right = boundary_handler.initialize_ghost_points(field)
        
        if bc_left.type == BCType.DIRICHLET:
            gp_left = boundary_handler.apply_dirichlet_boundary(
                gp_left, bc_left.value, first_deriv[0], DerivativeType.FIRST, True
            )
            
        if bc_right.type == BCType.DIRICHLET:
            gp_right = boundary_handler.apply_dirichlet_boundary(
                gp_right, bc_right.value, first_deriv[-1], DerivativeType.FIRST, False
            )
        
        # 境界点での微分値の更新
        first_deriv = first_deriv.at[0].set(
            matrix_builder.build_derivative_coefficients(
                gp_left, field, DerivativeType.FIRST, True
            )
        )
        first_deriv = first_deriv.at[-1].set(
            matrix_builder.build_derivative_coefficients(
                gp_right, field, DerivativeType.FIRST, False
            )
        )
        
        second_deriv = second_deriv.at[0].set(
            matrix_builder.build_derivative_coefficients(
                gp_left, field, DerivativeType.SECOND, True
            )
        )
        second_deriv = second_deriv.at[-1].set(
            matrix_builder.build_derivative_coefficients(
                gp_right, field, DerivativeType.SECOND, False
            )
        )
        
        return first_deriv, second_deriv

    def _compute_interior_derivatives(
        self,
        field: ArrayLike,
        dx: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        """内部点での微分を計算 - CCDの理論に基づく実装"""
        
        def interior_derivatives(i):
            # CCDスキームの係数
            coef = self.coefficients.first_order
            alpha, beta, gamma = coef['alpha'], coef['beta'], coef['gamma']
            
            # 一階微分の計算
            first_deriv = (alpha * (field[i+1] - field[i-1]) / (2*dx) +
                          beta * (field[i+2] - field[i-2]) / (4*dx) +
                          gamma * (field[i+3] - field[i-3]) / (6*dx))
            
            # 二階微分の計算
            coef = self.coefficients.second_order
            alpha, beta, gamma = coef['alpha'], coef['beta'], coef['gamma']
            second_deriv = (alpha * (field[i+1] - 2*field[i] + field[i-1]) / dx**2 +
                           beta * (field[i+2] - 2*field[i] + field[i-2]) / (4*dx**2) +
                           gamma * (field[i+3] - 2*field[i] + field[i-3]) / (9*dx**2))
            
            return first_deriv, second_deriv
        
        # 内部点のインデックス（境界から3点離れた点から）
        interior_indices = jnp.arange(3, field.shape[0] - 3)
        
        # 配列の初期化
        first_deriv = jnp.zeros_like(field)
        second_deriv = jnp.zeros_like(field)
        
        # 内部点の計算
        interior_results = jax.vmap(interior_derivatives)(interior_indices)
        first_interior, second_interior = interior_results
        
        # 結果の代入
        first_deriv = first_deriv.at[3:-3].set(first_interior)
        second_deriv = second_deriv.at[3:-3].set(second_interior)
        
        return first_deriv, second_deriv

    def solve_system(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
        field: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        連立方程式の求解
        
        Args:
            lhs: 左辺行列
            rhs: 右辺行列
            field: 入力フィールド
            
        Returns:
            (一階微分, 二階微分)のタプル
        """
        try:
            # JAXのLeast Squaresソルバーを使用
            solution = jnp.linalg.lstsq(lhs, rhs @ field)[0]
            
            # 解を一階微分と二階微分に分離
            first_deriv = solution[0::2]  # 偶数インデックス
            second_deriv = solution[1::2]  # 奇数インデックス
            
            return first_deriv, second_deriv
            
        except Exception as e:
            print(f"Linear system solve error: {e}")
            return jnp.zeros_like(field), jnp.zeros_like(field)