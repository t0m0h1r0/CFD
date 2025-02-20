from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple
from enum import Enum

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import CompactDifferenceBase
from ...common.grid import GridManager
from ...common.types import BoundaryCondition, BCType

@dataclass
class DifferenceCoefficients:
    """CCD法の差分係数を管理するデータクラス"""
    # 一階微分用係数
    alpha_1st: float
    beta_1st: float
    gamma_1st: float
    
    # 二階微分用係数
    alpha_2nd: float
    beta_2nd: float
    gamma_2nd: float
    
    @classmethod
    def create_for_order(cls, order: int) -> 'DifferenceCoefficients':
        """指定された精度次数の係数を生成"""
        if order == 8:
            return cls(
                # 一階微分用係数
                alpha_1st=15/16,
                beta_1st=-7/16,
                gamma_1st=1/16,
                
                # 二階微分用係数
                alpha_2nd=12/13,
                beta_2nd=-3/13,
                gamma_2nd=1/13
            )
        raise NotImplementedError(f"Order {order} is not supported")

class BoundaryHandler:
    """境界条件の処理を担当するクラス"""
    
    def __init__(self, dx: float, coefficients: DifferenceCoefficients):
        self.dx = dx
        self.coef = coefficients
        
    def initialize_ghost_points(self, field: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """ゴーストポイントの初期化"""
        gp_left = (7*field[0] - 21*field[1] + 35*field[2] - 
                  35*field[3] + 21*field[4] - 7*field[5] + field[6])
        gp_right = (7*field[-1] - 21*field[-2] + 35*field[-3] - 
                   35*field[-4] + 21*field[-5] - 7*field[-6] + field[-7])
        return gp_left, gp_right
        
    def apply_dirichlet_condition(
        self, 
        ghost_point: ArrayLike,
        boundary_value: float,
        first_deriv: ArrayLike,
        is_left: bool
    ) -> ArrayLike:
        """Dirichlet境界条件の適用"""
        factor = -10 - 150 * self.coef.alpha_1st
        delta = (60 * self.dx) / factor * (boundary_value - first_deriv)
        return ghost_point + delta
        
    def apply_neumann_condition(
        self,
        ghost_point: ArrayLike,
        boundary_value: float,
        second_deriv: ArrayLike,
        is_left: bool
    ) -> ArrayLike:
        """Neumann境界条件の適用"""
        factor = 137 + 180 * self.coef.gamma_2nd
        delta = (180 * self.dx**2) / factor * (boundary_value - second_deriv)
        return ghost_point + delta

class LaplacianMatrixBuilder:
    """ラプラシアン行列の構築を担当するクラス"""
    
    def __init__(self, dx: float, coefficients: DifferenceCoefficients):
        self.dx = dx
        self.coef = coefficients
        
    def build_first_derivative_coefficients(
        self,
        field_gp: ArrayLike,
        field: ArrayLike,
        is_left: bool
    ) -> ArrayLike:
        """一階微分の係数行列を構築"""
        sign = -1 if is_left else 1
        return sign * (1/60) * (
            (10 + 150*self.coef.alpha_1st)*field_gp +
            (77 - 840*self.coef.alpha_1st)*field[0 if is_left else -1] +
            sign*(150 - 1950*self.coef.alpha_1st)*field[1 if is_left else -2] +
            sign*(100 - 2400*self.coef.alpha_1st)*field[2 if is_left else -3] +
            sign*(50 - 1650*self.coef.alpha_1st)*field[3 if is_left else -4] +
            sign*(15 - 600*self.coef.alpha_1st)*field[4 if is_left else -5] +
            sign*(2 - 90*self.coef.alpha_1st)*field[5 if is_left else -6]
        ) / self.dx

    def build_second_derivative_coefficients(
        self,
        field_gp: ArrayLike,
        field: ArrayLike,
        is_left: bool
    ) -> ArrayLike:
        """二階微分の係数行列を構築"""
        return (1/180) * (
            (137 + 180*self.coef.gamma_2nd)*field_gp +
            -(147 + 1080*self.coef.gamma_2nd)*field[0 if is_left else -1] +
            -(255 - 2700*self.coef.gamma_2nd)*field[1 if is_left else -2] +
            (470 - 3600*self.coef.gamma_2nd)*field[2 if is_left else -3] +
            -(285 - 2700*self.coef.gamma_2nd)*field[3 if is_left else -4] +
            (93 - 1080*self.coef.gamma_2nd)*field[4 if is_left else -5] +
            -(13 - 180*self.coef.gamma_2nd)*field[5 if is_left else -6]
        ) / self.dx**2

class CCDLaplacianSolver(CompactDifferenceBase):
    """Combined Compact Difference (CCD) Laplacian Solver"""
    
    def __init__(
        self, 
        grid_manager: GridManager, 
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        order: int = 8
    ):
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = DifferenceCoefficients.create_for_order(order)
        self.order = order
    
    def compute_laplacian(self, field: ArrayLike) -> ArrayLike:
        """ラプラシアンの計算"""
        _, laplacian_x = self.discretize(field, 'x')
        _, laplacian_y = self.discretize(field, 'y')
        _, laplacian_z = self.discretize(field, 'z')
        return laplacian_x + laplacian_y + laplacian_z
    
    def discretize(
        self, 
        field: ArrayLike, 
        direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """空間離散化の実行"""
        dx = self.grid_manager.get_grid_spacing(direction)
        
        # 係数行列の生成
        matrix_builder = LaplacianMatrixBuilder(dx, self.coefficients)
        boundary_handler = BoundaryHandler(dx, self.coefficients)
        
        # 内部点での離散化
        first_deriv, second_deriv = self._compute_interior_derivatives(field, dx)
        
        # 境界条件の適用
        first_deriv, second_deriv = self.apply_boundary_conditions(
            field, (first_deriv, second_deriv), direction
        )
        
        return first_deriv, second_deriv
    
    def _compute_interior_derivatives(
        self, 
        field: ArrayLike, 
        dx: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        """内部点での微分計算"""
        first_deriv = jnp.zeros_like(field)
        second_deriv = jnp.zeros_like(field)
        
        first_deriv = first_deriv.at[1:-1].set(
            (field[2:] - field[:-2]) / (2 * dx)
        )
        second_deriv = second_deriv.at[1:-1].set(
            (field[2:] - 2 * field[1:-1] + field[:-2]) / (dx**2)
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
        
        boundary_handler = BoundaryHandler(dx, self.coefficients)
        matrix_builder = LaplacianMatrixBuilder(dx, self.coefficients)
        
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
            gp_left = boundary_handler.apply_dirichlet_condition(
                gp_left, bc_left.value, first_deriv[0], True
            )
        elif bc_left.type == BCType.NEUMANN:
            gp_left = boundary_handler.apply_neumann_condition(
                gp_left, bc_left.value, second_deriv[0], True
            )
            
        if bc_right.type == BCType.DIRICHLET:
            gp_right = boundary_handler.apply_dirichlet_condition(
                gp_right, bc_right.value, first_deriv[-1], False
            )
        elif bc_right.type == BCType.NEUMANN:
            gp_right = boundary_handler.apply_neumann_condition(
                gp_right, bc_right.value, second_deriv[-1], False
            )
        
        # 境界での微分の計算
        first_deriv = first_deriv.at[0].set(
            matrix_builder.build_first_derivative_coefficients(gp_left, field, True)
        )
        first_deriv = first_deriv.at[-1].set(
            matrix_builder.build_first_derivative_coefficients(gp_right, field, False)
        )
        
        second_deriv = second_deriv.at[0].set(
            matrix_builder.build_second_derivative_coefficients(gp_left, field, True)
        )
        second_deriv = second_deriv.at[-1].set(
            matrix_builder.build_second_derivative_coefficients(gp_right, field, False)
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