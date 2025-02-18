from typing import Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import CompactDifferenceBase
from ...common.types import Grid, BoundaryCondition
from ...common.grid import GridManager

class CombinedCompactDifference(CompactDifferenceBase):
    """Combined Compact Difference (CCD) スキームの実装"""
    
    def __init__(self,
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None,
                 order: int = 6):
        """
        CCDスキームの初期化
        
        Args:
            grid_manager: 格子管理オブジェクト
            boundary_conditions: 境界条件の辞書
            order: 精度次数 (デフォルト: 6)
        """
        # 精度に応じた係数の計算
        coefficients = self._calculate_coefficients(order)
        super().__init__(grid_manager, boundary_conditions, coefficients)
        self.order = order
        
    def _calculate_coefficients(self, order: int) -> dict:
        """
        指定された精度次数に応じたCCD係数を計算
        
        Args:
            order: 精度次数
            
        Returns:
            係数の辞書
        
        Raises:
            NotImplementedError: 未実装の精度次数が指定された場合
        """
        if order == 6:
            return {
                'a1': 15/16,
                'b1': -7/16,
                'c1': 1/16,
                'a2': 3/4,
                'b2': -9/8,
                'c2': 1/8
            }
        else:
            raise NotImplementedError(f"Order {order} not implemented")
            
    def build_coefficient_matrices(self, 
                                 direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        CCD係数行列を構築
        
        Args:
            direction: 係数行列を構築する方向
            
        Returns:
            (左辺行列, 右辺行列) のタプル
        """
        dx = self.get_grid_spacing(direction)
        n_points = self.get_grid_points(direction)
        
        # 係数の取得
        a1, b1, c1 = (self.coefficients[k] for k in ['a1', 'b1', 'c1'])
        a2, b2, c2 = (self.coefficients[k] for k in ['a2', 'b2', 'c2'])
        
        # 行列の初期化
        lhs = jnp.zeros((2*n_points, 2*n_points))
        rhs = jnp.zeros((2*n_points, n_points))
        
        # 内部点のステンシル構築
        for i in range(1, n_points-1):
            # 一階微分の方程式
            lhs = lhs.at[2*i, 2*i].set(1.0)
            lhs = lhs.at[2*i, 2*(i-1)].set(b1)
            lhs = lhs.at[2*i, 2*(i+1)].set(b1)
            lhs = lhs.at[2*i, 2*(i-1)+1].set(c1/dx)
            lhs = lhs.at[2*i, 2*(i+1)+1].set(-c1/dx)
            
            rhs = rhs.at[2*i, i+1].set(a1/(2*dx))
            rhs = rhs.at[2*i, i-1].set(-a1/(2*dx))
            
            # 二階微分の方程式
            lhs = lhs.at[2*i+1, 2*i+1].set(1.0)
            lhs = lhs.at[2*i+1, 2*(i-1)+1].set(c2)
            lhs = lhs.at[2*i+1, 2*(i+1)+1].set(c2)
            lhs = lhs.at[2*i+1, 2*(i-1)].set(b2/dx)
            lhs = lhs.at[2*i+1, 2*(i+1)].set(-b2/dx)
            
            rhs = rhs.at[2*i+1, i-1].set(a2/dx**2)
            rhs = rhs.at[2*i+1, i].set(-2*a2/dx**2)
            rhs = rhs.at[2*i+1, i+1].set(a2/dx**2)
            
        return lhs, rhs
    
    @partial(jax.jit, static_argnums=(0,))
    def solve_system(self,
                    lhs: ArrayLike,
                    rhs: ArrayLike,
                    field: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        CCD方程式系を解く
        
        Args:
            lhs: 左辺行列
            rhs: 右辺行列
            field: 入力場
            
        Returns:
            (一階微分, 二階微分) のタプル
        """
        # 右辺ベクトルの計算
        rhs_vector = jnp.matmul(rhs, field)
        
        # 連立方程式を解く
        solution = jax.scipy.linalg.solve(lhs, rhs_vector)
        
        # 一階微分と二階微分を取り出す
        n_points = len(field)
        first_deriv = solution[::2]
        second_deriv = solution[1::2]
        
        return first_deriv, second_deriv
    
    def discretize(self,
                  field: ArrayLike,
                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        CCDスキームを用いて空間微分を計算
        
        Args:
            field: 入力場
            direction: 微分方向
            
        Returns:
            (一階微分, 二階微分) のタプル
        """
        # 係数行列の構築
        lhs, rhs = self.build_coefficient_matrices(direction)
        
        # 方程式系を解く
        derivatives = self.solve_system(lhs, rhs, field)
        
        # 境界条件の適用
        derivatives = self.apply_boundary_conditions(field, derivatives, direction)
        
        return derivatives
    
    def apply_boundary_conditions(self,
                                field: ArrayLike,
                                derivatives: Tuple[ArrayLike, ArrayLike],
                                direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        CCDスキームの境界条件を適用
        
        Args:
            field: 入力場
            derivatives: (一階微分, 二階微分) のタプル
            direction: 境界条件を適用する方向
            
        Returns:
            境界条件適用後の微分値
        """
        first_deriv, second_deriv = derivatives
        
        if direction not in self.boundary_conditions:
            return derivatives
            
        bc = self.boundary_conditions[direction]
        dx = self.get_grid_spacing(direction)
        
        # CCD用の高精度境界ステンシルを使用
        def apply_boundary_deriv(deriv, order):
            # 6次精度の境界ステンシル係数
            if order == 1:  # 一階微分
                coef = jnp.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]) / dx
            elif order == 2:  # 二階微分
                coef = jnp.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]) / dx**2
            
            # 左境界
            stencil_left = jnp.zeros_like(deriv)
            for i in range(3):
                values = jnp.roll(field, shift=i-3, axis=0)[:7]
                stencil_left = stencil_left.at[i].set(jnp.sum(values * coef))
            
            # 右境界
            stencil_right = jnp.zeros_like(deriv)
            for i in range(3):
                values = jnp.roll(field, shift=i-3, axis=0)[-7:]
                stencil_right = stencil_right.at[-i-1].set(jnp.sum(values * coef[::-1]))
            
            # 境界値の適用
            mask_left = jnp.arange(len(deriv)) < 3
            mask_right = jnp.arange(len(deriv)) >= len(deriv) - 3
            
            return jnp.where(mask_left, stencil_left, 
                           jnp.where(mask_right, stencil_right, deriv))
        
        # 境界条件タイプに応じた処理
        if bc.type == bc.type.PERIODIC:
            # 周期境界条件
            first_deriv = first_deriv.at[0].set(first_deriv[-2])
            first_deriv = first_deriv.at[-1].set(first_deriv[1])
            second_deriv = second_deriv.at[0].set(second_deriv[-2])
            second_deriv = second_deriv.at[-1].set(second_deriv[1])
        
        elif bc.type == bc.type.DIRICHLET:
            # Dirichlet境界条件
            first_deriv = apply_boundary_deriv(first_deriv, order=1)
            second_deriv = apply_boundary_deriv(second_deriv, order=2)
            
        elif bc.type == bc.type.NEUMANN:
            # Neumann境界条件
            # 境界での勾配値を指定
            if callable(bc.value):
                grad_left = bc.value(0.0)
                grad_right = bc.value(1.0)
            else:
                grad_left = grad_right = bc.value
                
            # 一階微分の境界値を設定
            first_deriv = first_deriv.at[0].set(grad_left)
            first_deriv = first_deriv.at[-1].set(grad_right)
            
            # 二階微分の境界値を高精度に計算
            second_deriv = apply_boundary_deriv(second_deriv, order=2)
            
        return first_deriv, second_deriv
            
    def check_symmetry(self,
                      field: ArrayLike,
                      direction: str,
                      tolerance: float = 1e-10) -> bool:
        """
        離散化スキームの対称性をチェック
        
        Args:
            field: テスト場
            direction: チェックする方向
            tolerance: 許容誤差
            
        Returns:
            対称性が保たれているかのブール値
        """
        # オリジナルの場での微分を計算
        deriv1, deriv2 = self.discretize(field, direction)
        
        # 場を反転
        field_reversed = jnp.flip(field)
        deriv1_rev, deriv2_rev = self.discretize(field_reversed, direction)
        
        # 微分の反転との比較
        deriv1_check = jnp.allclose(-jnp.flip(deriv1), deriv1_rev, atol=tolerance)
        deriv2_check = jnp.allclose(jnp.flip(deriv2), deriv2_rev, atol=tolerance)
        
        return deriv1_check and deriv2_check
        
    def estimate_error_order(self,
                           field: ArrayLike,
                           direction: str,
                           exact_deriv1: ArrayLike,
                           exact_deriv2: ArrayLike) -> Tuple[float, float]:
        """
        数値的に誤差の次数を推定
        
        Args:
            field: テスト場
            direction: 評価する方向
            exact_deriv1: 厳密な一階微分
            exact_deriv2: 厳密な二階微分
            
        Returns:
            (一階微分の次数, 二階微分の次数) のタプル
        """
        # 現在の格子での誤差を計算
        deriv1, deriv2 = self.discretize(field, direction)
        error1 = jnp.linalg.norm(deriv1 - exact_deriv1)
        error2 = jnp.linalg.norm(deriv2 - exact_deriv2)
        
        # 格子を2倍細かくした場合の誤差を計算
        dx_fine = self.get_grid_spacing(direction) / 2
        x_fine = jnp.linspace(0, 1, 2*len(field)-1)
        field_fine = jnp.interp(x_fine, jnp.linspace(0, 1, len(field)), field)
        
        # 細かい格子での離散化
        deriv1_fine, deriv2_fine = self.discretize(field_fine, direction)
        error1_fine = jnp.linalg.norm(deriv1_fine[::2] - exact_deriv1)
        error2_fine = jnp.linalg.norm(deriv2_fine[::2] - exact_deriv2)
        
        # 誤差の次数を計算
        order1 = -jnp.log2(error1_fine/error1)
        order2 = -jnp.log2(error2_fine/error2)
        
        return float(order1), float(order2)