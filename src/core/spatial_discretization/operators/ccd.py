from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Union

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...common.grid import GridManager
from ...common.types import BoundaryCondition, BCType


# -----------------------------------------------------------------------------
# 境界条件適用クラス
# -----------------------------------------------------------------------------
class BoundaryConditionApplicator:
    """
    境界条件の適用を担当するクラス。
    各軸の境界条件マッピングは以下の通り:
      - x: ('left', 'right')
      - y: ('front', 'back')
      - z: ('bottom', 'top')   ← z軸は高さを表す
    """
    def __init__(self, grid_manager: GridManager):
        self.grid_manager = grid_manager

    def apply_boundary_condition(
        self,
        field: ArrayLike,
        boundary_conditions: Dict[str, BoundaryCondition],
        direction: str
    ) -> ArrayLike:
        boundary_map = {
            'x': ('left', 'right'),
            'y': ('front', 'back'),
            'z': ('bottom', 'top')
        }
        try:
            left_key, right_key = boundary_map[direction]
        except KeyError:
            raise ValueError(f"Invalid direction: {direction}")

        default_bc = BoundaryCondition(type=BCType.DIRICHLET, value=0.0, location='default')
        left_bc = boundary_conditions.get(left_key, default_bc)
        right_bc = boundary_conditions.get(right_key, default_bc)

        modified_field = field.copy()
        # 左境界
        if left_bc.type == BCType.DIRICHLET:
            modified_field = modified_field.at[0].set(left_bc.value)
        elif left_bc.type == BCType.PERIODIC:
            modified_field = modified_field.at[0].set(modified_field[-1])
        # 右境界
        if right_bc.type == BCType.DIRICHLET:
            modified_field = modified_field.at[-1].set(right_bc.value)
        elif right_bc.type == BCType.PERIODIC:
            modified_field = modified_field.at[-1].set(modified_field[0])
        return modified_field


# -----------------------------------------------------------------------------
# 空間離散化基底クラス（discretize()メソッドは不要と判断）
# -----------------------------------------------------------------------------
class SpatialDiscretizationBase(ABC):
    """
    空間離散化の抽象基底クラス。
    CCDCompactDifferenceでは、solve_system（＝ compute_derivatives ）を直接利用するため、
    discretize() は不要としています。
    """
    def __init__(
        self,
        grid_manager: GridManager,
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None
    ):
        self.grid_manager = grid_manager
        self.boundary_conditions = boundary_conditions or {}
        self.boundary_condition_applicator = BoundaryConditionApplicator(grid_manager)

    def apply_boundary_conditions(
        self,
        field: ArrayLike,
        derivatives: Tuple[ArrayLike, ...],
        direction: str
    ) -> Tuple[ArrayLike, ...]:
        # CCDソルバ内で既に境界補正済みであればそのまま返す
        return derivatives


# -----------------------------------------------------------------------------
# CompactDifferenceBase（CCD固有の行列解法用抽象クラス）
# -----------------------------------------------------------------------------
class CompactDifferenceBase(SpatialDiscretizationBase, ABC):
    def __init__(
        self,
        grid_manager: GridManager,
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        coefficients: Optional[Dict] = None
    ):
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = coefficients or {}

    @abstractmethod
    def solve_system(self, field: ArrayLike) -> Tuple[ArrayLike, ...]:
        """
        連立方程式を解いて、各階微分を返す
        """
        raise NotImplementedError("solve_system は派生クラスで実装してください。")


# -----------------------------------------------------------------------------
# CCD（結合コンパクト差分）ソルバ実装
# -----------------------------------------------------------------------------
class CCDCompactDifference(CompactDifferenceBase):
    """
    CCD（結合コンパクト差分）による微分ソルバ。

    ・インスタンス生成時に、グリッド情報（点数、間隔）と指定係数から
      一次、二次、三次微分用の行列 (A, B) を構築します。
    ・各内部点では、以下の連立方程式
         A * D = γ · (B @ field)
      を解くことで高精度な微分値（D）を得ます。
    ・境界点では、一側差分により補正を行っています。
    """
    def __init__(
        self,
        grid_manager: GridManager,
        direction: str,
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        coefficients: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Args:
          grid_manager: グリッド管理オブジェクト
          direction: 微分を行う軸 ('x', 'y', 'z')  ※z軸は高さを表す
          boundary_conditions: 境界条件の辞書
          coefficients: 微分スキーム用係数の辞書
             既定値（上書き可能）:
               'first':  {'alpha': 15/16, 'beta': -7/16, 'gamma': 1/16}
               'second': {'alpha': 12/13, 'beta': -3/13, 'gamma': 1/13}
               'third':  {'alpha': 10/11, 'beta': -5/11, 'gamma': 1/11}
        """
        super().__init__(grid_manager, boundary_conditions)
        self.direction = direction
        self.dx = self.grid_manager.get_grid_spacing(direction)
        # もし dx が配列なら、1次元目の値をスカラーとして採用（x軸テストの場合）
        if hasattr(self.dx, "ndim") and self.dx.ndim > 0:
            self.dx = float(self.dx[0])
        self.n = self.grid_manager.get_grid_points(direction)
        # 既定の係数（上書き可能）
        default_coeff = {
            'first': {'alpha': 15/16, 'beta': -7/16, 'gamma': 1/16},
            'second': {'alpha': 12/13, 'beta': -3/13, 'gamma': 1/13},
            'third': {'alpha': 10/11, 'beta': -5/11, 'gamma': 1/11}
        }
        self.coefficients = {**default_coeff, **(coefficients or {})}
        self.A_first, self.B_first = self._build_first_derivative_matrices(self.n, self.dx, self.coefficients['first'])
        self.A_second, self.B_second = self._build_second_derivative_matrices(self.n, self.dx, self.coefficients['second'])
        self.A_third, self.B_third = self._build_third_derivative_matrices(self.n, self.dx, self.coefficients['third'])

    @staticmethod
    def _build_first_derivative_matrices(n: int, dx: float, coeff: Dict[str, float]) -> Tuple[ArrayLike, ArrayLike]:
        """
        一次微分用行列の構築

        内部（i = 1,..., n-2）:
          A[i,i] = alpha, A[i, i-1] = A[i, i+1] = beta
          B[i, i-1] = -1/(2dx), B[i, i+1] = 1/(2dx)
        境界（i = 0, n-1）は前方／後方差分で設定
        """
        # A行列
        A = jnp.eye(n)
        interior = jnp.arange(1, n - 1)
        A = A.at[interior, interior].set(coeff['alpha'])
        A = A.at[interior, interior - 1].set(coeff['beta'])
        A = A.at[interior, interior + 1].set(coeff['beta'])

        # B行列
        B = jnp.zeros((n, n))
        # 内部: 中心差分
        B = B.at[interior, interior - 1].set(-1/(2 * dx))
        B = B.at[interior, interior + 1].set(1/(2 * dx))
        # 境界: 前方／後方差分
        forward = jnp.zeros(n).at[0].set(-1/dx).at[1].set(1/dx)
        backward = jnp.zeros(n).at[-2].set(-1/dx).at[-1].set(1/dx)
        B = B.at[0, :].set(forward)
        B = B.at[-1, :].set(backward)
        return A, B

    @staticmethod
    def _build_second_derivative_matrices(n: int, dx: float, coeff: Dict[str, float]) -> Tuple[ArrayLike, ArrayLike]:
        """
        二次微分用行列の構築

        内部（i = 1,..., n-2）:
          A[i,i] = alpha, A[i, i-1] = A[i, i+1] = beta
          B[i, i-1] = 1/(dx²), B[i, i] = -2/(dx²), B[i, i+1] = 1/(dx²)
        境界は3点前方／後方差分のstencilで補正
        """
        # A行列
        A = jnp.eye(n)
        interior = jnp.arange(1, n - 1)
        A = A.at[interior, interior].set(coeff['alpha'])
        A = A.at[interior, interior - 1].set(coeff['beta'])
        A = A.at[interior, interior + 1].set(coeff['beta'])

        # B行列
        B = jnp.zeros((n, n))
        B = B.at[interior, interior - 1].set(1/(dx**2))
        B = B.at[interior, interior].set(-2/(dx**2))
        B = B.at[interior, interior + 1].set(1/(dx**2))
        # 境界: 前方差分（i=0）
        row0 = jnp.zeros(n)
        row0 = row0.at[0].set(1/(dx**2))
        row0 = row0.at[1].set(-2/(dx**2))
        row0 = row0.at[2].set(1/(dx**2))
        B = B.at[0, :].set(row0)
        # 境界: 後方差分（i=n-1）
        row_last = jnp.zeros(n)
        row_last = row_last.at[-3].set(1/(dx**2))
        row_last = row_last.at[-2].set(-2/(dx**2))
        row_last = row_last.at[-1].set(1/(dx**2))
        B = B.at[-1, :].set(row_last)
        return A, B

    @staticmethod
    def _build_third_derivative_matrices(n: int, dx: float, coeff: Dict[str, float]) -> Tuple[ArrayLike, ArrayLike]:
        """
        三次微分用行列の構築

        内部（i = 2,..., n-3）:
          A[i,i] = alpha, A[i, i-1] = A[i, i+1] = beta
          B[i, i-2] = 1/(2dx³), B[i, i-1] = -2/(2dx³),
          B[i, i+1] = 2/(2dx³), B[i, i+2] = -1/(2dx³)
        境界は前方／後方差分で補正（4点stencil）
        """
        # A行列
        A = jnp.eye(n)
        interior = jnp.arange(2, n - 2)
        A = A.at[interior, interior].set(coeff['alpha'])
        A = A.at[interior, interior - 1].set(coeff['beta'])
        A = A.at[interior, interior + 1].set(coeff['beta'])

        # B行列
        B = jnp.zeros((n, n))
        # 内部: 5点中心差分 stencil (分母は 2*dx³)
        B = B.at[interior, interior - 2].set(1/(2 * dx**3))
        B = B.at[interior, interior - 1].set(-2/(2 * dx**3))
        B = B.at[interior, interior + 1].set(2/(2 * dx**3))
        B = B.at[interior, interior + 2].set(-1/(2 * dx**3))
        # 境界: 前方差分（i=0,1）
        row0 = jnp.zeros(n)
        row0 = row0.at[0].set(1/(dx**3))
        row0 = row0.at[1].set(-3/(dx**3))
        row0 = row0.at[2].set(3/(dx**3))
        row0 = row0.at[3].set(-1/(dx**3))
        B = B.at[0, :].set(row0)
        row1 = jnp.zeros(n)
        row1 = row1.at[0].set(1/(dx**3))
        row1 = row1.at[1].set(-3/(dx**3))
        row1 = row1.at[2].set(3/(dx**3))
        row1 = row1.at[3].set(-1/(dx**3))
        B = B.at[1, :].set(row1)
        # 境界: 後方差分（i=n-2, n-1）
        row_last = jnp.zeros(n)
        row_last = row_last.at[-4].set(1/(dx**3))
        row_last = row_last.at[-3].set(-3/(dx**3))
        row_last = row_last.at[-2].set(3/(dx**3))
        row_last = row_last.at[-1].set(-1/(dx**3))
        B = B.at[-1, :].set(row_last)
        row_second_last = jnp.zeros(n)
        row_second_last = row_second_last.at[-4].set(1/(dx**3))
        row_second_last = row_second_last.at[-3].set(-3/(dx**3))
        row_second_last = row_second_last.at[-2].set(3/(dx**3))
        row_second_last = row_second_last.at[-1].set(-1/(dx**3))
        B = B.at[-2, :].set(row_second_last)
        return A, B

    def solve_system(self, field: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        事前構築済みのCCD行列を用いて、入力フィールドから一次・二次・三次微分を計算する。

        方程式:
           A * D = γ · (B @ field)
        """
        # 一次微分
        rhs_first = self.coefficients['first']['gamma'] * (self.B_first @ field)
        D_first = jnp.linalg.solve(self.A_first, rhs_first)
        # 二次微分
        rhs_second = self.coefficients['second']['gamma'] * (self.B_second @ field)
        D_second = jnp.linalg.solve(self.A_second, rhs_second)
        # 三次微分
        rhs_third = self.coefficients['third']['gamma'] * (self.B_third @ field)
        D_third = jnp.linalg.solve(self.A_third, rhs_third)
        return D_first, D_second, D_third

    def compute_derivatives(self, field: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        CCDソルバを用いて、一・二・三階微分を計算する。

        境界条件適用（必要に応じて内部で既に補正済みの想定）後の値を返す。
        """
        derivatives = self.solve_system(field)
        return self.apply_boundary_conditions(field, derivatives, self.direction)
