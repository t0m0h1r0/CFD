from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import CompactDifferenceBase
from ...common.types import Grid, BoundaryCondition, BCType
from ...common.grid import GridManager


class CombinedCompactDifference(CompactDifferenceBase):
    """Implementation of Combined Compact Difference (CCD) scheme."""

    def __init__(self,
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None,
                 order: int = 6):
        """
        Initialize CCD scheme.

        Args:
            grid_manager: Grid management object
            boundary_conditions: Dictionary of boundary conditions
                                 e.g. {'left': BC(...), 'right': BC(...), ...}
            order: Order of accuracy (default: 6)
        """
        coefficients = self._calculate_coefficients(order)
        super().__init__(grid_manager, boundary_conditions, coefficients)
        self.order = order

    def _calculate_coefficients(self, order: int) -> dict:
        """
        Calculate CCD coefficients for given order.

        Args:
            order: Order of accuracy

        Returns:
            Dictionary of coefficients
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
        Build CCD coefficient matrices for the given direction.
        This includes both interior points and boundary points.

        Args:
            direction: 'x' or 'y'

        Returns:
            Tuple of (lhs_matrix, rhs_matrix)
        """
        # dx が配列の場合に備え、スカラーに変換
        dx_val = self.grid_manager.get_grid_spacing(direction)
        if hasattr(dx_val, 'ndim') and dx_val.ndim > 0:
            dx = dx_val[0]
        else:
            dx = dx_val

        n_points = self.grid_manager.get_grid_points(direction)

        # 方向に対応する境界条件を取得
        # （left/right を x方向に、bottom/top を y方向に割り当てる例）
        if direction == 'x':
            bc_left = self.boundary_conditions.get('left', None)
            bc_right = self.boundary_conditions.get('right', None)
        elif direction == 'y':
            bc_left = self.boundary_conditions.get('bottom', None)
            bc_right = self.boundary_conditions.get('top', None)
        else:
            bc_left = None
            bc_right = None

        # CCDの係数
        a1, b1, c1 = (self.coefficients[k] for k in ['a1', 'b1', 'c1'])
        a2, b2, c2 = (self.coefficients[k] for k in ['a2', 'b2', 'c2'])

        # 行列サイズは (2*n_points) x (2*n_points)
        lhs = jnp.zeros((2 * n_points, 2 * n_points))
        # 右辺は (2*n_points) x (n_points)
        rhs = jnp.zeros((2 * n_points, n_points))

        # -----------------------------
        # 内部点の行列組み立て
        # -----------------------------
        for i in range(1, n_points - 1):
            # First derivative equation (行: 2*i)
            lhs = lhs.at[2*i, 2*i].set(1.0)
            lhs = lhs.at[2*i, 2*(i-1)].set(b1)
            lhs = lhs.at[2*i, 2*(i+1)].set(b1)
            lhs = lhs.at[2*i, 2*(i-1)+1].set(c1/dx)
            lhs = lhs.at[2*i, 2*(i+1)+1].set(-c1/dx)

            rhs = rhs.at[2*i, i+1].set(a1/(2*dx))
            rhs = rhs.at[2*i, i-1].set(-a1/(2*dx))

            # Second derivative equation (行: 2*i+1)
            lhs = lhs.at[2*i+1, 2*i+1].set(1.0)
            lhs = lhs.at[2*i+1, 2*(i-1)+1].set(c2)
            lhs = lhs.at[2*i+1, 2*(i+1)+1].set(c2)
            lhs = lhs.at[2*i+1, 2*(i-1)].set(b2/dx)
            lhs = lhs.at[2*i+1, 2*(i+1)].set(-b2/dx)

            rhs = rhs.at[2*i+1, i-1].set(a2/dx**2)
            rhs = rhs.at[2*i+1, i].set(-2*a2/dx**2)
            rhs = rhs.at[2*i+1, i+1].set(a2/dx**2)

        # -----------------------------
        # 境界点の行列組み立て
        # -----------------------------
        # 例: Dirichlet(導関数=0)とみなして、1次導関数/2次導関数ともに 0 に固定
        # あるいは boundary_conditions の type を見て切り替える
        #
        # i = 0 (left or bottom)
        if bc_left is not None and bc_left.type == BCType.DIRICHLET:
            # 1次導関数を0に固定
            lhs = lhs.at[2*0, :].set(0.0)
            lhs = lhs.at[2*0, 2*0].set(1.0)
            rhs = rhs.at[2*0, :].set(0.0)
            # 2次導関数を0に固定
            lhs = lhs.at[2*0+1, :].set(0.0)
            lhs = lhs.at[2*0+1, 2*0+1].set(1.0)
            rhs = rhs.at[2*0+1, :].set(0.0)
        else:
            # 必要に応じて Periodic や Neumann を設定する
            # とりあえず Dirichlet(0)と同じ処理にしておく
            lhs = lhs.at[2*0, :].set(0.0)
            lhs = lhs.at[2*0, 2*0].set(1.0)
            rhs = rhs.at[2*0, :].set(0.0)
            lhs = lhs.at[2*0+1, :].set(0.0)
            lhs = lhs.at[2*0+1, 2*0+1].set(1.0)
            rhs = rhs.at[2*0+1, :].set(0.0)

        # i = n_points - 1 (right or top)
        if bc_right is not None and bc_right.type == BCType.DIRICHLET:
            lhs = lhs.at[2*(n_points-1), :].set(0.0)
            lhs = lhs.at[2*(n_points-1), 2*(n_points-1)].set(1.0)
            rhs = rhs.at[2*(n_points-1), :].set(0.0)

            lhs = lhs.at[2*(n_points-1)+1, :].set(0.0)
            lhs = lhs.at[2*(n_points-1)+1, 2*(n_points-1)+1].set(1.0)
            rhs = rhs.at[2*(n_points-1)+1, :].set(0.0)
        else:
            # 同上
            lhs = lhs.at[2*(n_points-1), :].set(0.0)
            lhs = lhs.at[2*(n_points-1), 2*(n_points-1)].set(1.0)
            rhs = rhs.at[2*(n_points-1), :].set(0.0)

            lhs = lhs.at[2*(n_points-1)+1, :].set(0.0)
            lhs = lhs.at[2*(n_points-1)+1, 2*(n_points-1)+1].set(1.0)
            rhs = rhs.at[2*(n_points-1)+1, :].set(0.0)

        return lhs, rhs

    def solve_system(self,
                     lhs: ArrayLike,
                     rhs: ArrayLike,
                     field: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Solve the CCD system.

        Args:
            lhs: Left-hand side matrix, shape = (2*n_points, 2*n_points)
            rhs: Right-hand side matrix, shape = (2*n_points, n_points)
            field: Input field, shape = (n_points,)

        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        # RHSベクトルを組み立て (2*n_points, )
        rhs_vector = jnp.matmul(rhs, field)  # (2*n_points, n_points) x (n_points,) -> (2*n_points,)

        # JAXの線形ソルバで lhs * solution = rhs_vector を解く
        solution = jax.scipy.linalg.solve(lhs, rhs_vector)

        # 解から1次導関数と2次導関数を抽出
        n_points = len(field)
        first_deriv = solution[0::2]   # 偶数インデックス
        second_deriv = solution[1::2]  # 奇数インデックス

        return first_deriv, second_deriv

    def discretize(self,
                   field: ArrayLike,
                   direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute spatial derivatives using CCD scheme.

        Args:
            field: 1D array (n_points,) in the given direction
            direction: 'x' or 'y'

        Returns:
            Tuple of (first_derivative, second_derivative)
        """
        lhs, rhs = self.build_coefficient_matrices(direction)
        derivatives = self.solve_system(lhs, rhs, field)
        # solve_system後に apply_boundary_conditions を呼ぶ場合は、
        # そこでもう一度境界を再調整するロジックを入れることがある。
        derivatives = self.apply_boundary_conditions(field, derivatives, direction)
        return derivatives

    def apply_boundary_conditions(self,
                                  field: ArrayLike,
                                  derivatives: Tuple[ArrayLike, ArrayLike],
                                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Apply boundary conditions for CCD scheme after solving.

        Args:
            field: Input field
            derivatives: (first_derivative, second_derivative)
            direction: 'x' or 'y'

        Returns:
            Tuple of corrected (first_derivative, second_derivative)
        """
        first_deriv, second_deriv = derivatives

        # 例: periodic の場合はここで端点をコピー
        # ただし行列自体を周期境界対応にしないと厳密には不整合になる場合が多い
        # とりあえず既存のサンプルを残しておく
        if direction == 'x':
            bc_left = self.boundary_conditions.get('left', None)
            bc_right = self.boundary_conditions.get('right', None)
        else:  # direction == 'y'
            bc_left = self.boundary_conditions.get('bottom', None)
            bc_right = self.boundary_conditions.get('top', None)

        if bc_left and bc_left.type == BCType.PERIODIC:
            first_deriv = first_deriv.at[0].set(first_deriv[-2])
            second_deriv = second_deriv.at[0].set(second_deriv[-2])
        if bc_right and bc_right.type == BCType.PERIODIC:
            first_deriv = first_deriv.at[-1].set(first_deriv[1])
            second_deriv = second_deriv.at[-1].set(second_deriv[1])

        return first_deriv, second_deriv
