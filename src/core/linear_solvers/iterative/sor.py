from typing import Optional, Dict
import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from ...spatial_discretization.base import SpatialDiscretizationBase
from ...common.types import BCType, BoundaryCondition

# 既存のCCD離散化クラス（前述の CombinedCompactDifference ）はそのまま利用

class SORCCDPoissonSolver:
    """
    CCD離散化を用い、SOR法でポアソン方程式 (Δu = f) を解くソルバ。
    """
    def __init__(self,
                 grid_manager,
                 boundary_conditions: Optional[Dict[str, 'BoundaryCondition']] = None,
                 order: int = 6,
                 omega: float = 1.5,
                 tol: float = 1e-6,
                 max_iter: int = 10000,
                 config: Optional[Dict] = None,
                 discretization: Optional[SpatialDiscretizationBase] = None):
        # config が渡された場合、そこから各パラメータを上書きする
        if config is not None:
            grid_manager = config.get('grid_manager', grid_manager)
            boundary_conditions = config.get('boundary_conditions', boundary_conditions)
            order = config.get('order', order)
            omega = config.get('omega', omega)
            tol = config.get('tol', tol)
            max_iter = config.get('max_iter', max_iter)
            discretization = config.get('discretization', discretization)

        self.grid_manager = grid_manager
        self.boundary_conditions = boundary_conditions
        self.omega = omega
        self.tol = tol
        self.max_iter = max_iter
        self.discretization = discretization

    def solve(self, source: ArrayLike, initial_guess: Optional[ArrayLike] = None) -> ArrayLike:
        """
        ポアソン方程式 Δu = f をSOR法により解く。
        
        Args:
            source: 右辺項 f （グリッド上の配列）
            initial_guess: 初期解（与えなければゼロ初期）
        
        Returns:
            解 u （グリッド上の配列）
        """
        # 初期解の設定
        if initial_guess is None:
            u = jnp.zeros_like(source)
        else:
            u = initial_guess

        # 格子間隔（例：均一格子の場合）
        dx = self.grid_manager.get_grid_spacing('x')
        dy = self.grid_manager.get_grid_spacing('y')
        # ここでは簡単のため、中央差分と同等の対角項を仮定
        # CCDの離散化で用いる係数に合わせて調整する必要があります
        diag_coef = - (2 / dx**2 + 2 / dy**2)
        
        # SOR反復の状態（iteration count, 解, 残差のノルム）
        def cond_fun(state):
            iter_count, u, res_norm = state
            return jnp.logical_and(iter_count < self.max_iter, res_norm > self.tol)

        def body_fun(state):
            iter_count, u_old, _ = state
            # CCD離散化によるラプラシアンの計算（x, y方向それぞれ）
            _, u_xx = self.ccd.discretize(u_old, 'x')
            _, u_yy = self.ccd.discretize(u_old, 'y')
            laplacian = u_xx + u_yy
            
            # 残差（右辺と離散ラプラシアンの差）
            res = source - laplacian
            
            # SOR更新：各格子点で更新を行う（境界は後で条件適用）
            # ※ここでは全体をベクトル化して更新していますが、実際は格子内部のみ更新するように工夫する必要があります
            u_new = u_old + self.omega / diag_coef * res
            
            # 境界条件の適用（CCDクラスの仕組みを再利用するか、ここで直接操作）
            u_new = self._apply_boundary(u_new)
            
            res_norm = jnp.linalg.norm(res)
            return (iter_count + 1, u_new, res_norm)

        # 初期状態
        init_state = (0, u, jnp.inf)
        _, u_final, _ = lax.while_loop(cond_fun, body_fun, init_state)
        return u_final

    def _apply_boundary(self, u: ArrayLike) -> ArrayLike:
        """
        CCDで設定した境界条件を適用する。
        ここでは単純にDirichlet境界（値固定）などを想定。
        """
        # 例：x方向の左右境界にDirichlet条件が設定されている場合
        if self.boundary_conditions is not None:
            if 'left' in self.boundary_conditions:
                bc_left = self.boundary_conditions['left']
                if bc_left.type == BCType.DIRICHLET:
                    u = u.at[:, 0].set(bc_left.value)
            if 'right' in self.boundary_conditions:
                bc_right = self.boundary_conditions['right']
                if bc_right.type == BCType.DIRICHLET:
                    u = u.at[:, -1].set(bc_right.value)
            # y方向も同様に適用
            if 'bottom' in self.boundary_conditions:
                bc_bottom = self.boundary_conditions['bottom']
                if bc_bottom.type == BCType.DIRICHLET:
                    u = u.at[0, :].set(bc_bottom.value)
            if 'top' in self.boundary_conditions:
                bc_top = self.boundary_conditions['top']
                if bc_top.type == BCType.DIRICHLET:
                    u = u.at[-1, :].set(bc_top.value)
        return u
