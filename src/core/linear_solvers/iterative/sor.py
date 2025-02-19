from typing import Optional, Tuple, Dict, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import LinearSolverBase, LinearSolverConfig
from ...spatial_discretization.base import SpatialDiscretizationBase

class SORSolver(LinearSolverBase):
    """逐次過緩和法（SOR）による線形ソルバー"""
    
    def __init__(
        self, 
        config: LinearSolverConfig = LinearSolverConfig(),
        discretization: Optional[SpatialDiscretizationBase] = None,
        omega: float = 1.5
    ):
        """
        SORソルバーの初期化
        
        Args:
            config: ソルバー設定
            discretization: 空間離散化スキーム
            omega: 緩和パラメータ（1 < omega < 2）
        """
        super().__init__(config, discretization)
        self.omega = omega
    
    @partial(jax.jit, static_argnums=(0,))
    def solve(
        self, 
        operator: ArrayLike, 
        b: ArrayLike, 
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """
        SOR法による線形システムの解法
        
        Args:
            operator: 線形作用素（行列）
            b: 右辺ベクトル
            x0: 初期推定解
        
        Returns:
            解のタプル（解ベクトル、収束情報辞書）
        """
        # 初期値設定
        x = x0 if x0 is not None else jnp.zeros_like(b)
        diag = jnp.diag(operator)
        
        # 初期残差の計算
        def body_fun(state):
            x, iteration = state
            x_new = x.copy()
            
            for i in range(len(x_new)):
                r_i = b[i] - operator[i] @ x_new
                x_new = x_new.at[i].set(
                    x_new[i] + self.omega * r_i / diag[i]
                )
            
            return (x_new, iteration + 1)
        
        def cond_fun(state):
            x, iteration = state
            residual = b - operator @ x
            residual_norm = jnp.linalg.norm(residual)
            return jnp.logical_and(
                iteration < self.config.max_iterations,
                residual_norm >= self.config.tolerance
            )
        
        # 反復計算
        final_x, final_iteration = jax.lax.while_loop(
            cond_fun, 
            body_fun, 
            (x, 0)
        )
        
        # 最終残差の計算
        final_residual = jnp.linalg.norm(b - operator @ final_x)
        
        # 結果の辞書を構築
        history = {
            'converged': final_residual < self.config.tolerance,
            'iterations': final_iteration,
            'final_residual': final_residual
        }
        
        return final_x, history
    
    def optimize_relaxation_parameter(
        self, 
        operator: ArrayLike, 
        b: ArrayLike, 
        x0: Optional[ArrayLike] = None,
        omega_range: Tuple[float, float] = (1.0, 2.0),
        n_points: int = 10
    ) -> float:
        """
        最適な緩和パラメータを探索
        
        Args:
            operator: 線形作用素
            b: 右辺ベクトル
            x0: 初期推定解
            omega_range: 探索する緩和パラメータの範囲
            n_points: 探索点数
        
        Returns:
            最適な緩和パラメータ
        """
        def test_omega(omega: float):
            solver = SORSolver(
                config=LinearSolverConfig(
                    max_iterations=self.config.max_iterations,
                    tolerance=self.config.tolerance
                ),
                omega=omega
            )
            _, history = solver.solve(operator, b, x0)
            return history['iterations']
        
        # 異なるomegaでテスト
        omegas = jnp.linspace(omega_range[0], omega_range[1], n_points)
        iterations = jnp.array([test_omega(float(omega)) for omega in omegas])
        
        # 最小反復回数のomegaを返す
        return float(omegas[jnp.argmin(iterations)])