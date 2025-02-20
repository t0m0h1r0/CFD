# src/core/linear_solvers/gauss_seidel.py

from typing import Tuple, Optional, Dict, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
from .base import LinearSolverBase, LinearSolverConfig
from ..spatial_discretization.base import SpatialDiscretizationBase

class GaussSeidelSolver(LinearSolverBase):
    """ガウス=ザイデル法による反復解法"""
    
    def __init__(
        self,
        config: LinearSolverConfig = LinearSolverConfig(),
        discretization: Optional[SpatialDiscretizationBase] = None,
        omega: float = 1.5
    ):
        super().__init__(config, discretization)
        self.omega = omega

    def solve(
        self,
        operator: CCDLaplacianSolver,
        rhs: ArrayLike,
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """反復法による解法"""
        # 初期化
        field = x0 if x0 is not None else jnp.zeros_like(rhs)
        history = self.create_history_dict()

        @partial(jax.jit, static_argnums=(1,))
        def iteration_step(carry, op):
            """単一ステップの実行（JIT最適化）"""
            field, residual_norm = carry
            
            # ラプラシアン計算
            laplacian = op.compute_laplacian(field)
            
            # 残差計算
            residual = rhs - laplacian
            
            # ラプラシアン演算子の対角項の計算
            dx = op.grid_manager.get_grid_spacing('x')
            diag = -6.0 / (dx * dx)  # 7点ステンシルの中心係数
            
            # ガウス=ザイデル更新
            new_field = field + self.omega * residual / diag
            
            # 収束判定用の相対残差
            new_residual_norm = jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)
            
            return (new_field, new_residual_norm)

        @partial(jax.jit, static_argnums=(1,))
        def cond_fun(carry, op):
            """収束判定（JIT最適化）"""
            _, residual_norm = carry
            return residual_norm > self.config.tolerance

        # 初期残差の計算
        initial_laplacian = operator.compute_laplacian(field)
        initial_residual = jnp.linalg.norm(rhs - initial_laplacian) / jnp.linalg.norm(rhs)
        carry = (field, initial_residual)
        
        # メインの反復
        iteration = 0
        while (iteration < self.config.max_iterations and 
               cond_fun(carry, operator)):
            carry = iteration_step(carry, operator)
            field, residual_norm = carry
            
            # 履歴の更新
            if self.config.record_history:
                history['residual_history'].append(float(residual_norm))
            
            iteration += 1

        # 最終状態の取得
        final_field, final_residual = carry
        
        # 収束情報の更新
        history.update({
            'converged': final_residual < self.config.tolerance,
            'iterations': iteration,
            'final_residual': float(final_residual)
        })
        
        return final_field, history