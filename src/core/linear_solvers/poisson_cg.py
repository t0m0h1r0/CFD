from __future__ import annotations
from typing import Optional, Tuple, Dict, Union, Protocol
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..spatial_discretization.base import SpatialDiscretizationBase
from ..spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
from ..common.grid import GridManager
from ..common.types import BCType, BoundaryCondition
from .base import LinearSolverBase, LinearSolverConfig

@dataclass 
class CGSolverConfig(LinearSolverConfig):
    """共役勾配法ソルバの設定"""
    relaxation_factor: float = 0.8  # 緩和係数（0.8に変更）
    preconditioner_type: str = 'diagonal'  # 前処理の種類
    stabilization_threshold: float = 1e-12  # 数値安定化パラメータ（より小さく）
    adaptive_tolerance: bool = True  # アダプティブ許容誤差の使用
    max_relative_increase: float = 1.5  # 許容される残差の相対増加率
    
    def validate(self):
        """設定値の検証"""
        super().validate()
        if self.relaxation_factor <= 0 or self.relaxation_factor > 1:
            raise ValueError(f"Invalid relaxation factor: {self.relaxation_factor}")
        if self.preconditioner_type not in ['diagonal', 'none']:
            raise ValueError(f"Invalid preconditioner type: {self.preconditioner_type}")

class DiagonalPreconditioner:
    """対角スケーリング前処理"""
    
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def apply(
        residual: ArrayLike, 
        operator: CCDLaplacianSolver,
        scaling: float
    ) -> ArrayLike:
        """前処理を適用（改良版）"""
        dx = operator.grid_manager.get_grid_spacing('x')
        center_coef = 6.0 / (dx * dx)  # 7点ステンシルの中心係数
        return residual / center_coef

class ConjugateGradientSolver(LinearSolverBase):
    """
    JAX最適化された共役勾配法によるポアソン方程式ソルバ
    
    方程式: -Δu = f
    """
    
    def __init__(
        self,
        config: CGSolverConfig = CGSolverConfig(),
        discretization: Optional[SpatialDiscretizationBase] = None,
        grid_manager: Optional[GridManager] = None
    ):
        """CG法ソルバの初期化"""
        if discretization is None and grid_manager is not None:
            discretization = CCDLaplacianSolver(grid_manager=grid_manager)
            
        super().__init__(config, discretization)
        self.grid_manager = grid_manager
        self.preconditioner = (
            DiagonalPreconditioner() if config.preconditioner_type == 'diagonal'
            else None
        )

    def enforce_boundary_conditions(
        self,
        field: ArrayLike,
        operator: CCDLaplacianSolver
    ) -> ArrayLike:
        """境界条件の適用（改良版）"""
        modified_field = field
        
        for direction, bc in operator.boundary_conditions.items():
            if bc.type == BCType.DIRICHLET:
                # 境界値を強制（同次条件を想定）
                if direction == 'left':
                    modified_field = modified_field.at[0, :, :].set(0.0)
                elif direction == 'right':
                    modified_field = modified_field.at[-1, :, :].set(0.0)
                elif direction == 'bottom':
                    modified_field = modified_field.at[:, 0, :].set(0.0)
                elif direction == 'top':
                    modified_field = modified_field.at[:, -1, :].set(0.0)
                elif direction == 'front':
                    modified_field = modified_field.at[:, :, 0].set(0.0)
                elif direction == 'back':
                    modified_field = modified_field.at[:, :, -1].set(0.0)
        
        return modified_field

    def adaptive_tolerance_adjustment(
        self, 
        initial_tolerance: float, 
        iteration_history: Dict
    ) -> float:
        """適応的許容誤差の調整"""
        if not self.config.adaptive_tolerance:
            return initial_tolerance
            
        iterations = iteration_history.get('iterations', 1)
        
        # 指数減衰による適応調整
        decay_rate = -iterations / 100.0
        factor = jnp.exp(jnp.maximum(decay_rate, -10.0))  # 下限設定
        adjusted_tol = initial_tolerance * (1.0 + factor)
        
        # 下限と上限の設定
        min_tol = initial_tolerance * 1e-2
        max_tol = initial_tolerance * 10.0
        return jnp.clip(adjusted_tol, min_tol, max_tol)

    @partial(jax.jit, static_argnums=(0,1))
    def _solve_iteration(
        self,
        operator: CCDLaplacianSolver,
        x: ArrayLike,
        r: ArrayLike,
        p: ArrayLike,
        rz: ArrayLike,
        alpha_prev: ArrayLike,
        scaling: float,
        rhs: ArrayLike
    ) -> Tuple[ArrayLike, ...]:
        """CG法の1反復を実行（改良版）"""
        # ラプラシアン計算
        Ap = operator.compute_laplacian(p)
        
        # α の計算と安定化
        pAp = jnp.sum(p * Ap)
        alpha = jnp.where(
            pAp > self.config.stabilization_threshold,
            self.config.relaxation_factor * rz / (pAp + self.config.stabilization_threshold),
            0.0
        )
        
        # 解の更新と境界条件の適用
        x_new = x + alpha * p
        x_new = self.enforce_boundary_conditions(x_new, operator)
        
        # 残差の更新
        r_new = r - alpha * Ap
        
        # 前処理適用
        z_new = self.preconditioner.apply(r_new, operator, scaling)
        rz_new = jnp.sum(r_new * z_new)
        
        # β の計算と安定化
        beta = jnp.where(
            rz > self.config.stabilization_threshold,
            rz_new / rz,
            0.0
        )
        
        # 方向ベクトルの更新
        p_new = z_new + beta * p
        
        # 残差の正規化
        residual_norm = jnp.sqrt(jnp.sum(r_new * r_new))
        rhs_norm = jnp.sqrt(jnp.sum(rhs * rhs))
        normalized_residual = residual_norm / (rhs_norm + self.config.stabilization_threshold)
        
        return x_new, r_new, p_new, rz_new, normalized_residual, alpha

    def solve(
        self,
        operator: CCDLaplacianSolver,
        rhs: ArrayLike,
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """共役勾配法による解法（改良版）"""
        # 初期化
        x = x0 if x0 is not None else jnp.zeros_like(rhs)
        history = self.create_history_dict()
        
        # rhsの符号を反転（-Δu = f の形式）
        rhs = -rhs
        
        # スケーリング係数
        dx = operator.grid_manager.get_grid_spacing('x')
        scaling = dx * dx
        
        # 初期残差
        r = rhs + operator.compute_laplacian(x)
        initial_residual_norm = float(jnp.linalg.norm(r))
        
        # 前処理適用
        z = self.preconditioner.apply(r, operator, scaling)
        p = z
        
        # 初期内積
        rz = jnp.sum(r * z)
        alpha = jnp.array(scaling)
        
        # 収束履歴
        residual_norms = []
        prev_residual_norm = initial_residual_norm
        
        # メインのCG反復
        for iteration in range(self.config.max_iterations):
            # 1反復の実行
            x, r, p, rz, residual_norm, alpha = self._solve_iteration(
                operator, x, r, p, rz, alpha, scaling, rhs
            )
            
            # 残差履歴の記録
            if self.config.record_history:
                residual_norms.append(float(residual_norm))
            
            # 発散チェック
            if not jnp.isfinite(residual_norm):
                history.update({
                    'converged': False,
                    'iterations': iteration + 1,
                    'final_residual': float('inf'),
                    'residual_norms': residual_norms,
                    'initial_residual': initial_residual_norm
                })
                return x, history
            
            # 残差増加のチェック
            if residual_norm > self.config.max_relative_increase * prev_residual_norm:
                history.update({
                    'converged': False,
                    'iterations': iteration + 1,
                    'final_residual': float(residual_norm),
                    'residual_norms': residual_norms,
                    'initial_residual': initial_residual_norm
                })
                return x, history
            
            # 収束判定
            if residual_norm < self.config.tolerance:
                history.update({
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_residual': float(residual_norm),
                    'residual_norms': residual_norms,
                    'initial_residual': initial_residual_norm
                })
                return x, history
            
            prev_residual_norm = residual_norm
        
        # 最大反復回数に達した場合
        history.update({
            'converged': False,
            'iterations': self.config.max_iterations,
            'final_residual': float(residual_norm),
            'residual_norms': residual_norms,
            'initial_residual': initial_residual_norm
        })
        
        return x, history

    def compute_diagnostics(
        self,
        solution: ArrayLike,
        operator: CCDLaplacianSolver,
        rhs: ArrayLike
    ) -> Dict[str, float]:
        """数値解の診断メトリクスを計算（改良版）"""
        # 符号を反転したrhsで計算
        rhs = -rhs
        
        # ラプラシアンの計算
        laplacian = operator.compute_laplacian(solution)
        
        # 残差の計算
        residual = rhs + laplacian
        
        # スケーリングの適用
        dx = operator.grid_manager.get_grid_spacing('x')
        scaling = dx * dx
        
        # 診断メトリクス
        metrics = {
            'residual_norm': float(jnp.linalg.norm(residual)),
            'relative_residual': float(
                jnp.linalg.norm(residual) / 
                (jnp.linalg.norm(rhs) + self.config.stabilization_threshold)
            ),
            'solution_magnitude': float(jnp.linalg.norm(solution)),
            'solution_min': float(jnp.min(solution)),
            'solution_max': float(jnp.max(solution)),
            'laplacian_magnitude': float(jnp.linalg.norm(laplacian))
        }
        
        return metrics
