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
    """高度に最適化された数値安定性を考慮したガウス=ザイデル法"""
    
    def __init__(
        self,
        config: LinearSolverConfig = LinearSolverConfig(),
        discretization: Optional[SpatialDiscretizationBase] = None,
        omega: float = 1.2,  # より慎重な緩和パラメータ
        stabilization_factor: float = 1e-10  # 数値安定性のための微小係数
    ):
        """
        ロバストなガウス=ザイデル法ソルバーの初期化
        
        Args:
            config: ソルバー設定
            discretization: 空間離散化スキーム
            omega: 緩和パラメータ
            stabilization_factor: 数値安定性のための微小係数
        """
        super().__init__(config, discretization)
        self.omega = omega
        self.stabilization_factor = stabilization_factor

    def _diagonal_preconditioner(self, operator: CCDLaplacianSolver, rhs: ArrayLike) -> ArrayLike:
        """
        対角前処理の計算
        
        Args:
            operator: ラプラシアン演算子
            rhs: 右辺ベクトル
        
        Returns:
            前処理された対角スケール
        """
        # グリッドスペーシングの取得
        dx = operator.grid_manager.get_grid_spacing('x')
        
        # 対角スケーリング
        # 7点ステンシルの中心係数に基づく
        diag_scale = 6.0 / (dx * dx)
        
        # RHSのスケール
        rhs_scale = jnp.linalg.norm(rhs) + self.stabilization_factor
        
        return diag_scale / rhs_scale

    def solve(
        self,
        operator: CCDLaplacianSolver,
        rhs: ArrayLike,
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """
        高度に最適化された反復解法
        
        Args:
            operator: ラプラシアン演算子
            rhs: 右辺ベクトル
            x0: 初期推定解

        Returns:
            解と収束履歴の辞書
        """
        # 前処理のセットアップ
        preconditioner = self._diagonal_preconditioner(operator, rhs)
        
        # 初期化
        field = x0 if x0 is not None else jnp.zeros_like(rhs)
        history = self.create_history_dict()
        
        # グリッドスペーシングと基本パラメータ
        dx = operator.grid_manager.get_grid_spacing('x')
        
        @partial(jax.jit, static_argnums=(1,))
        def iteration_step(field, op):
            """
            高度に最適化された単一反復ステップ
            
            Args:
                field: 現在の場
                op: 空間離散化演算子
            
            Returns:
                更新された場と正規化残差
            """
            # ラプラシアンの計算
            laplacian = op.compute_laplacian(field)
            
            # 残差の計算
            residual = rhs - laplacian
            
            # 対角スケーリング（7点ステンシル）
            diag_scale = 6.0 / (dx * dx)
            
            # 安全な更新
            safe_residual = jnp.where(
                jnp.isfinite(residual),
                residual,
                jnp.zeros_like(residual)
            )
            
            # 緩和付き更新（前処理を考慮）
            updated_field = field + self.omega * (safe_residual / diag_scale)
            
            # 相対残差の計算（数値安定性を考慮）
            residual_norm = (
                jnp.linalg.norm(safe_residual) / 
                (jnp.linalg.norm(rhs) + self.stabilization_factor)
            )
            
            return updated_field, residual_norm
        
        # メインの反復ループ
        for iteration in range(self.config.max_iterations):
            try:
                field, residual_norm = iteration_step(field, operator)
                
                # 履歴の記録
                if self.config.record_history:
                    history['residual_history'].append(float(residual_norm))
                
                # 収束判定
                if (residual_norm < self.config.tolerance or 
                    jnp.isnan(residual_norm) or 
                    iteration == self.config.max_iterations - 1):
                    
                    history.update({
                        'converged': residual_norm < self.config.tolerance,
                        'iterations': iteration + 1,
                        'final_residual': float(residual_norm)
                    })
                    
                    return field, history
            
            except Exception as e:
                print(f"Iteration {iteration} error: {e}")
                break
        
        # 収束失敗時の処理
        history.update({
            'converged': False,
            'iterations': self.config.max_iterations,
            'final_residual': float('inf')
        })
        
        return field, history

    def diagnostics(
        self, 
        operator: CCDLaplacianSolver, 
        rhs: ArrayLike,
        solution: ArrayLike
    ) -> Dict[str, float]:
        """
        数値解の診断情報を提供

        Args:
            operator: ラプラシアン演算子
            rhs: 右辺ベクトル
            solution: 数値解

        Returns:
            診断情報の辞書
        """
        # 安全な診断情報計算
        try:
            # ラプラシアンの計算
            laplacian = operator.compute_laplacian(solution)
            
            # 残差の計算
            residual = rhs - laplacian
            
            return {
                'residual_norm': float(jnp.linalg.norm(residual)),
                'relative_residual': float(
                    jnp.linalg.norm(residual) / (jnp.linalg.norm(rhs) + 1e-10)
                ),
                'max_residual': float(jnp.max(jnp.abs(residual))),
                'solution_magnitude': float(jnp.linalg.norm(solution)),
                'min_solution': float(jnp.min(solution)),
                'max_solution': float(jnp.max(solution))
            }
        except Exception as e:
            print(f"Diagnostics error: {e}")
            return {
                'residual_norm': float('nan'),
                'relative_residual': float('nan'),
                'max_residual': float('nan'),
                'solution_magnitude': float('nan'),
                'min_solution': float('nan'),
                'max_solution': float('nan')
            }