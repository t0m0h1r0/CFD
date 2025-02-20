# src/core/linear_solvers/poisson_solver.py

from __future__ import annotations
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..spatial_discretization.base import SpatialDiscretizationBase
from ..spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
from ..common.grid import GridManager
from .base import LinearSolverBase, LinearSolverConfig

@dataclass
class PoissonSolverConfig(LinearSolverConfig):
    """ポアソン方程式専用のソルバー設定"""
    relaxation_factor: float = 1.2
    stabilization_threshold: float = 1e-10
    preconditioner_type: str = 'diagonal'
    adaptive_tolerance: bool = True

class PoissonPreconditioner:
    """
    ポアソン方程式用の前処理クラス
    
    高度に最適化された前処理戦略を提供
    """
    
    @staticmethod
    def diagonal_preconditioner(
        operator: CCDLaplacianSolver, 
        rhs: ArrayLike,
        stabilization_threshold: float = 1e-10
    ) -> ArrayLike:
        """
        対角スケーリング前処理
        
        Args:
            operator: ラプラシアン演算子
            rhs: 右辺ベクトル
            stabilization_threshold: 数値安定性のための微小係数
        
        Returns:
            対角スケーリングベクトル
        """
        # グリッドスペーシングの取得
        dx = operator.grid_manager.get_grid_spacing('x')
        
        # 7点ステンシルの中心係数に基づく対角スケーリング
        diag_scale = 6.0 / (dx * dx)
        
        # RHSのスケール
        rhs_scale = jnp.linalg.norm(rhs) + stabilization_threshold
        
        return diag_scale / rhs_scale

class PoissonSolverDiagnostics:
    """ポアソン方程式の数値解析診断クラス"""
    
    @staticmethod
    def compute_diagnostic_metrics(
        solution: ArrayLike, 
        operator: CCDLaplacianSolver, 
        rhs: ArrayLike
    ) -> Dict[str, float]:
        """
        数値解の診断メトリクスを計算
        
        Args:
            solution: 数値解
            operator: 空間離散化演算子
            rhs: 右辺ベクトル
        
        Returns:
            診断情報の辞書
        """
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
                'solution_min': float(jnp.min(solution)),
                'solution_max': float(jnp.max(solution))
            }
        except Exception as e:
            print(f"Diagnostic calculation error: {e}")
            return {key: float('nan') for key in [
                'residual_norm', 'relative_residual', 'max_residual', 
                'solution_magnitude', 'solution_min', 'solution_max'
            ]}

class PoissonSolver(LinearSolverBase):
    """
    高度に最適化されたJAXベースのポアソン方程式ソルバー
    
    特徴:
    - CCDLaplacianSolverとの緊密な統合
    - JAX最適化による高性能反復解法
    - ロバストな収束判定
    - 詳細な診断情報
    """
    
    def __init__(
        self,
        config: PoissonSolverConfig = PoissonSolverConfig(),
        discretization: Optional[SpatialDiscretizationBase] = None,
        grid_manager: Optional[GridManager] = None
    ):
        """
        ポアソン方程式ソルバーの初期化
        
        Args:
            config: ソルバー設定
            discretization: 空間離散化スキーム
            grid_manager: グリッド管理オブジェクト
        """
        # デフォルトの空間離散化スキームを作成
        if discretization is None and grid_manager is not None:
            from src.core.spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver
            discretization = CCDLaplacianSolver(grid_manager=grid_manager)
        
        super().__init__(config, discretization)
        self.config = config

    def solve(
        self,
        operator: CCDLaplacianSolver,
        rhs: ArrayLike,
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """
        ポアソン方程式の反復解法
        
        Args:
            operator: ラプラシアン演算子
            rhs: 右辺ベクトル
            x0: 初期推定解

        Returns:
            解と収束履歴の辞書
        """
        # グリッドスペーシングの取得
        dx = operator.grid_manager.get_grid_spacing('x')
        
        # 初期化
        field = x0 if x0 is not None else jnp.zeros_like(rhs)
        history = self.create_history_dict()
        
        # 適応的許容誤差
        adaptive_tolerance = (
            self.config.tolerance if self.config.adaptive_tolerance 
            else self.config.tolerance * jnp.linalg.norm(rhs)
        )
        
        # メインの反復ループ
        for iteration in range(self.config.max_iterations):
            try:
                # ラプラシアンの計算
                laplacian = operator.compute_laplacian(field)
                
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
                
                # 緩和付き更新
                field = field + (
                    self.config.relaxation_factor * 
                    safe_residual / diag_scale
                )
                
                # 相対残差の計算（数値安定性を考慮）
                residual_norm = (
                    jnp.linalg.norm(safe_residual) / 
                    (jnp.linalg.norm(rhs) + self.config.stabilization_threshold)
                )
                
                # 履歴の記録
                if self.config.record_history:
                    history.setdefault('residual_history', []).append(float(residual_norm))
                
                # 収束判定
                if (residual_norm < adaptive_tolerance or 
                    jnp.isnan(residual_norm) or 
                    iteration == self.config.max_iterations - 1):
                    
                    history.update({
                        'converged': residual_norm < adaptive_tolerance,
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

    def compute_diagnostics(
        self, 
        solution: ArrayLike,
        operator: CCDLaplacianSolver,
        rhs: ArrayLike
    ) -> Dict[str, float]:
        """
        詳細な数値解析診断
        
        Args:
            solution: 数値解
            operator: 空間離散化演算子
            rhs: 右辺ベクトル
        
        Returns:
            診断情報の辞書
        """
        return PoissonSolverDiagnostics.compute_diagnostic_metrics(
            solution, operator, rhs
        )
    
    def adaptive_tolerance_adjustment(
        self, 
        initial_tolerance: float, 
        iteration_history: Dict
    ) -> float:
        """
        反復回数に基づく許容誤差の適応的調整
        
        Args:
            initial_tolerance: 初期許容誤差
            iteration_history: 反復履歴
        
        Returns:
            調整後の許容誤差
        """
        iterations = iteration_history.get('iterations', 1)
        convergence_factor = 1.0 / (1.0 + 0.1 * iterations)
        return initial_tolerance * convergence_factor