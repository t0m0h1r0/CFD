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
    relaxation_factor: float = 1.0  # 緩和係数
    preconditioner_type: str = 'diagonal'  # 前処理の種類
    stabilization_threshold: float = 1e-10  # 数値安定化パラメータ
    adaptive_tolerance: bool = True  # アダプティブ許容誤差の使用
    
    def validate(self):
        """設定値の検証"""
        super().validate()
        if self.relaxation_factor <= 0 or self.relaxation_factor > 2:
            raise ValueError(f"Invalid relaxation factor: {self.relaxation_factor}")
        if self.preconditioner_type not in ['diagonal', 'none']:
            raise ValueError(f"Invalid preconditioner type: {self.preconditioner_type}")

class PreconditionerStrategy(Protocol):
    """前処理のための抽象インターフェース"""
    def apply(
        self, 
        residual: ArrayLike,
        operator: CCDLaplacianSolver,
        scaling: float
    ) -> ArrayLike:
        """前処理を適用"""
        ...

class DiagonalPreconditioner:
    """対角スケーリング前処理"""
    
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def apply(
        residual: ArrayLike, 
        operator: CCDLaplacianSolver,
        scaling: float
    ) -> ArrayLike:
        """
        対角スケーリング前処理を適用
        
        Args:
            residual: 残差ベクトル
            operator: ラプラシアン演算子
            scaling: スケーリング係数
        
        Returns:
            前処理適用後の残差
        """
        return residual * scaling

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
        """
        CG法ソルバの初期化
        
        Args:
            config: ソルバー設定
            discretization: 空間離散化スキーム
            grid_manager: グリッド管理オブジェクト（オプション）
        """        
        # grid_managerが指定されている場合はCCDLaplacianSolverを生成
        if discretization is None and grid_manager is not None:
            discretization = CCDLaplacianSolver(grid_manager=grid_manager)
            
        super().__init__(config, discretization)
        self.grid_manager = grid_manager
        
        # 前処理の初期化
        self.preconditioner = (
            DiagonalPreconditioner() if config.preconditioner_type == 'diagonal'
            else None
        )

    def enforce_boundary_conditions(
        self,
        field: ArrayLike,
        operator: CCDLaplacianSolver
    ) -> ArrayLike:
        """
        境界条件を強制的に適用
        
        Args:
            field: 入力場
            operator: ラプラシアン演算子
            
        Returns:
            境界条件適用後の場
        """
        modified_field = field
        
        for direction, bc in operator.boundary_conditions.items():
            if bc.type == BCType.DIRICHLET:
                if direction == 'left':
                    modified_field = modified_field.at[0, :, :].set(bc.value)
                elif direction == 'right':
                    modified_field = modified_field.at[-1, :, :].set(bc.value)
                elif direction == 'bottom':
                    modified_field = modified_field.at[:, 0, :].set(bc.value)
                elif direction == 'top':
                    modified_field = modified_field.at[:, -1, :].set(bc.value)
                elif direction == 'front':
                    modified_field = modified_field.at[:, :, 0].set(bc.value)
                elif direction == 'back':
                    modified_field = modified_field.at[:, :, -1].set(bc.value)
        
        return modified_field

    def _solve_iteration(
        self,
        operator: CCDLaplacianSolver,
        x: ArrayLike,
        r: ArrayLike,
        p: ArrayLike,
        rz: ArrayLike,
        alpha_prev: ArrayLike,
        scaling: float
    ) -> Tuple[ArrayLike, ...]:
        """
        CG法の1反復を実行
        
        Args:
            operator: ラプラシアン演算子
            x: 現在の解
            r: 残差
            p: 探索方向
            rz: 内積値
            alpha_prev: 前回のステップ幅
            scaling: スケーリング係数
            
        Returns:
            更新された状態変数のタプル
        """
        # Ap の計算（符号に注意）
        Ap = operator.compute_laplacian(p)
        
        # α の計算（スケーリング込み）
        pAp = jnp.sum(p * Ap)
        alpha = self.config.relaxation_factor * rz * scaling / (
            pAp + self.config.stabilization_threshold
        )
        
        # 数値的安定化
        alpha = jnp.where(
            pAp > self.config.stabilization_threshold,
            alpha,
            alpha_prev
        )
        
        # 解と残差の更新
        x_new = x + alpha * p
        x_new = self.enforce_boundary_conditions(x_new, operator)
        r_new = r - alpha * Ap
        
        # 前処理適用
        z_new = (
            self.preconditioner.apply(r_new, operator, scaling)
            if self.preconditioner is not None else r_new * scaling
        )
        
        # 内積計算
        rz_new = jnp.sum(r_new * z_new)
        
        # β の計算と方向ベクトルの更新
        beta = jnp.where(
            rz > self.config.stabilization_threshold,
            rz_new / rz,
            0.0
        )
        p_new = z_new + beta * p
        
        # 残差ノルムの計算
        residual_norm = jnp.sqrt(jnp.sum(r_new * r_new))
        
        return x_new, r_new, p_new, rz_new, residual_norm, alpha

    def solve(
        self,
        operator: CCDLaplacianSolver,
        rhs: ArrayLike,
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """
        共役勾配法による解法
        
        Args:
            operator: ラプラシアン演算子
            rhs: 右辺ベクトル
            x0: 初期推定解
            
        Returns:
            解と収束情報のタプル
        """
        # 初期化
        x = x0 if x0 is not None else jnp.zeros_like(rhs)
        history = self.create_history_dict()
        
        # スケーリング係数の計算
        dx = operator.grid_manager.get_grid_spacing('x')
        scaling = dx * dx
        
        # 初期残差の計算（rhs = -Δu なので符号に注意）
        r = rhs - operator.compute_laplacian(x)
        initial_residual_norm = float(jnp.linalg.norm(r))
        
        # 前処理適用
        z = (
            self.preconditioner.apply(r, operator, scaling)
            if self.preconditioner is not None else r * scaling
        )
        p = z
        
        # 初期内積
        rz = jnp.sum(r * z)
        alpha = jnp.array(scaling)  # 初期α
        
        # 収束履歴
        residual_norms = []
        
        # メインのCG反復
        for iteration in range(self.config.max_iterations):
            # 1反復の実行
            x, r, p, rz, residual_norm, alpha = self._solve_iteration(
                operator, x, r, p, rz, alpha, scaling
            )
            
            # 残差履歴の記録
            if self.config.record_history:
                residual_norms.append(float(residual_norm))
            
            # 相対残差の計算
            relative_residual = residual_norm / initial_residual_norm
            
            # 収束判定
            if relative_residual < self.config.tolerance:
                history.update({
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_residual': float(residual_norm),
                    'relative_residual': float(relative_residual),
                    'residual_norms': residual_norms,
                    'initial_residual': initial_residual_norm
                })
                return x, history
            
            # 発散チェック
            if not jnp.isfinite(residual_norm):
                history.update({
                    'converged': False,
                    'iterations': iteration + 1,
                    'final_residual': float('inf'),
                    'relative_residual': float('inf'),
                    'residual_norms': residual_norms,
                    'initial_residual': initial_residual_norm
                })
                return x, history
        
        # 最大反復回数に達した場合
        history.update({
            'converged': False,
            'iterations': self.config.max_iterations,
            'final_residual': float(residual_norm),
            'relative_residual': float(relative_residual),
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
        """
        数値解の診断メトリクスを計算
        
        Args:
            solution: 数値解
            operator: ラプラシアン演算子  
            rhs: 右辺ベクトル
            
        Returns:
            診断情報の辞書
        """
        # ラプラシアンの計算
        laplacian = operator.compute_laplacian(solution)
        
        # 残差の計算（-Δu = f の形式）
        residual = rhs - laplacian
        
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
