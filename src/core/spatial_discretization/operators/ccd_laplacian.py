from __future__ import annotations
from typing import Dict, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import CompactDifferenceBase
from ...common.grid import GridManager
from ...common.types import BoundaryCondition, BCType
from .ccd import CombinedCompactDifference

class CCDLaplacianSolver(CompactDifferenceBase):
    """高精度CCDスキームを用いたラプラシアンソルバー"""
    
    def __init__(
        self, 
        grid_manager: GridManager, 
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        order: int = 8
    ):
        """
        CCDラプラシアンソルバーの初期化
        
        Args:
            grid_manager: グリッド管理オブジェクト
            boundary_conditions: 境界条件の辞書（オプション）
            order: 精度次数（デフォルト: 8）
        """
        super().__init__(grid_manager, boundary_conditions)
        
        # CCD差分計算機の初期化
        self.ccd = CombinedCompactDifference(
            grid_manager=grid_manager,
            boundary_conditions=boundary_conditions,
            order=order
        )
        self.order = order

    @partial(jax.jit, static_argnums=(0,))
    def compute_laplacian(self, field: ArrayLike) -> ArrayLike:
        """
        ラプラシアンの計算
        
        Args:
            field: 入力フィールド
            
        Returns:
            ラプラシアン
        """
        # 各方向の二階微分を計算
        _, laplacian_x = self.discretize(field, 'x')
        _, laplacian_y = self.discretize(field, 'y')
        _, laplacian_z = self.discretize(field, 'z')
        
        # ラプラシアンの合成
        return laplacian_x + laplacian_y + laplacian_z

    @partial(jax.jit, static_argnums=(0,2))
    def discretize(
        self, 
        field: ArrayLike,
        direction: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        指定方向の微分を計算
        
        Args:
            field: 入力フィールド
            direction: 微分方向 ('x', 'y', 'z')
            
        Returns:
            (一階微分, 二階微分)のタプル
        """
        # CCDに委譲
        return self.ccd.discretize(field, direction)

    def solve_system(
        self,
        lhs: ArrayLike, 
        rhs: ArrayLike,
        field: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        連立方程式の求解
        
        Args:
            lhs: 左辺行列
            rhs: 右辺行列
            field: 入力フィールド
            
        Returns:
            求解された一階微分と二階微分のタプル
        """
        # CCDに委譲
        return self.ccd.solve_system(lhs, rhs, field)

    def verify_solution(
        self,
        numerical_solution: ArrayLike,
        exact_solution: ArrayLike
    ) -> Dict[str, float]:
        """
        数値解の検証
        
        Args:
            numerical_solution: 数値解
            exact_solution: 厳密解
            
        Returns:
            検証結果の辞書
        """
        # 相対誤差の計算
        error = jnp.linalg.norm(numerical_solution - exact_solution)
        relative_error = error / (jnp.linalg.norm(exact_solution) + 1e-10)
        
        # 最大誤差の計算
        max_error = jnp.max(jnp.abs(numerical_solution - exact_solution))
        
        # ラプラシアンの計算と比較
        numerical_laplacian = self.compute_laplacian(numerical_solution)
        exact_laplacian = self.compute_laplacian(exact_solution)
        laplacian_error = jnp.linalg.norm(numerical_laplacian - exact_laplacian)
        
        return {
            'relative_error': float(relative_error),
            'max_error': float(max_error),
            'laplacian_error': float(laplacian_error)
        }