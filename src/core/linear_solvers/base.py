from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..spatial_discretization.base import SpatialDiscretizationBase
from ..spatial_discretization.operators.ccd import CombinedCompactDifference


class LinearSolverBase(ABC):
    """線形ソルバーの抽象基底クラス"""
    
    def __init__(
        self, 
        discretization: Optional[SpatialDiscretizationBase] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        record_history: bool = False
    ):
        """
        線形ソルバーの初期化
        
        Args:
            discretization: 空間離散化スキーム
            max_iterations: 最大反復回数
            tolerance: 収束判定許容誤差
            record_history: 収束履歴の記録フラグ
        """
        self.discretization = discretization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.record_history = record_history
    
    @abstractmethod
    def solve(
        self, 
        operator: ArrayLike, 
        b: ArrayLike, 
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """
        線形システムを解く
        
        Args:
            operator: 線形作用素（行列またはCallable）
            b: 右辺ベクトル
            x0: 初期推定解（オプション）
        
        Returns:
            解のタプル（解ベクトル、収束情報辞書）
        """
        pass
    
    def create_history_dict(self) -> Dict[str, Union[bool, float, list]]:
        """
        収束履歴辞書を初期化
        
        Returns:
            収束履歴の初期辞書
        """
        return {
            'converged': False,
            'iterations': 0,
            'final_residual': float('inf'),
            'residual_history': [] if self.record_history else None
        }
    
    def check_convergence(
        self, 
        residual_norm: ArrayLike, 
        iteration: int
    ) -> ArrayLike:
        """
        JAX互換の収束判定
        
        Args:
            residual_norm: 残差ノルム
            iteration: 現在の反復回数
        
        Returns:
            収束判定結果
        """
        # JAX互換の収束判定
        return jnp.logical_or(
            residual_norm < self.tolerance,
            iteration >= self.max_iterations
        )