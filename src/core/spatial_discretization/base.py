from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..common.types import BoundaryCondition
from ..common.grid import GridManager

class SpatialDiscretizationBase(ABC):
    """空間離散化スキームの基底クラス"""
    
    def __init__(self, 
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None):
        """
        空間離散化スキームを初期化
        
        Args:
            grid_manager: グリッド管理オブジェクト
            boundary_conditions: 境界条件の辞書（オプション）
        """
        self.grid_manager = grid_manager
        self.boundary_conditions = boundary_conditions or {}
        
    @abstractmethod
    def discretize(self, 
                  field: ArrayLike,
                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        与えられた場の空間微分を計算
        
        Args:
            field: 微分する場
            direction: 微分方向 ('x', 'y', または 'z')
            
        Returns:
            一階および二階微分のタプル
        """
        pass
    
    @abstractmethod
    def apply_boundary_conditions(self,
                                field: ArrayLike,
                                derivatives: Tuple[ArrayLike, ArrayLike],
                                direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        計算された微分に境界条件を適用
        
        Args:
            field: 元の場
            derivatives: 計算された微分のタプル (一階微分, 二階微分)
            direction: 微分方向
            
        Returns:
            境界条件適用後の微分タプル
        """
        pass