from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..common.types import Grid, BoundaryCondition
from ..common.grid import GridManager

class SpatialDiscretizationBase(ABC):
    """空間離散化スキームの基底クラス"""
    
    def __init__(self, 
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None):
        """
        空間離散化スキームの初期化
        
        Args:
            grid_manager: 格子管理オブジェクト
            boundary_conditions: 各境界の境界条件を定義する辞書
        """
        self.grid_manager = grid_manager
        self.boundary_conditions = boundary_conditions or {}
        
    @abstractmethod
    def discretize(self, 
                  field: ArrayLike,
                  direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        場の空間微分を計算
        
        Args:
            field: 入力場
            direction: 微分方向 ('x', 'y', 'z')
            
        Returns:
            (一階微分, 二階微分) のタプル
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
            field: 入力場
            derivatives: (一階微分, 二階微分) のタプル
            direction: 境界条件を適用する方向
            
        Returns:
            境界条件適用後の (一階微分, 二階微分) のタプル
        """
        pass
    
    def validate_direction(self, direction: str) -> None:
        """
        微分方向の妥当性チェック
        
        Args:
            direction: 指定された方向
            
        Raises:
            ValueError: 無効な方向が指定された場合
        """
        if direction not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid direction: {direction}. Must be 'x', 'y', or 'z'")
        
    def get_grid_spacing(self, direction: str) -> float:
        """
        指定方向の格子間隔を取得
        
        Args:
            direction: 格子間隔を取得する方向
            
        Returns:
            格子間隔
        """
        self.validate_direction(direction)
        return self.grid_manager.get_grid_spacing(direction)
    
    def get_grid_points(self, direction: str) -> int:
        """
        指定方向の格子点数を取得
        
        Args:
            direction: 格子点数を取得する方向
            
        Returns:
            格子点数
        """
        self.validate_direction(direction)
        return self.grid_manager.get_grid_points(direction)

class CompactDifferenceBase(SpatialDiscretizationBase):
    """コンパクト差分スキームの基底クラス"""
    
    def __init__(self,
                 grid_manager: GridManager,
                 boundary_conditions: Optional[dict[str, BoundaryCondition]] = None,
                 coefficients: Optional[dict] = None):
        """
        コンパクト差分スキームの初期化
        
        Args:
            grid_manager: 格子管理オブジェクト
            boundary_conditions: 境界条件の辞書
            coefficients: 差分係数の辞書
        """
        super().__init__(grid_manager, boundary_conditions)
        self.coefficients = coefficients or {}
        
    def build_coefficient_matrices(self, 
                                 direction: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        コンパクトスキームの係数行列を構築
        
        Args:
            direction: 係数行列を構築する方向
            
        Returns:
            (左辺行列, 右辺行列) のタプル
        """
        self.validate_direction(direction)
        dx = self.get_grid_spacing(direction)
        n_points = self.get_grid_points(direction)
        
        # 行列の初期化
        lhs = jnp.zeros((2*n_points, 2*n_points))
        rhs = jnp.zeros((2*n_points, n_points))
        
        return lhs, rhs
    
    @abstractmethod
    def solve_system(self,
                    lhs: ArrayLike,
                    rhs: ArrayLike,
                    field: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        コンパクト差分系を解く
        
        Args:
            lhs: 左辺行列
            rhs: 右辺行列
            field: 入力場
            
        Returns:
            (一階微分, 二階微分) のタプル
        """
        pass