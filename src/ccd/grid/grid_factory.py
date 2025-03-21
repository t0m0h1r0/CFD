"""
グリッドファクトリーモジュール

このモジュールは、適切な次元のグリッドを作成するためのファクトリ関数を提供します。
"""

from typing import Tuple, Union, Optional, List

from .grid1d import Grid1D
from .grid2d import Grid2D
from .grid3d import Grid3D
from ..common.base_grid import BaseGrid

class GridFactory:
    """
    計算格子を作成するファクトリクラス
    """
    
    @staticmethod
    def create_grid(
        dimension: int,
        nx_points: int,
        ny_points: Optional[int] = None,
        nz_points: Optional[int] = None,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None
    ) -> BaseGrid:
        """
        適切な次元のグリッドを作成
        
        Args:
            dimension: グリッドの次元 (1, 2, または 3)
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数（2D/3Dの場合）
            nz_points: z方向の格子点数（3Dの場合）
            x_range: x方向の範囲 (x_min, x_max)
            y_range: y方向の範囲 (y_min, y_max)（2D/3Dの場合）
            z_range: z方向の範囲 (z_min, z_max)（3Dの場合）
            
        Returns:
            作成されたグリッドオブジェクト
            
        Raises:
            ValueError: 次元に適切なパラメータが提供されていない場合
        """
        # デフォルト範囲の設定
        if y_range is None:
            y_range = x_range
        if z_range is None:
            z_range = x_range
        
        if dimension == 1:
            return Grid1D(nx_points, x_range)
        elif dimension == 2:
            if ny_points is None:
                raise ValueError("2Dグリッドにはny_pointsが必要です")
            return Grid2D(nx_points, ny_points, x_range, y_range)
        elif dimension == 3:
            if ny_points is None or nz_points is None:
                raise ValueError("3Dグリッドにはny_pointsとnz_pointsが必要です")
            return Grid3D(nx_points, ny_points, nz_points, x_range, y_range, z_range)
        else:
            raise ValueError(f"サポートされていない次元です: {dimension}")
    
    @staticmethod
    def create_uniform_grid(
        dimension: int,
        n_points: int,
        range_limits: Tuple[float, float] = (-1.0, 1.0)
    ) -> BaseGrid:
        """
        各方向が同じ点数と範囲の均一なグリッドを作成
        
        Args:
            dimension: グリッドの次元 (1, 2, または 3)
            n_points: 各方向の格子点数
            range_limits: 各方向の範囲 (min, max)
            
        Returns:
            作成されたグリッドオブジェクト
            
        Raises:
            ValueError: サポートされていない次元の場合
        """
        if dimension == 1:
            return Grid1D(n_points, range_limits)
        elif dimension == 2:
            return Grid2D(n_points, n_points, range_limits, range_limits)
        elif dimension == 3:
            return Grid3D(n_points, n_points, n_points, range_limits, range_limits, range_limits)
        else:
            raise ValueError(f"サポートされていない次元です: {dimension}")
    
    @staticmethod
    def convert_old_grid(old_grid) -> BaseGrid:
        """
        旧形式のグリッドを新しいクラスに変換
        
        Args:
            old_grid: 旧形式のグリッドオブジェクト
            
        Returns:
            新しいグリッドクラスのインスタンス
            
        Raises:
            ValueError: 変換できないグリッドの場合
        """
        # 旧グリッドの次元を判定
        is_2d = getattr(old_grid, 'is_2d', False)
        is_3d = getattr(old_grid, 'is_3d', False)
        
        if is_3d:
            return Grid3D(
                old_grid.nx_points, old_grid.ny_points, old_grid.nz_points,
                (old_grid.x_min, old_grid.x_max),
                (old_grid.y_min, old_grid.y_max),
                (old_grid.z_min, old_grid.z_max)
            )
        elif is_2d:
            return Grid2D(
                old_grid.nx_points, old_grid.ny_points,
                (old_grid.x_min, old_grid.x_max),
                (old_grid.y_min, old_grid.y_max)
            )
        else:
            return Grid1D(old_grid.n_points, (old_grid.x_min, old_grid.x_max))
