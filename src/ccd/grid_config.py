"""
グリッド設定モジュール

計算グリッドの設定と境界条件を管理するためのクラスを提供します。
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GridConfig:
    """グリッド設定を保持するデータクラス"""

    n_points: int  # グリッド点の数
    h: float  # グリッド幅

    # 境界条件の設定
    dirichlet_values: Optional[List[float]] = None  # ディリクレ境界条件値 [左端, 右端]
    neumann_values: Optional[List[float]] = None  # ノイマン境界条件値 [左端, 右端]
    
    # 係数の設定
    coeffs: Optional[List[float]] = None  # [a, b, c, d] 係数リスト

    def __post_init__(self):
        """初期化後の処理 - 係数設定と境界条件設定の処理"""
        # デフォルト係数の設定
        if self.coeffs is None:
            self.coeffs = [1.0, 0.0, 0.0, 0.0]
            
        # 単一値が指定された場合にリストに変換
        if self.dirichlet_values is not None and not isinstance(
            self.dirichlet_values, (list, tuple)
        ):
            self.dirichlet_values = [self.dirichlet_values, self.dirichlet_values]

        if self.neumann_values is not None and not isinstance(
            self.neumann_values, (list, tuple)
        ):
            self.neumann_values = [self.neumann_values, self.neumann_values]

    @property
    def is_dirichlet(self) -> bool:
        """
        ディリクレ境界条件が有効かどうかを返す
        coeffs=[1,0,0,0]の場合は常にFalse
        """
        if self.dirichlet_values is None:
            return False
        
        # coeffs=[1,0,0,0]の場合はFalse
        if self.coeffs == [1.0, 0.0, 0.0, 0.0]:
            return False
            
        return True

    @property
    def is_neumann(self) -> bool:
        """
        ノイマン境界条件が有効かどうかを返す
        coeffs=[0,1,0,0]の場合は常にFalse
        """
        if self.neumann_values is None:
            return False
            
        # coeffs=[0,1,0,0]の場合はFalse
        if self.coeffs == [0.0, 1.0, 0.0, 0.0]:
            return False
            
        return True
