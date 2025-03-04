"""
グリッド設定モジュール

計算グリッドの設定と境界条件を管理するためのクラスを提供します。
ディリクレ境界条件の処理を改善し、積分定数による変動を抑制します。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import jax.numpy as jnp


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
    
    # 積分定数による補正を有効にするフラグ
    enable_boundary_correction: bool = True

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
        """
        if self.dirichlet_values is None:
            return False
        
        # coeffs=[1,0,0,0]の場合はFalse（基本関数の場合は境界条件不要）
        if self.coeffs == [1.0, 0.0, 0.0, 0.0]:
            return False
            
        return True

    @property
    def is_neumann(self) -> bool:
        """
        ノイマン境界条件が有効かどうかを返す
        """
        if self.neumann_values is None:
            return False
            
        # coeffs=[0,1,0,0]の場合はFalse（1階導関数の場合は境界条件不要）
        if self.coeffs == [0.0, 1.0, 0.0, 0.0]:
            return False
            
        return True
        
    def get_dirichlet_boundary_values(self) -> Tuple[float, float]:
        """
        最適化されたディリクレ境界値を取得
        
        積分定数による変動を抑制するため、両端の差分を使用
        
        Returns:
            (左端の修正境界値, 右端の修正境界値)
        """
        if not self.is_dirichlet or self.dirichlet_values is None:
            return 0.0, 0.0
            
        # 両端値の差分を使用
        left_val, right_val = self.dirichlet_values
        
        if self.enable_boundary_correction:
            # 変動を抑制するために差分を使用
            return (left_val - right_val) / 2.0, (-left_val + right_val) / 2.0
        else:
            # 従来通りの境界値を返す
            return left_val, right_val
            
    def get_neumann_boundary_values(self) -> Tuple[float, float]:
        """
        ノイマン境界値を取得
        
        Returns:
            (左端の境界値, 右端の境界値)
        """
        if not self.is_neumann or self.neumann_values is None:
            return 0.0, 0.0
            
        return self.neumann_values[0], self.neumann_values[1]
        
    def apply_boundary_correction(self, psi: jnp.ndarray) -> jnp.ndarray:
        """
        計算結果に対して境界条件による補正を適用
        
        Args:
            psi: 計算された関数値
            
        Returns:
            補正された関数値
        """
        if not self.is_dirichlet or self.dirichlet_values is None or not self.enable_boundary_correction:
            return psi
            
        # 両端の平均値を足して補正
        left_val, right_val = self.dirichlet_values
        correction = (left_val + right_val) / 2.0
        
        return psi + correction