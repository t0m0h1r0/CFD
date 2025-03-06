"""
2次元グリッド設定モジュール

1次元のGridConfigを拡張し、2次元問題に対応します。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import cupy as cp

from grid_config import GridConfig  # 1次元のGridConfigをインポート


@dataclass
class GridConfig2D:
    """2次元グリッド設定を保持するデータクラス（CuPy対応）"""

    nx_points: int  # x方向のグリッド点の数
    ny_points: int  # y方向のグリッド点の数
    hx: float  # x方向のグリッド幅
    hy: float  # y方向のグリッド幅

    # 境界条件設定 (辺ごとに定義)
    # 各辺の境界値は、その辺に沿った配列として指定
    boundary_values: Optional[Dict[str, List[float]]] = None
    
    # 係数の設定
    coeffs: Optional[List[float]] = None  # [a, b, c, d] 係数リスト

    # 積分定数による補正を有効にするフラグ
    enable_boundary_correction: bool = True

    # 1次元グリッド設定のキャッシュ
    _grid_x: Optional[GridConfig] = None
    _grid_y: Optional[GridConfig] = None

    def __post_init__(self):
        """初期化後の処理 - 係数設定と境界条件設定の正規化"""
        # デフォルト係数の設定
        if self.coeffs is None:
            self.coeffs = [1.0, 0.0, 0.0, 0.0]

        # 境界条件の初期化
        if self.boundary_values is None:
            self.boundary_values = {
                "left": [0.0] * self.ny_points,   # 左辺境界
                "right": [0.0] * self.ny_points,  # 右辺境界
                "bottom": [0.0] * self.nx_points, # 下辺境界
                "top": [0.0] * self.nx_points,    # 上辺境界
            }

    def get_grid_x(self) -> GridConfig:
        """x方向の1次元グリッド設定を取得"""
        if self._grid_x is None:
            self._grid_x = GridConfig(
                n_points=self.nx_points,
                h=self.hx,
                dirichlet_values=[0.0, 0.0],  # ダミー値（各行ごとに適切な値を設定）
                coeffs=self.coeffs,
                enable_boundary_correction=self.enable_boundary_correction
            )
        return self._grid_x

    def get_grid_y(self) -> GridConfig:
        """y方向の1次元グリッド設定を取得"""
        if self._grid_y is None:
            self._grid_y = GridConfig(
                n_points=self.ny_points,
                h=self.hy,
                dirichlet_values=[0.0, 0.0],  # ダミー値（各列ごとに適切な値を設定）
                coeffs=self.coeffs,
                enable_boundary_correction=self.enable_boundary_correction
            )
        return self._grid_y

    def is_dirichlet(self) -> bool:
        """ディリクレ境界条件が有効かどうかを返す"""
        # ディリクレ条件が有効なのは、係数が純粋なfを表す場合
        return self.coeffs != [1.0, 0.0, 0.0, 0.0]

    def is_neumann(self) -> bool:
        """ノイマン境界条件が有効かどうかを返す"""
        # ノイマン条件が有効なのは、係数が純粋なf'