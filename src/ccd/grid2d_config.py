"""
2次元グリッド設定モジュール

2次元の計算グリッドの設定と境界条件を管理するためのクラス
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import cupy as cp
import numpy as np


@dataclass
class Grid2DConfig:
    """2次元グリッド設定を保持するデータクラス（CuPy対応）"""

    nx: int  # x方向のグリッド点の数
    ny: int  # y方向のグリッド点の数
    hx: float  # x方向のグリッド幅
    hy: float  # y方向のグリッド幅

    # 境界条件の設定
    dirichlet_values_x: Optional[List[float]] = None  # x方向ディリクレ境界条件 [左端, 右端]
    dirichlet_values_y: Optional[List[float]] = None  # y方向ディリクレ境界条件 [下端, 上端]
    neumann_values_x: Optional[List[float]] = None  # x方向ノイマン境界条件 [左端, 右端]
    neumann_values_y: Optional[List[float]] = None  # y方向ノイマン境界条件 [下端, 上端]

    # 係数の設定 - 拡張微分方程式の係数
    # [a, bx, by, cxx, cyy, cxy, ...]
    # f = a*psi + bx*d(psi)/dx + by*d(psi)/dy + cxx*d²(psi)/dx² + ...
    coeffs: Optional[Dict[str, float]] = field(default_factory=dict)

    # 積分定数による補正を有効にするフラグ
    enable_boundary_correction: bool = True

    # 微分次数（各方向の最大階数）
    x_deriv_order: int = 3  # x方向の最大微分階数
    y_deriv_order: int = 3  # y方向の最大微分階数

    def __post_init__(self):
        """初期化後の処理 - 係数設定と境界条件設定の正規化"""
        # デフォルト係数の設定（psi = f に相当）
        if not self.coeffs:
            self.coeffs = {"a": 1.0}

        # 単一値が指定された場合にリストに変換（x方向）
        if self.dirichlet_values_x is not None and not isinstance(
            self.dirichlet_values_x, (list, tuple)
        ):
            self.dirichlet_values_x = [
                self.dirichlet_values_x,
                self.dirichlet_values_x,
            ]

        if self.neumann_values_x is not None and not isinstance(
            self.neumann_values_x, (list, tuple)
        ):
            self.neumann_values_x = [self.neumann_values_x, self.neumann_values_x]

        # 単一値が指定された場合にリストに変換（y方向）
        if self.dirichlet_values_y is not None and not isinstance(
            self.dirichlet_values_y, (list, tuple)
        ):
            self.dirichlet_values_y = [
                self.dirichlet_values_y,
                self.dirichlet_values_y,
            ]

        if self.neumann_values_y is not None and not isinstance(
            self.neumann_values_y, (list, tuple)
        ):
            self.neumann_values_y = [self.neumann_values_y, self.neumann_values_y]

    @property
    def is_dirichlet_x(self) -> bool:
        """x方向のディリクレ境界条件が有効かどうかを返す"""
        return self.dirichlet_values_x is not None

    @property
    def is_dirichlet_y(self) -> bool:
        """y方向のディリクレ境界条件が有効かどうかを返す"""
        return self.dirichlet_values_y is not None

    @property
    def is_neumann_x(self) -> bool:
        """x方向のノイマン境界条件が有効かどうかを返す"""
        return self.neumann_values_x is not None

    @property
    def is_neumann_y(self) -> bool:
        """y方向のノイマン境界条件が有効かどうかを返す"""
        return self.neumann_values_y is not None

    def get_dirichlet_boundary_values_x(self) -> Tuple[float, float]:
        """x方向のディリクレ境界値を取得"""
        if not self.is_dirichlet_x or self.dirichlet_values_x is None:
            return 0.0, 0.0

        left_val, right_val = self.dirichlet_values_x

        if self.enable_boundary_correction:
            # 積分定数による変動を抑制するために差分を使用
            return (left_val - right_val) / 2.0, (-left_val + right_val) / 2.0
        else:
            return left_val, right_val

    def get_dirichlet_boundary_values_y(self) -> Tuple[float, float]:
        """y方向のディリクレ境界値を取得"""
        if not self.is_dirichlet_y or self.dirichlet_values_y is None:
            return 0.0, 0.0

        bottom_val, top_val = self.dirichlet_values_y

        if self.enable_boundary_correction:
            # 積分定数による変動を抑制するために差分を使用
            return (bottom_val - top_val) / 2.0, (-bottom_val + top_val) / 2.0
        else:
            return bottom_val, top_val

    def get_neumann_boundary_values_x(self) -> Tuple[float, float]:
        """x方向のノイマン境界値を取得"""
        if not self.is_neumann_x or self.neumann_values_x is None:
            return 0.0, 0.0

        return self.neumann_values_x[0], self.neumann_values_x[1]

    def get_neumann_boundary_values_y(self) -> Tuple[float, float]:
        """y方向のノイマン境界値を取得"""
        if not self.is_neumann_y or self.neumann_values_y is None:
            return 0.0, 0.0

        return self.neumann_values_y[0], self.neumann_values_y[1]

    def apply_boundary_correction(self, psi: cp.ndarray) -> cp.ndarray:
        """計算結果に対して境界条件による補正を適用"""
        if not self.enable_boundary_correction:
            return psi

        correction_x = 0.0
        correction_y = 0.0

        # x方向の補正
        if self.is_dirichlet_x and self.dirichlet_values_x is not None:
            left_val, right_val = self.dirichlet_values_x
            correction_x = (left_val + right_val) / 2.0

        # y方向の補正
        if self.is_dirichlet_y and self.dirichlet_values_y is not None:
            bottom_val, top_val = self.dirichlet_values_y
            correction_y = (bottom_val + top_val) / 2.0

        # 両方向の補正を合成（平均値を使用）
        if self.is_dirichlet_x and self.is_dirichlet_y:
            correction = (correction_x + correction_y) / 2.0
        elif self.is_dirichlet_x:
            correction = correction_x
        elif self.is_dirichlet_y:
            correction = correction_y
        else:
            correction = 0.0

        return psi + correction

    def get_grid_points(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """グリッド点の座標を生成"""
        # x方向のグリッド点
        x = cp.linspace(0, (self.nx - 1) * self.hx, self.nx)
        # y方向のグリッド点
        y = cp.linspace(0, (self.ny - 1) * self.hy, self.ny)
        
        # メッシュグリッドを返す
        return cp.meshgrid(x, y)

    def get_unknown_count(self) -> int:
        """未知数の総数を計算"""
        # 各グリッド点での未知数の数（関数値 + 各方向の導関数）
        unknowns_per_point = 1  # 関数値

        # x方向の導関数
        for i in range(1, self.x_deriv_order + 1):
            unknowns_per_point += 1

        # y方向の導関数
        for i in range(1, self.y_deriv_order + 1):
            unknowns_per_point += 1

        # 混合導関数（例：d^2f/dxdy）
        for i in range(1, self.x_deriv_order + 1):
            for j in range(1, self.y_deriv_order + 1):
                unknowns_per_point += 1

        # 総グリッド点数
        total_points = self.nx * self.ny

        return total_points * unknowns_per_point

    def flatten_indices(self, i: int, j: int) -> int:
        """2次元インデックス(i,j)を1次元インデックスに変換"""
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i * self.ny + j
        else:
            raise IndexError(f"インデックス ({i},{j}) が範囲外です。")

    def unflatten_index(self, k: int) -> Tuple[int, int]:
        """1次元インデックスkを2次元インデックス(i,j)に変換"""
        if 0 <= k < self.nx * self.ny:
            i = k // self.ny
            j = k % self.ny
            return i, j
        else:
            raise IndexError(f"インデックス {k} が範囲外です。")
