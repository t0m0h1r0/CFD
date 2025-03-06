"""
2次元グリッド設定モジュール

2次元計算グリッドの設定と境界条件を集中管理するためのクラスを提供します。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import cupy as cp


@dataclass
class Grid2DConfig:
    """2次元グリッド設定を保持するデータクラス（CuPy対応）"""

    nx: int  # x方向のグリッド点の数
    ny: int  # y方向のグリッド点の数
    hx: float  # x方向のグリッド幅
    hy: float  # y方向のグリッド幅

    # 微分の最大次数
    x_deriv_order: int = 3  # x方向の微分の最大次数
    y_deriv_order: int = 3  # y方向の微分の最大次数
    mixed_deriv_order: int = 1  # 混合微分の最大次数 (例: ∂²f/∂x∂y は1)

    # 境界条件の設定
    x_dirichlet_values: Optional[List[Tuple[float, float]]] = None  # x方向ディリクレ境界条件値 [(左端y=0, 右端y=0), (左端y=1, 右端y=1), ...]
    y_dirichlet_values: Optional[List[Tuple[float, float]]] = None  # y方向ディリクレ境界条件値 [(下端x=0, 上端x=0), (下端x=1, 上端x=1), ...]
    x_neumann_values: Optional[List[Tuple[float, float]]] = None  # x方向ノイマン境界条件値 [(左端y=0, 右端y=0), ...]
    y_neumann_values: Optional[List[Tuple[float, float]]] = None  # y方向ノイマン境界条件値 [(下端x=0, 上端x=0), ...]

    # 係数の設定 (f = a*ψ + b*ψ_x + c*ψ_y + d*ψ_xx + e*ψ_yy + f*ψ_xy + ...)
    coeffs: Dict[str, float] = field(default_factory=dict)

    # 積分定数による補正を有効にするフラグ
    enable_boundary_correction: bool = True

    def __post_init__(self):
        """初期化後の処理 - 係数設定と境界条件設定の正規化"""
        # デフォルト係数の設定
        if not self.coeffs:
            self.coeffs = {
                "f": 1.0,         # 関数値
                "f_x": 0.0,       # x偏導関数
                "f_y": 0.0,       # y偏導関数
                "f_xx": 0.0,      # x二階偏導関数
                "f_yy": 0.0,      # y二階偏導関数
                "f_xy": 0.0,      # 混合偏導関数
                # 必要に応じて高次の微分も追加可能
            }

        # 境界条件の初期化
        if self.x_dirichlet_values is None and self.is_x_dirichlet:
            self.x_dirichlet_values = [(0.0, 0.0)] * self.ny

        if self.y_dirichlet_values is None and self.is_y_dirichlet:
            self.y_dirichlet_values = [(0.0, 0.0)] * self.nx

        if self.x_neumann_values is None and self.is_x_neumann:
            self.x_neumann_values = [(0.0, 0.0)] * self.ny

        if self.y_neumann_values is None and self.is_y_neumann:
            self.y_neumann_values = [(0.0, 0.0)] * self.nx

    @property
    def is_x_dirichlet(self) -> bool:
        """x方向のディリクレ境界条件が有効かどうかを返す"""
        return self.x_dirichlet_values is not None and self.coeffs.get("f", 0.0) != 0.0

    @property
    def is_y_dirichlet(self) -> bool:
        """y方向のディリクレ境界条件が有効かどうかを返す"""
        return self.y_dirichlet_values is not None and self.coeffs.get("f", 0.0) != 0.0

    @property
    def is_x_neumann(self) -> bool:
        """x方向のノイマン境界条件が有効かどうかを返す"""
        return self.x_neumann_values is not None and self.coeffs.get("f_x", 0.0) != 0.0

    @property
    def is_y_neumann(self) -> bool:
        """y方向のノイマン境界条件が有効かどうかを返す"""
        return self.y_neumann_values is not None and self.coeffs.get("f_y", 0.0) != 0.0

    def get_unknown_count_per_point(self) -> int:
        """1つの格子点あたりの未知数の数を返す"""
        # x方向の微分の数
        x_count = self.x_deriv_order + 1  # f, f_x, f_xx, ... を含む
        
        # y方向の微分の数
        y_count = self.y_deriv_order + 1  # f, f_y, f_yy, ... を含む
        
        # 混合微分の数（例: f_xy, f_xxy, f_xyy, ...）
        # 単純化のため、現在は1次の混合微分（f_xy）のみ考慮
        mixed_count = self.mixed_deriv_order
        
        # 最適化: f は両方でカウントしないように調整
        return x_count + y_count + mixed_count - 1

    def get_total_unknown_count(self) -> int:
        """システム全体の未知数の総数を返す"""
        return self.nx * self.ny * self.get_unknown_count_per_point()

    def get_coefficient_matrix_shape(self) -> Tuple[int, int]:
        """係数行列のサイズを返す"""
        total_unknowns = self.get_total_unknown_count()
        return (total_unknowns, total_unknowns)

    def get_index_mapping(self, i: int, j: int, deriv_type: str) -> int:
        """
        格子点(i,j)での特定の微分タイプに対応する未知数のインデックスを返す

        Args:
            i: x方向のインデックス（0～nx-1）
            j: y方向のインデックス（0～ny-1）
            deriv_type: 微分タイプ（"f", "f_x", "f_y", "f_xx", ...）

        Returns:
            全体の未知数ベクトル内でのインデックス
        """
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            raise ValueError(f"インデックス ({i},{j}) が範囲外です")

        # 各格子点での未知数の順序を定義
        deriv_order = {
            "f": 0,    # 関数値
            "f_x": 1,  # x偏導関数
            "f_y": 2,  # y偏導関数
            "f_xx": 3, # x二階偏導関数
            "f_yy": 4, # y二階偏導関数
            "f_xy": 5, # 混合偏導関数
            # 必要に応じて拡張
        }

        if deriv_type not in deriv_order:
            raise ValueError(f"未知の微分タイプ: {deriv_type}")

        # 点(i,j)のベースインデックス
        base_idx = (i * self.ny + j) * self.get_unknown_count_per_point()
        
        # 特定の微分タイプのオフセット
        offset = deriv_order[deriv_type]
        
        return base_idx + offset

    def apply_boundary_correction(self, solution: cp.ndarray) -> cp.ndarray:
        """
        計算結果に対して境界条件による補正を適用

        Args:
            solution: 計算された解（2D配列）

        Returns:
            補正された解（2D配列）
        """
        if not self.enable_boundary_correction:
            return solution

        # ここでは簡易的な実装
        # 実際のアプリケーションでは、より複雑な補正が必要かもしれない
        return solution
