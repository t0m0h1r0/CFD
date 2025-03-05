"""
2次元グリッド設定モジュール

2次元計算グリッドの設定と境界条件を管理するためのクラスを提供します。
"""

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class Grid2DConfig:
    """2次元問題用のグリッド設定を管理するデータクラス"""
    
    nx: int  # x方向のグリッド点数
    ny: int  # y方向のグリッド点数
    hx: float  # x方向のグリッド幅
    hy: float  # y方向のグリッド幅
    
    # 境界条件の設定 - ディリクレ境界条件
    # 各キーは 'left', 'right', 'bottom', 'top' で、値は各境界のグリッド点に対応する値の配列
    dirichlet_values: Dict[str, jnp.ndarray] = field(default_factory=dict)
    
    # ノイマン境界条件
    # 各キーは 'left', 'right', 'bottom', 'top' で、値は各境界のグリッド点に対応する法線微分値の配列
    neumann_values: Dict[str, jnp.ndarray] = field(default_factory=dict)
    
    # 係数設定 [a, b, c, d, e, f] は f = a*ψ + b*ψx + c*ψxx + d*ψy + e*ψyy + f*ψxy などの設定に使用
    coeffs: Optional[List[float]] = None
    
    # その他の設定
    enable_boundary_correction: bool = True
    
    def __post_init__(self):
        """初期化後の処理 - デフォルト値と境界配列の検証"""
        # デフォルト係数の設定
        if self.coeffs is None:
            # 2次元問題用のデフォルト係数（例：f = ψ）
            self.coeffs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 境界値配列の検証と変換
        self._validate_and_convert_boundary_arrays()
    
    def _validate_and_convert_boundary_arrays(self):
        """境界値配列を検証し、必要に応じて変換する"""
        # 各境界の期待される長さ
        boundary_lengths = {
            'left': self.ny,
            'right': self.ny,
            'bottom': self.nx,
            'top': self.nx
        }
        
        # ディリクレ境界値の検証
        for boundary, values in list(self.dirichlet_values.items()):
            if boundary not in boundary_lengths:
                raise ValueError(f"無効な境界名: {boundary}")
            
            # スカラー値の場合は配列に変換
            if np.isscalar(values):
                self.dirichlet_values[boundary] = jnp.full(
                    boundary_lengths[boundary], values)
            # 長さの検証
            elif len(values) != boundary_lengths[boundary]:
                raise ValueError(
                    f"{boundary}境界の値の長さが不正: 期待値{boundary_lengths[boundary]}、実際{len(values)}")
                
        # ノイマン境界値も同様に検証
        for boundary, values in list(self.neumann_values.items()):
            if boundary not in boundary_lengths:
                raise ValueError(f"無効な境界名: {boundary}")
            
            # スカラー値の場合は配列に変換
            if np.isscalar(values):
                self.neumann_values[boundary] = jnp.full(
                    boundary_lengths[boundary], values)
            # 長さの検証
            elif len(values) != boundary_lengths[boundary]:
                raise ValueError(
                    f"{boundary}境界の値の長さが不正: 期待値{boundary_lengths[boundary]}、実際{len(values)}")
    
    def get_boundary_indices(self) -> Dict[str, jnp.ndarray]:
        """各境界点のグローバルインデックスを取得"""
        # 左境界
        left_indices = jnp.array([i * self.nx for i in range(self.ny)])
        
        # 右境界
        right_indices = jnp.array([(i + 1) * self.nx - 1 for i in range(self.ny)])
        
        # 下境界
        bottom_indices = jnp.arange(self.nx)
        
        # 上境界
        top_indices = jnp.arange((self.ny - 1) * self.nx, self.ny * self.nx)
        
        return {
            'left': left_indices,
            'right': right_indices,
            'bottom': bottom_indices,
            'top': top_indices
        }
    
    def get_interior_indices(self) -> jnp.ndarray:
        """内部点のグローバルインデックスを取得"""
        all_indices = jnp.arange(self.nx * self.ny)
        
        # 境界インデックスを取得
        boundary_indices = self.get_boundary_indices()
        
        # 全ての境界インデックスを結合
        all_boundary = jnp.concatenate(list(boundary_indices.values()))
        
        # 境界以外のインデックスを取得
        mask = jnp.ones(self.nx * self.ny, dtype=bool)
        mask = mask.at[all_boundary].set(False)
        
        return all_indices[mask]
    
    def create_1d_configs(self) -> Tuple["GridConfig", "GridConfig"]:
        """x方向とy方向の1次元GridConfigを生成
        
        Note:
            この関数を使用するには、既存の1次元CCD実装の
            GridConfigクラスをインポートする必要があります。
        
        Returns:
            x方向とy方向の1次元GridConfigのタプル
        """
        from ccd.grid_config import GridConfig
        
        # x方向の1次元設定
        x_dirichlet = None
        if 'left' in self.dirichlet_values and 'right' in self.dirichlet_values:
            x_dirichlet = [
                self.dirichlet_values['left'][0],
                self.dirichlet_values['right'][0]
            ]
            
        # x方向のノイマン条件
        x_neumann = None
        if 'left' in self.neumann_values and 'right' in self.neumann_values:
            x_neumann = [
                self.neumann_values['left'][0],
                self.neumann_values['right'][0]
            ]
        
        # x方向の係数（最初の3つを使用）
        x_coeffs = self.coeffs[:3] if len(self.coeffs) >= 3 else self.coeffs
        
        x_config = GridConfig(
            n_points=self.nx,
            h=self.hx,
            dirichlet_values=x_dirichlet,
            neumann_values=x_neumann,
            coeffs=x_coeffs,
            enable_boundary_correction=self.enable_boundary_correction
        )
        
        # y方向の1次元設定
        y_dirichlet = None
        if 'bottom' in self.dirichlet_values and 'top' in self.dirichlet_values:
            y_dirichlet = [
                self.dirichlet_values['bottom'][0],
                self.dirichlet_values['top'][0]
            ]
            
        # y方向のノイマン条件
        y_neumann = None
        if 'bottom' in self.neumann_values and 'top' in self.neumann_values:
            y_neumann = [
                self.neumann_values['bottom'][0],
                self.neumann_values['top'][0]
            ]
        
        # y方向の係数（4-6番目を使用）
        y_coeffs = None
        if len(self.coeffs) >= 6:
            y_coeffs = [self.coeffs[0], self.coeffs[3], self.coeffs[4], self.coeffs[5] if len(self.coeffs) > 5 else 0.0]
        else:
            y_coeffs = self.coeffs  # 不十分な場合はそのまま使用
            
        y_config = GridConfig(
            n_points=self.ny,
            h=self.hy,
            dirichlet_values=y_dirichlet,
            neumann_values=y_neumann,
            coeffs=y_coeffs,
            enable_boundary_correction=self.enable_boundary_correction
        )
        
        return x_config, y_config
    
    def is_dirichlet_boundary(self, boundary: str) -> bool:
        """指定した境界にディリクレ条件が設定されているか確認"""
        return boundary in self.dirichlet_values and self.dirichlet_values[boundary] is not None
    
    def is_neumann_boundary(self, boundary: str) -> bool:
        """指定した境界にノイマン条件が設定されているか確認"""
        return boundary in self.neumann_values and self.neumann_values[boundary] is not None
