"""
ベクトルビルダーモジュール

CCD法の右辺ベクトルを生成するクラスを提供します。
"""

import jax.numpy as jnp
from typing import List, Optional

from grid_config import GridConfig


class CCDRightHandBuilder:
    """右辺ベクトルを生成するクラス"""

    def build_vector(
        self,
        grid_config: GridConfig,
        values: jnp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> jnp.ndarray:
        """関数値から右辺ベクトルを生成"""
        # coeffsが指定されていない場合はgrid_configから取得
        if coeffs is None:
            coeffs = grid_config.coeffs

        n = grid_config.n_points
        depth = 4

        # 境界条件の状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet

        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann

        # 境界値
        dirichlet_values = (
            grid_config.dirichlet_values if grid_config.is_dirichlet else [0.0, 0.0]
        )
        neumann_values = (
            grid_config.neumann_values if grid_config.is_neumann else [0.0, 0.0]
        )

        # 右辺ベクトルを生成
        rhs = jnp.zeros(n * depth)

        # 関数値を設定
        indices = jnp.arange(0, n * depth, depth)
        rhs = rhs.at[indices].set(values)

        # 境界条件インデックス
        left_neu_idx = 1  # 左端ノイマン条件
        left_dir_idx = 3  # 左端ディリクレ条件
        right_neu_idx = (n - 1) * depth + 1  # 右端ノイマン条件
        right_dir_idx = n * depth - 1  # 右端ディリクレ条件

        # 境界条件を設定
        if dirichlet_enabled:
            rhs = rhs.at[left_dir_idx].set(dirichlet_values[0])
            rhs = rhs.at[right_dir_idx].set(dirichlet_values[1])

        if neumann_enabled:
            rhs = rhs.at[left_neu_idx].set(neumann_values[0])
            rhs = rhs.at[right_neu_idx].set(neumann_values[1])

        return rhs
