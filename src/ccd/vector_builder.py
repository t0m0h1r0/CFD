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
        """
        関数値から右辺ベクトルを生成
        
        Args:
            grid_config: グリッド設定
            values: 関数値
            coeffs: 係数（省略時はgrid_configから取得）
            dirichlet_enabled: ディリクレ境界条件の有効/無効（省略時はgrid_configから判断）
            neumann_enabled: ノイマン境界条件の有効/無効（省略時はgrid_configから判断）
            
        Returns:
            右辺ベクトル
        """
        # 境界条件と係数の状態を決定
        use_dirichlet = grid_config.is_dirichlet if dirichlet_enabled is None else dirichlet_enabled
        use_neumann = grid_config.is_neumann if neumann_enabled is None else neumann_enabled
        
        n = grid_config.n_points
        depth = 4

        # 右辺ベクトルを生成
        rhs = jnp.zeros(n * depth)

        # 関数値を設定
        indices = jnp.arange(0, n * depth, depth)
        rhs = rhs.at[indices].set(values)

        # 境界条件インデックス
        left_neu_idx = 1
        left_dir_idx = 3
        right_neu_idx = (n - 1) * depth + 1
        right_dir_idx = n * depth - 1

        # 境界条件を設定
        if use_dirichlet:
            left_value, right_value = grid_config.get_dirichlet_boundary_values()
            rhs = rhs.at[left_dir_idx].set(left_value)
            rhs = rhs.at[right_dir_idx].set(right_value)

        if use_neumann:
            left_value, right_value = grid_config.get_neumann_boundary_values()
            rhs = rhs.at[left_neu_idx].set(left_value)
            rhs = rhs.at[right_neu_idx].set(right_value)

        return rhs