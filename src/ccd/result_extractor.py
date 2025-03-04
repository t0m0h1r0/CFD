"""
結果抽出モジュール

CCD法の解ベクトルから各成分を抽出するクラスを提供します。
"""

import jax.numpy as jnp
from typing import Tuple

from grid_config import GridConfig


class CCDResultExtractor:
    """CCDソルバーの結果から各成分を抽出するクラス"""

    def extract_components(
        self, grid_config: GridConfig, solution: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出
        
        Args:
            grid_config: グリッド設定
            solution: 解ベクトル
            
        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        n = grid_config.n_points
        depth = 4

        # JAXの効率的なインデックス操作を使用
        indices = jnp.arange(n * depth).reshape(n, depth)
        psi0 = solution[indices[:, 0]]
        psi1 = solution[indices[:, 1]]
        psi2 = solution[indices[:, 2]]
        psi3 = solution[indices[:, 3]]

        # ディリクレ境界条件が有効な場合、境界補正を適用
        if grid_config.is_dirichlet:
            # 境界条件による補正を適用
            psi0 = grid_config.apply_boundary_correction(psi0)
            
            # 補正後、境界値を厳密に設定
            if grid_config.enable_boundary_correction and grid_config.dirichlet_values is not None:
                psi0 = psi0.at[0].set(grid_config.dirichlet_values[0])
                psi0 = psi0.at[n - 1].set(grid_config.dirichlet_values[1])

        return psi0, psi1, psi2, psi3