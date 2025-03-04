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

        # 各成分のインデックス
        indices0 = jnp.arange(0, n * 4, 4)
        indices1 = indices0 + 1
        indices2 = indices0 + 2
        indices3 = indices0 + 3

        # 成分の抽出
        psi0 = solution[indices0]
        psi1 = solution[indices1]
        psi2 = solution[indices2]
        psi3 = solution[indices3]

        # ディリクレ境界条件が有効な場合、境界値を明示的に設定
        if grid_config.is_dirichlet and grid_config.dirichlet_values is not None:
            psi0 = psi0.at[0].set(grid_config.dirichlet_values[0])
            psi0 = psi0.at[n - 1].set(grid_config.dirichlet_values[1])

        return psi0, psi1, psi2, psi3
