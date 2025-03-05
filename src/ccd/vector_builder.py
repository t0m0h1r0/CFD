"""
CuPy対応ベクトルビルダーモジュール

CCD法の右辺ベクトルを生成するクラス
"""

import cupy as cp
from typing import List, Optional

from grid_config import GridConfig


class CCDRightHandBuilder:
    """右辺ベクトルを生成するクラス（CuPy対応）"""

    def build_vector(
        self,
        grid_config: GridConfig,
        values: cp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> cp.ndarray:
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

        # 右辺ベクトルを生成（CuPy配列）
        rhs = cp.zeros(n * depth)

        # 関数値を設定
        indices = cp.arange(0, n * depth, depth)
        for i, val in zip(indices, values):
            rhs[int(i)] = val

        # 境界条件インデックス
        left_neu_idx = 1
        left_dir_idx = 3
        right_neu_idx = (n - 1) * depth + 1
        right_dir_idx = n * depth - 1

        # 境界条件を設定
        if use_dirichlet:
            left_value, right_value = grid_config.get_dirichlet_boundary_values()
            rhs[left_dir_idx] = left_value
            rhs[right_dir_idx] = right_value

        if use_neumann:
            left_value, right_value = grid_config.get_neumann_boundary_values()
            rhs[left_neu_idx] = left_value
            rhs[right_neu_idx] = right_value

        return rhs