"""
2次元CCD右辺ベクトルビルダーモジュール（修正版）

2次元CCD法の右辺ベクトルを生成するクラスを提供します。
"""

import cupy as cp
from typing import List, Optional, Dict, Tuple

from grid2d_config import Grid2DConfig


class CCD2DRightHandBuilder:
    """2次元CCD右辺ベクトルを生成するクラス（CuPy対応）"""

    def build_vector(
        self,
        grid_config: Grid2DConfig,
        values: cp.ndarray,
        coeffs: Optional[Dict[str, float]] = None,
        x_dirichlet_enabled: bool = None,
        y_dirichlet_enabled: bool = None,
        x_neumann_enabled: bool = None,
        y_neumann_enabled: bool = None,
    ) -> cp.ndarray:
        """
        2次元関数値から右辺ベクトルを生成

        Args:
            grid_config: 2次元グリッド設定
            values: 2次元格子上の関数値 (shape: (nx, ny))
            coeffs: 係数（省略時はgrid_configから取得）
            x_dirichlet_enabled: x方向ディリクレ境界条件の有効/無効
            y_dirichlet_enabled: y方向ディリクレ境界条件の有効/無効
            x_neumann_enabled: x方向ノイマン境界条件の有効/無効
            y_neumann_enabled: y方向ノイマン境界条件の有効/無効

        Returns:
            右辺ベクトル（CuPy配列）
        """
        if coeffs is None:
            coeffs = grid_config.coeffs

        # 境界条件の状態を決定
        use_x_dirichlet = (
            grid_config.is_x_dirichlet if x_dirichlet_enabled is None else x_dirichlet_enabled
        )
        use_y_dirichlet = (
            grid_config.is_y_dirichlet if y_dirichlet_enabled is None else y_dirichlet_enabled
        )
        use_x_neumann = (
            grid_config.is_x_neumann if x_neumann_enabled is None else x_neumann_enabled
        )
        use_y_neumann = (
            grid_config.is_y_neumann if y_neumann_enabled is None else y_neumann_enabled
        )

        # 値の形状を確認
        if values.shape != (grid_config.nx, grid_config.ny):
            raise ValueError(
                f"関数値の形状 {values.shape} が、グリッド設定の (nx, ny) = ({grid_config.nx}, {grid_config.ny}) と一致しません"
            )

        # 未知数の総数と各点の未知数の数
        nx, ny = grid_config.nx, grid_config.ny
        depth = 4  # 1次元CCDでの未知数の数（f, f', f'', f'''）

        # 行列サイズを計算（クロネッカー積の結果と同じにする）
        total_size = nx * ny * depth

        # 右辺ベクトルを初期化
        rhs = cp.zeros(total_size, dtype=cp.float64)

        # デバッグ情報
        print(f"右辺ベクトルサイズ: {rhs.shape}")
        print(f"関数値の形状: {values.shape}")

        # 行列の次元構造に合わせて関数値を設定
        # 注: 2次元配列のインデックス (i,j) が右辺ベクトルのどの位置に対応するかを考慮する必要がある
        
        # 簡略化のため、まず関数値を平坦化
        values_flat = values.flatten()
        
        # 各点の関数値の配置（4つの未知数ごとに最初の位置に関数値を設定）
        for idx, val in enumerate(values_flat):
            # 平坦化されたインデックスに depth をかけたものが、対応する右辺ベクトルでの位置になる
            vec_idx = idx * depth
            rhs[vec_idx] = val

        # 境界条件の設定（簡略化された実装）
        if use_x_dirichlet or use_y_dirichlet:
            # ディリクレ境界条件は、境界上の格子点での関数値（rhs[vec_idx]）としてすでに設定済み
            # 必要に応じて追加の処理をここに実装
            pass
        
        if use_x_neumann:
            # x方向のノイマン境界条件の設定例（左端と右端）
            for j in range(ny):
                if j < len(grid_config.x_neumann_values or []):
                    left_val, right_val = grid_config.x_neumann_values[j]
                    
                    # 左端 (i=0, j)
                    left_idx = (j * nx) * depth + 1  # +1 は導関数を示す
                    
                    # 右端 (i=nx-1, j)
                    right_idx = (j * nx + nx - 1) * depth + 1
                    
                    rhs[left_idx] = left_val
                    rhs[right_idx] = right_val
        
        if use_y_neumann:
            # y方向のノイマン境界条件の設定例（下端と上端）
            for i in range(nx):
                if i < len(grid_config.y_neumann_values or []):
                    bottom_val, top_val = grid_config.y_neumann_values[i]
                    
                    # 下端 (i, j=0)
                    bottom_idx = i * depth + 2  # +2 は y方向導関数を示す
                    
                    # 上端 (i, j=ny-1)
                    top_idx = ((ny - 1) * nx + i) * depth + 2
                    
                    rhs[bottom_idx] = bottom_val
                    rhs[top_idx] = top_val

        return rhs