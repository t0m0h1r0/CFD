"""
2次元CCD右辺ベクトルビルダーモジュール（メモリ最適化版）

2次元CCD法の右辺ベクトルを生成するクラスを提供します。
メモリ使用量を最小化する実装です。
"""

import cupy as cp
from typing import List, Optional, Dict, Tuple

from grid2d_config import Grid2DConfig


class CCD2DRightHandBuilder:
    """2次元CCD右辺ベクトルを生成するクラス（CuPy対応、メモリ最適化版）"""

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
        2次元関数値から右辺ベクトルを生成（メモリ効率版）

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
        nx, ny = grid_config.nx, grid_config.ny
        if values.shape != (nx, ny):
            print(f"警告: 関数値の形状 {values.shape} がグリッド設定 ({nx}, {ny}) と一致しません")
            # 適切なサイズに調整（必要に応じて）
            if values.size > 0:
                # 既存の値を使用して新しい配列を作成
                adjusted_values = cp.zeros((nx, ny), dtype=values.dtype)
                min_nx = min(nx, values.shape[0])
                min_ny = min(ny, values.shape[1])
                adjusted_values[:min_nx, :min_ny] = values[:min_nx, :min_ny]
                values = adjusted_values
                print(f"関数値を調整しました: {values.shape}")
            else:
                # 空の配列の場合は0で初期化
                values = cp.zeros((nx, ny), dtype=cp.float64)
                print("空の関数値配列を0で初期化しました")

        # 各点の未知数の数
        depth = 4  # f, f_x, f_y, f_xx

        # 合計未知数
        total_unknowns = nx * ny * depth
        
        # メモリ効率のため、ブロックごとに配列を構築
        print(f"右辺ベクトル構築 - サイズ: {total_unknowns}, メモリ効率モード")
        
        # 右辺ベクトルを初期化
        rhs = cp.zeros(total_unknowns, dtype=cp.float64)
        
        # 1. 関数値（f）の設定
        # 2次元のインデックス→フラット化されたインデックスに変換しながら
        for j in range(ny):
            for i in range(nx):
                # f の位置は各点の先頭（各点は depth=4 個の未知数を持つ）
                idx = (j * nx + i) * depth
                rhs[idx] = values[i, j]
        
        # 2. 境界条件の設定
        # x方向ディリクレ境界条件（左右端）
        if use_x_dirichlet and grid_config.x_dirichlet_values is not None:
            for j in range(ny):
                if j < len(grid_config.x_dirichlet_values):
                    left_val, right_val = grid_config.x_dirichlet_values[j]
                    
                    # 左端（i=0）の関数値
                    left_idx = (j * nx) * depth
                    rhs[left_idx] = left_val
                    
                    # 右端（i=nx-1）の関数値
                    right_idx = (j * nx + nx - 1) * depth
                    rhs[right_idx] = right_val
        
        # y方向ディリクレ境界条件（上下端）
        if use_y_dirichlet and grid_config.y_dirichlet_values is not None:
            for i in range(nx):
                if i < len(grid_config.y_dirichlet_values):
                    bottom_val, top_val = grid_config.y_dirichlet_values[i]
                    
                    # 下端（j=0）の関数値
                    bottom_idx = i * depth
                    rhs[bottom_idx] = bottom_val
                    
                    # 上端（j=ny-1）の関数値
                    top_idx = ((ny - 1) * nx + i) * depth
                    rhs[top_idx] = top_val
        
        # x方向ノイマン境界条件
        if use_x_neumann and grid_config.x_neumann_values is not None:
            for j in range(ny):
                if j < len(grid_config.x_neumann_values):
                    left_val, right_val = grid_config.x_neumann_values[j]
                    
                    # 左端のx方向導関数（f_xはインデックス+1の位置）
                    left_idx = (j * nx) * depth + 1
                    rhs[left_idx] = left_val
                    
                    # 右端のx方向導関数
                    right_idx = (j * nx + nx - 1) * depth + 1
                    rhs[right_idx] = right_val
        
        # y方向ノイマン境界条件
        if use_y_neumann and grid_config.y_neumann_values is not None:
            for i in range(nx):
                if i < len(grid_config.y_neumann_values):
                    bottom_val, top_val = grid_config.y_neumann_values[i]
                    
                    # 下端のy方向導関数（f_yはインデックス+2の位置）
                    bottom_idx = i * depth + 2
                    rhs[bottom_idx] = bottom_val
                    
                    # 上端のy方向導関数
                    top_idx = ((ny - 1) * nx + i) * depth + 2
                    rhs[top_idx] = top_val
        
        # メモリ使用量の報告
        memory_mb = (total_unknowns * 8) / (1024 * 1024)  # 8バイト（倍精度浮動小数点）
        print(f"右辺ベクトル構築完了 - サイズ: {rhs.shape}, 推定メモリ使用量: {memory_mb:.2f} MB")
        
        return rhs