"""
2次元右辺ベクトルビルダーモジュール

2次元CCD法の右辺ベクトルを生成するクラス
"""

import cupy as cp
from typing import List, Optional, Dict, Any, Tuple

from grid2d_config import Grid2DConfig


class CCD2DRightHandBuilder:
    """2次元右辺ベクトルを生成するクラス（CuPy対応）"""

    def build_vector(
        self,
        grid_config: Grid2DConfig,
        values: cp.ndarray,
        coeffs: Optional[Dict[str, float]] = None,
        dirichlet_enabled_x: bool = None,
        dirichlet_enabled_y: bool = None,
        neumann_enabled_x: bool = None,
        neumann_enabled_y: bool = None,
    ) -> cp.ndarray:
        """
        関数値から2次元右辺ベクトルを生成

        Args:
            grid_config: 2次元グリッド設定
            values: 関数値（2次元配列）
            coeffs: 係数辞書
            dirichlet_enabled_x: x方向ディリクレ境界条件の有効/無効
            dirichlet_enabled_y: y方向ディリクレ境界条件の有効/無効
            neumann_enabled_x: x方向ノイマン境界条件の有効/無効
            neumann_enabled_y: y方向ノイマン境界条件の有効/無効

        Returns:
            右辺ベクトル
        """
        # 境界条件と係数の状態を決定
        use_dirichlet_x = (
            grid_config.is_dirichlet_x if dirichlet_enabled_x is None else dirichlet_enabled_x
        )
        use_dirichlet_y = (
            grid_config.is_dirichlet_y if dirichlet_enabled_y is None else dirichlet_enabled_y
        )
        use_neumann_x = (
            grid_config.is_neumann_x if neumann_enabled_x is None else neumann_enabled_x
        )
        use_neumann_y = (
            grid_config.is_neumann_y if neumann_enabled_y is None else neumann_enabled_y
        )

        nx, ny = grid_config.nx, grid_config.ny
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order

        # 1グリッドあたりの未知数の計算
        # パターン1: 関数値 + 各方向の導関数（混合導関数を含まない簡略版）
        # vars_per_point = 1 + x_order + y_order
        
        # パターン2: 関数値 + x方向導関数 + y方向導関数 + 混合導関数
        # フルシステムでは関数値、x方向の導関数、y方向の導関数、混合導関数を含む
        vars_per_point = (x_order + 1) * (y_order + 1)  # フルシステム
        
        # 右辺ベクトルを生成（CuPy配列）
        total_size = nx * ny * vars_per_point
        rhs = cp.zeros(total_size)

        # 関数値を設定
        # values が2次元配列と仮定
        if values.ndim == 2 and values.shape == (nx, ny):
            for i in range(nx):
                for j in range(ny):
                    # グリッド点 (i,j) に対応する配列内の位置を計算
                    idx = self._compute_vector_index(grid_config, i, j, 0)
                    if idx < total_size:  # 配列の範囲内であることを確認
                        rhs[idx] = values[i, j]
        else:
            # valuesが平坦化されている場合、またはサイズが異なる場合の処理
            if values.size == nx * ny:
                values_reshaped = values.reshape(nx, ny)
                for i in range(nx):
                    for j in range(ny):
                        idx = self._compute_vector_index(grid_config, i, j, 0)
                        if idx < total_size:  # 配列の範囲内であることを確認
                            rhs[idx] = values_reshaped[i, j]
            else:
                # その他のケースはエラーまたは特別な処理が必要
                raise ValueError(f"関数値の形状が期待と異なります: {values.shape} vs 期待 ({nx}, {ny})")

        # 境界条件を設定
        self._apply_boundary_conditions(
            grid_config, rhs, total_size, use_dirichlet_x, use_dirichlet_y, use_neumann_x, use_neumann_y
        )

        return rhs

    def _compute_vector_index(
        self, grid_config: Grid2DConfig, i: int, j: int, var_idx: int
    ) -> int:
        """
        2次元グリッド上の特定の変数のインデックスを計算

        Args:
            grid_config: 2次元グリッド設定
            i: x方向のインデックス
            j: y方向のインデックス
            var_idx: 変数のインデックス（0: 関数値, 1: x方向1階微分, ...）

        Returns:
            ベクトル内のインデックス
        """
        nx, ny = grid_config.nx, grid_config.ny
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        
        # フルシステムでは関数値、x方向の導関数、y方向の導関数、混合導関数を含む
        vars_per_point = (x_order + 1) * (y_order + 1)
        
        # グリッド点のフラット化インデックス
        flat_idx = i * ny + j
        
        # 変数のオフセット（フルシステム用）
        return flat_idx * vars_per_point + var_idx

    def _apply_boundary_conditions(
        self,
        grid_config: Grid2DConfig,
        rhs: cp.ndarray,
        total_size: int,
        use_dirichlet_x: bool,
        use_dirichlet_y: bool,
        use_neumann_x: bool,
        use_neumann_y: bool,
    ) -> None:
        """
        右辺ベクトルに境界条件を適用

        Args:
            grid_config: 2次元グリッド設定
            rhs: 右辺ベクトル
            total_size: ベクトルの総サイズ
            use_dirichlet_x: x方向ディリクレ境界条件の有効/無効
            use_dirichlet_y: y方向ディリクレ境界条件の有効/無効
            use_neumann_x: x方向ノイマン境界条件の有効/無効
            use_neumann_y: y方向ノイマン境界条件の有効/無効
        """
        nx, ny = grid_config.nx, grid_config.ny
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        vars_per_point = (x_order + 1) * (y_order + 1)  # フルシステム

        # x方向ディリクレ境界条件
        if use_dirichlet_x:
            left_val, right_val = grid_config.get_dirichlet_boundary_values_x()
            
            # 左境界 (i=0, j=all)
            for j in range(ny):
                # 関数値のインデックス
                idx = self._compute_vector_index(grid_config, 0, j, 0)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = left_val
                
            # 右境界 (i=nx-1, j=all)
            for j in range(ny):
                idx = self._compute_vector_index(grid_config, nx-1, j, 0)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = right_val

        # y方向ディリクレ境界条件
        if use_dirichlet_y:
            bottom_val, top_val = grid_config.get_dirichlet_boundary_values_y()
            
            # 下境界 (i=all, j=0)
            for i in range(nx):
                idx = self._compute_vector_index(grid_config, i, 0, 0)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = bottom_val
                
            # 上境界 (i=all, j=ny-1)
            for i in range(nx):
                idx = self._compute_vector_index(grid_config, i, ny-1, 0)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = top_val

        # x方向ノイマン境界条件
        if use_neumann_x:
            left_deriv, right_deriv = grid_config.get_neumann_boundary_values_x()
            
            # 左境界 (i=0, j=all)
            for j in range(ny):
                # x方向1階微分のインデックス（変数配置に依存）
                x_deriv_idx = 1  # 1階微分を表す変数のインデックス
                idx = self._compute_vector_index(grid_config, 0, j, x_deriv_idx)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = left_deriv
                
            # 右境界 (i=nx-1, j=all)
            for j in range(ny):
                idx = self._compute_vector_index(grid_config, nx-1, j, x_deriv_idx)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = right_deriv

        # y方向ノイマン境界条件
        if use_neumann_y:
            bottom_deriv, top_deriv = grid_config.get_neumann_boundary_values_y()
            
            # 下境界 (i=all, j=0)
            for i in range(nx):
                # y方向1階微分のインデックス（変数配置に依存）
                y_deriv_idx = x_order + 1  # y方向1階微分を表す変数のインデックス
                idx = self._compute_vector_index(grid_config, i, 0, y_deriv_idx)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = bottom_deriv
                
            # 上境界 (i=all, j=ny-1)
            for i in range(nx):
                idx = self._compute_vector_index(grid_config, i, ny-1, y_deriv_idx)
                if idx < total_size:  # 配列の範囲内であることを確認
                    rhs[idx] = top_deriv