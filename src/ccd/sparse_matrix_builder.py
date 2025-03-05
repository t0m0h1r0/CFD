"""
スパース行列ビルダーモジュール

CCD法の左辺行列を疎行列として効率的に生成するクラスを提供します。
"""

import jax
import jax.numpy as jnp
import jax.scipy.sparse as jsp
import jax.scipy.sparse.linalg as jspl
from typing import List, Optional, Tuple

from grid_config import GridConfig


class SparseCCDLeftHandBuilder:
    """左辺ブロック行列を疎行列として生成するクラス"""

    def _build_interior_blocks(
        self, coeffs: Optional[List[float]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """内部点のブロック行列A, B, Cを生成"""
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # 左ブロック行列 A
        A = jnp.array(
            [
                [0, 0, 0, 0],
                [35 / 32, 19 / 32, 1 / 8, 1 / 96],
                [-4, -29 / 16, -5 / 16, -1 / 48],
                [-105 / 16, -105 / 16, -15 / 8, -3 / 16],
            ]
        )

        # 中央ブロック行列 B - 第1行を係数で置き換え
        B = jnp.array([[a, b, c, d], [0, 1, 0, 0], [8, 0, 1, 0], [0, 0, 0, 1]])

        # 右ブロック行列 C
        C = jnp.array(
            [
                [0, 0, 0, 0],
                [-35 / 32, 19 / 32, -1 / 8, 1 / 96],
                [-4, 29 / 16, -5 / 16, 1 / 48],
                [105 / 16, -105 / 16, 15 / 8, -3 / 16],
            ]
        )

        return A, B, C

    def _build_boundary_blocks(
        self,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = True,
        neumann_enabled: bool = False,
    ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """境界点のブロック行列を生成"""
        # デフォルト係数
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # 基本のブロック行列
        B0 = jnp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [11 / 2, 1, 0, 0],  # ノイマン境界用
                [-51 / 2, 0, 1, 0],
                [387 / 4, 0, 0, 1],  # ディリクレ境界用
            ]
        )

        C0 = jnp.array(
            [
                [0, 0, 0, 0],
                [24, 24, 4, 4 / 3],
                [-264, -216, -44, -12],
                [1644, 1236, 282, 66],
            ]
        )

        D0 = jnp.array(
            [
                [0, 0, 0, 0],
                [-59 / 2, 10, -1, 0],
                [579 / 2, -99, 10, 0],
                [-6963 / 4, 1203 / 2, -123 / 2, 0],
            ]
        )

        ZR = jnp.array(
            [
                [0, 0, 0, 0],
                [59 / 2, 10, 1, 0],
                [579 / 2, 99, 10, 0],
                [6963 / 4, 1203 / 2, 123 / 2, 0],
            ]
        )

        AR = jnp.array(
            [
                [0, 0, 0, 0],
                [-24, 24, -4, 4 / 3],
                [-264, 216, -44, 12],
                [-1644, 1236, -282, 66],
            ]
        )

        BR = jnp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [-11 / 2, 1, 0, 0],  # ノイマン境界用
                [-51 / 2, 0, 1, 0],
                [-387 / 4, 0, 0, 1],  # ディリクレ境界用
            ]
        )

        # 境界条件に応じて行を更新
        # ディリクレ境界条件
        if dirichlet_enabled:
            # 左端の第4行
            B0 = B0.at[3].set([1, 0, 0, 0])
            C0 = C0.at[3].set([0, 0, 0, 0])
            D0 = D0.at[3].set([0, 0, 0, 0])

            # 右端の第4行
            BR = BR.at[3].set([1, 0, 0, 0])
            AR = AR.at[3].set([0, 0, 0, 0])
            ZR = ZR.at[3].set([0, 0, 0, 0])

        # ノイマン境界条件
        if neumann_enabled:
            # 左端の第2行
            B0 = B0.at[1].set([0, 1, 0, 0])
            C0 = C0.at[1].set([0, 0, 0, 0])
            D0 = D0.at[1].set([0, 0, 0, 0])

            # 右端の第2行
            BR = BR.at[1].set([0, 1, 0, 0])
            AR = AR.at[1].set([0, 0, 0, 0])
            ZR = ZR.at[1].set([0, 0, 0, 0])

        return B0, C0, D0, ZR, AR, BR

    def _apply_scaling(self, blocks, h):
        """ブロック行列に次数行列のスケーリングを適用"""
        # 次数行列
        DEGREE = jnp.array(
            [
                [1, 1, 1, 1],
                [h**-1, h**0, h**1, h**2],
                [h**-2, h**-1, h**0, h**1],
                [h**-3, h**-2, h**1, h**0],
            ]
        )
        
        # 各ブロックにスケーリングを適用
        return {k: v * DEGREE for k, v in blocks.items()}
    
    def build_matrix_coo_data(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Tuple[int, int]]:
        """COO形式の疎行列データを生成"""
        n, h = grid_config.n_points, grid_config.h

        # coeffsが指定されていない場合はgrid_configから取得
        if coeffs is None:
            coeffs = grid_config.coeffs

        # 境界条件の状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet

        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann

        # ブロック行列を生成
        A, B, C = self._build_interior_blocks(coeffs)
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks(
            coeffs, dirichlet_enabled=dirichlet_enabled, neumann_enabled=neumann_enabled
        )
        
        # スケーリングを適用
        blocks = {'A': A, 'B': B, 'C': C, 'B0': B0, 'C0': C0, 'D0': D0, 
                 'ZR': ZR, 'AR': AR, 'BR': BR}
        scaled_blocks = self._apply_scaling(blocks, h)
        
        A = scaled_blocks['A']
        B = scaled_blocks['B']
        C = scaled_blocks['C']
        B0 = scaled_blocks['B0']
        C0 = scaled_blocks['C0']
        D0 = scaled_blocks['D0']
        ZR = scaled_blocks['ZR']
        AR = scaled_blocks['AR']
        BR = scaled_blocks['BR']

        # 全体の行列サイズと深さ
        depth = 4
        matrix_size = depth * n
        
        # 予想される非ゼロ要素数を計算
        # 境界ブロック + 内部ブロック
        estimated_nnz = (3 * depth * depth) * 2 + (3 * depth * depth) * (n - 2)
        
        # COO形式のデータを格納するための配列
        row_indices = []
        col_indices = []
        values = []
        
        # 左境界
        for i in range(depth):
            for j in range(depth):
                # B0ブロック
                if B0[i, j] != 0:
                    row_indices.append(i)
                    col_indices.append(j)
                    values.append(B0[i, j])
                
                # C0ブロック
                if C0[i, j] != 0:
                    row_indices.append(i)
                    col_indices.append(depth + j)
                    values.append(C0[i, j])
                
                # D0ブロック
                if D0[i, j] != 0:
                    row_indices.append(i)
                    col_indices.append(2 * depth + j)
                    values.append(D0[i, j])
        
        # 内部点
        for point_idx in range(1, n - 1):
            row_offset = depth * point_idx
            col_offset_left = depth * (point_idx - 1)
            col_offset_middle = depth * point_idx
            col_offset_right = depth * (point_idx + 1)
            
            for i in range(depth):
                for j in range(depth):
                    # Aブロック
                    if A[i, j] != 0:
                        row_indices.append(row_offset + i)
                        col_indices.append(col_offset_left + j)
                        values.append(A[i, j])
                    
                    # Bブロック
                    if B[i, j] != 0:
                        row_indices.append(row_offset + i)
                        col_indices.append(col_offset_middle + j)
                        values.append(B[i, j])
                    
                    # Cブロック
                    if C[i, j] != 0:
                        row_indices.append(row_offset + i)
                        col_indices.append(col_offset_right + j)
                        values.append(C[i, j])
        
        # 右境界
        row_offset = depth * (n - 1)
        col_offset_left2 = depth * (n - 3)
        col_offset_left = depth * (n - 2)
        col_offset_right = depth * (n - 1)
        
        for i in range(depth):
            for j in range(depth):
                # ZRブロック
                if ZR[i, j] != 0:
                    row_indices.append(row_offset + i)
                    col_indices.append(col_offset_left2 + j)
                    values.append(ZR[i, j])
                
                # ARブロック
                if AR[i, j] != 0:
                    row_indices.append(row_offset + i)
                    col_indices.append(col_offset_left + j)
                    values.append(AR[i, j])
                
                # BRブロック
                if BR[i, j] != 0:
                    row_indices.append(row_offset + i)
                    col_indices.append(col_offset_right + j)
                    values.append(BR[i, j])
        
        # JAX配列に変換
        row_indices = jnp.array(row_indices, dtype=jnp.int32)
        col_indices = jnp.array(col_indices, dtype=jnp.int32)
        values = jnp.array(values, dtype=jnp.float32)
        
        return row_indices, col_indices, values, (matrix_size, matrix_size)
    
    def build_sparse_matrix(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ):
        """JAX BCOO疎行列を構築"""
        # COOデータを取得
        row_indices, col_indices, values, shape = self.build_matrix_coo_data(
            grid_config, coeffs, dirichlet_enabled, neumann_enabled
        )
        
        # JAXのBCOO形式で疎行列を作成
        indices = jnp.column_stack([row_indices, col_indices])
        return jsp.bcoo_matrix((values, indices), shape=shape)
