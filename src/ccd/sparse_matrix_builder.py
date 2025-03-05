"""
CuPy対応スパース行列ビルダーモジュール

CCD法の左辺行列を疎行列として効率的に生成するクラスを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Tuple

from grid_config import GridConfig


class SparseCCDLeftHandBuilder:
    """左辺ブロック行列を疎行列として生成するクラス（CuPy対応）"""

    def _build_interior_blocks(
        self, coeffs: Optional[List[float]] = None
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """内部点のブロック行列A, B, Cを生成"""
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # CuPy配列として定義
        A = cp.array(
            [
                [0, 0, 0, 0],
                [35 / 32, 19 / 32, 1 / 8, 1 / 96],
                [-4, -29 / 16, -5 / 16, -1 / 48],
                [-105 / 16, -105 / 16, -15 / 8, -3 / 16],
            ]
        )

        # 中央ブロック行列 B - 第1行を係数で置き換え
        B = cp.array([[a, b, c, d], [0, 1, 0, 0], [8, 0, 1, 0], [0, 0, 0, 1]])

        # 右ブロック行列 C
        C = cp.array(
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
        cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray
    ]:
        """境界点のブロック行列を生成（CuPy対応）"""
        # デフォルト係数
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # 基本のブロック行列
        B0 = cp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [11 / 2, 1, 0, 0],  # ノイマン境界用
                [-51 / 2, 0, 1, 0],
                [387 / 4, 0, 0, 1],  # ディリクレ境界用
            ]
        )

        C0 = cp.array(
            [
                [0, 0, 0, 0],
                [24, 24, 4, 4 / 3],
                [-264, -216, -44, -12],
                [1644, 1236, 282, 66],
            ]
        )

        D0 = cp.array(
            [
                [0, 0, 0, 0],
                [-59 / 2, 10, -1, 0],
                [579 / 2, -99, 10, 0],
                [-6963 / 4, 1203 / 2, -123 / 2, 0],
            ]
        )

        ZR = cp.array(
            [
                [0, 0, 0, 0],
                [59 / 2, 10, 1, 0],
                [579 / 2, 99, 10, 0],
                [6963 / 4, 1203 / 2, 123 / 2, 0],
            ]
        )

        AR = cp.array(
            [
                [0, 0, 0, 0],
                [-24, 24, -4, 4 / 3],
                [-264, 216, -44, 12],
                [-1644, 1236, -282, 66],
            ]
        )

        BR = cp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [-11 / 2, 1, 0, 0],  # ノイマン境界用
                [-51 / 2, 0, 1, 0],
                [-387 / 4, 0, 0, 1],  # ディリクレ境界用
            ]
        )

        # ヘルパー関数を使用してスライスを更新
        def update_block(block, row_index, new_row):
            """指定された行を更新"""
            result = block.copy()
            result[row_index] = cp.array(new_row)  # ここでPythonのリストをCuPy配列に変換
            return result

        # ディリクレ境界条件
        if dirichlet_enabled:
            # 左端の第4行
            B0 = update_block(B0, 3, [1, 0, 0, 0])
            C0 = update_block(C0, 3, [0, 0, 0, 0])
            D0 = update_block(D0, 3, [0, 0, 0, 0])

            # 右端の第4行
            BR = update_block(BR, 3, [1, 0, 0, 0])
            AR = update_block(AR, 3, [0, 0, 0, 0])
            ZR = update_block(ZR, 3, [0, 0, 0, 0])

        # ノイマン境界条件
        if neumann_enabled:
            # 左端の第2行
            B0 = update_block(B0, 1, [0, 1, 0, 0])
            C0 = update_block(C0, 1, [0, 0, 0, 0])
            D0 = update_block(D0, 1, [0, 0, 0, 0])

            # 右端の第2行
            BR = update_block(BR, 1, [0, 1, 0, 0])
            AR = update_block(AR, 1, [0, 0, 0, 0])
            ZR = update_block(ZR, 1, [0, 0, 0, 0])

        return B0, C0, D0, ZR, AR, BR

    def build_matrix_coo_data(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, Tuple[int, int]]:
        """
        COO形式の疎行列データを生成（CuPy対応）
        
        Args:
            grid_config: グリッド設定
            coeffs: 行列の係数
            dirichlet_enabled: ディリクレ境界条件の有効/無効
            neumann_enabled: ノイマン境界条件の有効/無効
        
        Returns:
            (行インデックス, 列インデックス, 値, 行列形状)
        """
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
        
        # 全体の行列サイズと深さ
        depth = 4
        matrix_size = depth * n
        
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
        
        # CuPy配列に変換
        row_indices = cp.array(row_indices, dtype=cp.int32)
        col_indices = cp.array(col_indices, dtype=cp.int32)
        values = cp.array(values, dtype=cp.float32)
        
        return row_indices, col_indices, values, (matrix_size, matrix_size)

    def build_sparse_matrix(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ):
        """
        CuPy BCOO疎行列を構築
        
        Args:
            grid_config: グリッド設定
            coeffs: 行列の係数
            dirichlet_enabled: ディリクレ境界条件の有効/無効
            neumann_enabled: ノイマン境界条件の有効/無効
        
        Returns:
            CuPyのBCOO疎行列
        """
        # COOデータを取得
        row_indices, col_indices, values, shape = self.build_matrix_coo_data(
            grid_config, coeffs, dirichlet_enabled, neumann_enabled
        )
        
        # CuPyのBCOO行列を作成
        indices = cp.column_stack([row_indices, col_indices])
        return cpx_sparse.bcoo_matrix((values, indices), shape=shape)