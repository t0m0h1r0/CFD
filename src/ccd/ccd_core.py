import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridConfig:
    """グリッド設定を保持するデータクラス"""

    n_points: int  # グリッド点の数
    h: float  # グリッド幅


class BlockMatrixBuilder(ABC):
    """ブロック行列生成の抽象基底クラス"""

    @abstractmethod
    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """ブロック行列を生成する抽象メソッド"""
        pass


class LeftHandBlockBuilder(BlockMatrixBuilder):
    """左辺のブロック行列を生成するクラス"""

    def _build_interior_blocks(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """内部点のブロック行列A, B, Cを生成
        
        Returns:
            A: 左側のブロック行列
            B: 中央のブロック行列
            C: 右側のブロック行列
        """
        # 左ブロック行列 A
        A = jnp.array([
            # f',    f'',   f'''
            [ 19/32,  1/8,   1/96],  # 左側ブロックの1行目
            [-29/16, -5/16, -1/48],  # 左側ブロックの2行目
            [-105/16,-15/8, -3/16]   # 左側ブロックの3行目
        ])

        # 中央ブロック行列 B - 単位行列
        B = jnp.eye(3)

        # 右ブロック行列 C - Aに対して反対称的な構造
        C = jnp.array([
            # f',    f'',    f'''
            [ 19/32, -1/8,   1/96],  # 右側ブロックの1行目
            [ 29/16, -5/16,  1/48],  # 右側ブロックの2行目
            [-105/16, 15/8, -3/16]   # 右側ブロックの3行目
        ])

        return A, B, C

    def _build_boundary_blocks(self) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """境界点のブロック行列を生成
        
        Returns:
            B0: 左境界の主ブロック
            C0: 左境界の第2ブロック
            D0: 左境界の第3ブロック
            ZR: 右境界の第1ブロック
            AR: 右境界の第2ブロック
            BR: 右境界の主ブロック
        """
        # 左境界の行列群
        B0 = jnp.array([
            # f',  f'',  f'''
            [  1,   0,  0],  # Pattern 1
            [  0,   1,  0],  # Pattern 2
            [ 14,   2,  0]   # Pattern 3
        ])

        C0 = jnp.array([
            # f',  f'',  f'''
            [  2,  -1,  0],  # Pattern 1
            [ -6,   5,  0],  # Pattern 2
            [ 16,  -4,  0]   # Pattern 3
        ])

        D0 = jnp.array([
            # f',  f'',  f'''
            [  0,   0,   0],  # Pattern 1
            [  0,   0,   0],  # Pattern 2
            [  0,   0,   0]   # Pattern 3
        ])

        # 右境界の行列 - 左境界と完全に対称的に
        BR = jnp.array([
            # f',  f'',  f'''
            [  1,   0,  0],  # Pattern 1
            [  0,   1,  0],  # Pattern 2
            [ 14,  -2,  0]   # Pattern 3
        ])

        AR = jnp.array([
            # f',  f'',  f'''
            [  2,   1,  0],  # Pattern 1
            [  6,   5,  0],  # Pattern 2
            [ 16,   4,  0]   # Pattern 3
        ])

        ZR = jnp.array([
            # f',  f'',  f'''
            [  0,   0,   0],  # Pattern 1
            [  0,   0,   0],  # Pattern 2
            [  0,   0,   0]   # Pattern 3
        ])

        return B0, C0, D0, ZR, AR, BR

    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """左辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        A, B, C = self._build_interior_blocks()
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks()

        # 次数行列の定義
        DEGREE_I = jnp.array([
            [1, h, h**2],
            [1/h, 1, h],
            [1/h**2, h, 1],
            ])
        DEGREE_B = jnp.array([
            [1, h, h**2],
            [1, h, h**2],
            [1, h, h**2],
            ])

        # 次数を適用
        A = A * DEGREE_I
        B = B * DEGREE_I
        C = C * DEGREE_I
        B0 = B0 * DEGREE_B
        C0 = C0 * DEGREE_B
        D0 = D0 * DEGREE_B
        ZR = ZR * DEGREE_B
        AR = AR * DEGREE_B
        BR = BR * DEGREE_B

        matrix_size = 3 * n
        L = jnp.zeros((matrix_size, matrix_size))

        # 左境界条件を設定
        L = L.at[0:3, 0:3].set(B0)
        L = L.at[0:3, 3:6].set(C0)
        L = L.at[0:3, 6:9].set(D0)

        # 内部点を設定
        for i in range(1, n - 1):
            row_start = 3 * i
            L = L.at[row_start : row_start + 3, row_start - 3 : row_start].set(A)
            L = L.at[row_start : row_start + 3, row_start : row_start + 3].set(B)
            L = L.at[row_start : row_start + 3, row_start + 3 : row_start + 6].set(C)

        # 右境界条件を設定
        row_start = 3 * (n - 1)
        L = L.at[row_start : row_start + 3, row_start - 6 : row_start - 3].set(ZR)
        L = L.at[row_start : row_start + 3, row_start - 3 : row_start].set(AR)
        L = L.at[row_start : row_start + 3, row_start : row_start + 3].set(BR)

        return L


class RightHandBlockBuilder(BlockMatrixBuilder):
    """右辺のブロック行列を生成するクラス"""

    def _build_interior_block(self) -> jnp.ndarray:
        """右辺のブロック行列Kを生成"""
        K = jnp.array([
            # 左点,  中点,  右点
            [-35/32,    0,  35/32],  # 1階導関数の係数
            [     4,   -8,      4],  # 2階導関数の係数
            [105/16,    0, -105/16]  # 3階導関数の係数
        ])

        return K

    def _build_boundary_blocks(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """境界点のブロック行列を生成
        
        Returns:
            K0: 左境界用の行列
            KR: 右境界用の行列（K0と対称的な構造）
        """
        # 左境界用の行列
        K0 = jnp.array([
            # 左点,  中点,  右点
            [ -7/2,   4,  -1/2],  # 1階導関数の係数
            [    9, -12,     3],  # 2階導関数の係数
            [  -31,   2,    -1]   # 3階導関数の係数
        ])

        KR = jnp.array([
            # 左点,  中点,  右点
            [  1/2,  -4,   7/2],  # 1階導関数の係数
            [    3, -12,     9],  # 2階導関数の係数
            [    1,  -2,    31]   # 3階導関数の係数
        ])

        return K0, KR
    
    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """右辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        K_interior = self._build_interior_block()
        K0, KR = self._build_boundary_blocks()
        DEGREE_I = jnp.array([
            [1/h,  ],
            [1/h**2],
            [1/h**3],
            ])
        DEGREE_B = jnp.array([
            [1/h,  ],
            [1/h   ],
            [1/h   ],
            ])
        
        K_interior = K_interior * DEGREE_I
        K0 = K0 * DEGREE_B
        KR = KR * DEGREE_B

        matrix_size = 3 * n
        vector_size = n
        K = jnp.zeros((matrix_size, vector_size))

        # 左境界条件を設定
        K = K.at[0:3, 0:3].set(K0)

        # 内部点を設定
        for i in range(1, n - 1):
            row_start = 3 * i
            K = K.at[row_start : row_start + 3, i - 1 : i + 2].set(K_interior)

        # 右境界条件を設定
        row_start = 3 * (n - 1)
        K = K.at[row_start : row_start + 3, n - 3 : n].set(KR)

        return K