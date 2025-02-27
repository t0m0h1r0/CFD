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
            #      f,      f',     f'',    f'''
            [      0,       0,       0,       0],
            [  35/32,   19/32,     1/8,    1/96],  # 左側ブロックの1行目
            [   -7/9,   29/72,    7/24,  5/108],  # 左側ブロックの2行目
            [-105/16, -105/16,   -15/8,   -3/16],  # 左側ブロックの3行目
        ])

        # 中央ブロック行列 B - 単位行列
        B = jnp.array([
            #      f,      f',     f'',    f'''
            [      1,       0,       0,       0],
            [      0,       1,       0,       0],  # 中央ブロックの1行目
            [   14/9,       0,       1,       0],  # 中央ブロックの2行目
            [      0,       0,       0,       1],  # 中央ブロックの3行目
        ])

        # 右ブロック行列 C - Aに対して反対称的な構造
        C = jnp.array([
            #      f,      f',     f'',    f'''
            [      0,       0,       0,       0],
            [ -35/32,   19/32,    -1/8,    1/96],  # 右側ブロックの1行目
            [   -7/9,  -29/72,    7/24,  -5/108],  # 右側ブロックの2行目
            [ 105/16, -105/16,    15/8,   -3/16],  # 右側ブロックの3行目
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
            #    f,     f',    f'',   f'''
            [    1,      0,      0,      0],
            [  9/2,      1,      0,      0],
            [-33/2,      0,      1,      0],
            [189/4,      0,      0,      1],
        ])

        C0 = jnp.array([
            #    f,     f',    f'',   f'''
            [    0,      0,      0,      0],
            [  -24,     -8,     -4,      0],
            [  168,     72,     28,      0],
            [ -732,   -348,   -114,      0],
        ])

        D0 = jnp.array([
            #     f,     f',    f'',   f'''
            [     0,      0,      0,      0],
            [  39/2,     -8,      1,      0],
            [-303/2,     63,     -8,      0],
            [2739/4, -579/2,   75/2,      0],
        ])

        # 右境界の行列 - 左境界と完全に対称的な構造
        BR = jnp.array([
            #    f,     f',    f'',   f'''
            [    1,      0,      0,      0],
            [ -9/2,      1,      0,      0],
            [-33/2,      0,      1,      0],
            [-189/4,     0,      0,      1],
        ])

        AR = jnp.array([
            #    f,     f',    f'',   f'''
            [    0,      0,      0,      0],
            [   24,     -8,      4,      0],
            [  168,    -72,     28,      0],
            [  732,   -348,    114,      0],
        ])

        ZR = jnp.array([
            #     f,     f',    f'',   f'''
            [     0,      0,      0,      0],
            [ -39/2,     -8,     -1,      0],
            [-303/2,    -63,     -8,      0],
            [-2739/4, -579/2,  -75/2,      0],
        ])

        return B0, C0, D0, ZR, AR, BR

    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """左辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        A, B, C = self._build_interior_blocks()
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks()

        # 次数行列の定義
        DEGREE = jnp.array([
            [     1,      h,   h**2,   h**3],
#            [     1,      h,   h**2,   h**3],
#            [     1,      h,   h**2,   h**3],
#            [     1,      h,   h**2,   h**3],
        ])

        # 次数を適用
        A = A * DEGREE
        B = B * DEGREE
        C = C * DEGREE
        B0 = B0 * DEGREE
        C0 = C0 * DEGREE
        D0 = D0 * DEGREE
        ZR = ZR * DEGREE
        AR = AR * DEGREE
        BR = BR * DEGREE

        matrix_size = 4 * n
        L = jnp.zeros((matrix_size, matrix_size))

        # 左境界条件を設定
        L = L.at[0:4, 0:4].set(B0)
        L = L.at[0:4, 4:8].set(C0)
        L = L.at[0:4, 8:12].set(D0)

        # 内部点を設定
        for i in range(1, n - 1):
            row_start = 4 * i
            L = L.at[row_start : row_start + 4, row_start - 4 : row_start].set(A)
            L = L.at[row_start : row_start + 4, row_start : row_start + 4].set(B)
            L = L.at[row_start : row_start + 4, row_start + 4 : row_start + 8].set(C)

        # 右境界条件を設定
        row_start = 4 * (n - 1)
        L = L.at[row_start : row_start + 4, row_start - 8 : row_start - 4].set(ZR)
        L = L.at[row_start : row_start + 4, row_start - 4 : row_start].set(AR)
        L = L.at[row_start : row_start + 4, row_start : row_start + 4].set(BR)

        return L