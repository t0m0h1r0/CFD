import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class GridConfig:
    """グリッド設定を保持するデータクラス"""

    n_points: int  # グリッド点の数
    h: float  # グリッド幅

    # 境界条件の設定
    dirichlet_values: Optional[List[float]] = None  # ディリクレ境界条件値 [左端, 右端]
    neumann_values: Optional[List[float]] = None  # ノイマン境界条件値 [左端, 右端]

    def __post_init__(self):
        """初期化後の処理 - 単一値からリストへの変換のみ行う"""
        # 単一値が指定された場合にリストに変換
        if self.dirichlet_values is not None and not isinstance(
            self.dirichlet_values, (list, tuple)
        ):
            self.dirichlet_values = [self.dirichlet_values, self.dirichlet_values]

        if self.neumann_values is not None and not isinstance(
            self.neumann_values, (list, tuple)
        ):
            self.neumann_values = [self.neumann_values, self.neumann_values]

    @property
    def is_dirichlet(self) -> bool:
        """ディリクレ境界条件が有効かどうかを返す"""
        return self.dirichlet_values is not None

    @property
    def is_neumann(self) -> bool:
        """ノイマン境界条件が有効かどうかを返す"""
        return self.neumann_values is not None


class CCDLeftHandBuilder:
    """左辺ブロック行列を生成するクラス"""

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

    def build_matrix(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> jnp.ndarray:
        """左辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h

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

        # 次数行列
        DEGREE = jnp.array(
            [
                [1, 1, 1, 1],
                [h**-1, h**0, h**1, h**2],
                [h**-2, h**-1, h**0, h**1],
                [h**-3, h**-2, h**1, h**0],
            ]
        )

        # 次数を適用
        A *= DEGREE
        B *= DEGREE
        C *= DEGREE
        B0 *= DEGREE
        C0 *= DEGREE
        D0 *= DEGREE
        ZR *= DEGREE
        AR *= DEGREE
        BR *= DEGREE

        # 全体の行列を組み立て
        matrix_size = 4 * n
        L = jnp.zeros((matrix_size, matrix_size))

        # 左境界
        L = L.at[0:4, 0:4].set(B0)
        L = L.at[0:4, 4:8].set(C0)
        L = L.at[0:4, 8:12].set(D0)

        # 内部点
        for i in range(1, n - 1):
            row_start = 4 * i
            L = L.at[row_start : row_start + 4, row_start - 4 : row_start].set(A)
            L = L.at[row_start : row_start + 4, row_start : row_start + 4].set(B)
            L = L.at[row_start : row_start + 4, row_start + 4 : row_start + 8].set(C)

        # 右境界
        row_start = 4 * (n - 1)
        L = L.at[row_start : row_start + 4, row_start - 8 : row_start - 4].set(ZR)
        L = L.at[row_start : row_start + 4, row_start - 4 : row_start].set(AR)
        L = L.at[row_start : row_start + 4, row_start : row_start + 4].set(BR)

        return L


class CCDRightHandBuilder:
    """右辺ベクトルを生成するクラス"""

    def build_vector(
        self,
        grid_config: GridConfig,
        values: jnp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> jnp.ndarray:
        """関数値から右辺ベクトルを生成"""
        # デフォルト係数
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        n = grid_config.n_points
        depth = 4

        # 境界値
        dirichlet_values = (
            grid_config.dirichlet_values if grid_config.is_dirichlet else [0.0, 0.0]
        )
        neumann_values = (
            grid_config.neumann_values if grid_config.is_neumann else [0.0, 0.0]
        )

        # 右辺ベクトルを生成
        rhs = jnp.zeros(n * depth)

        # 関数値を設定
        indices = jnp.arange(0, n * depth, depth)
        rhs = rhs.at[indices].set(values)

        # 境界条件インデックス
        left_neu_idx = 1  # 左端ノイマン条件
        left_dir_idx = 3  # 左端ディリクレ条件
        right_neu_idx = (n - 1) * depth + 1  # 右端ノイマン条件
        right_dir_idx = n * depth - 1  # 右端ディリクレ条件

        # 境界条件を設定
        if dirichlet_enabled:
            rhs = rhs.at[left_dir_idx].set(dirichlet_values[0])
            rhs = rhs.at[right_dir_idx].set(dirichlet_values[1])

        if neumann_enabled:
            rhs = rhs.at[left_neu_idx].set(neumann_values[0])
            rhs = rhs.at[right_neu_idx].set(neumann_values[1])

        return rhs


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


class CCDSystemBuilder:
    """CCD方程式系全体の構築を担当するクラス"""

    def __init__(
        self,
        matrix_builder: CCDLeftHandBuilder,
        vector_builder: CCDRightHandBuilder,
        result_extractor: CCDResultExtractor,
    ):
        """コンポーネントの初期化"""
        self.matrix_builder = matrix_builder
        self.vector_builder = vector_builder
        self.result_extractor = result_extractor

    def build_system(
        self,
        grid_config: GridConfig,
        values: jnp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """線形方程式系 Lx = b を構築する"""
        # 境界条件の状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet

        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann

        # 左辺行列の構築
        L = self.matrix_builder.build_matrix(
            grid_config,
            coeffs,
            dirichlet_enabled=dirichlet_enabled,
            neumann_enabled=neumann_enabled,
        )

        # 右辺ベクトルの構築
        b = self.vector_builder.build_vector(
            grid_config,
            values,
            coeffs,
            dirichlet_enabled=dirichlet_enabled,
            neumann_enabled=neumann_enabled,
        )

        return L, b

    def extract_results(
        self, grid_config: GridConfig, solution: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """解ベクトルから関数値と各階導関数を抽出する"""
        return self.result_extractor.extract_components(grid_config, solution)
