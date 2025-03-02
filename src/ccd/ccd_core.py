import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Protocol, Union, Dict


@dataclass
class GridConfig:
    """グリッド設定を保持するデータクラス"""

    n_points: int  # グリッド点の数
    h: float  # グリッド幅
    
    # 境界条件の設定
    dirichlet_values: Optional[List[float]] = None  # ディリクレ境界条件値 [左端, 右端]
    neumann_values: Optional[List[float]] = None    # ノイマン境界条件値 [左端, 右端]
    
    def __post_init__(self):
        """初期化後の処理 - 値の検証と設定"""
        # いずれの境界条件も設定されていない場合、デフォルトとしてディリクレを使用
        if self.dirichlet_values is None and self.neumann_values is None:
            self.dirichlet_values = [0.0, 0.0]
            
        # dirichlet_valuesが設定されているがリストでない場合
        if self.dirichlet_values is not None and not isinstance(self.dirichlet_values, (list, tuple)):
            self.dirichlet_values = [self.dirichlet_values, self.dirichlet_values]
            
        # neumann_valuesが設定されているがリストでない場合
        if self.neumann_values is not None and not isinstance(self.neumann_values, (list, tuple)):
            self.neumann_values = [self.neumann_values, self.neumann_values]
    
    @property
    def is_dirichlet(self) -> bool:
        """ディリクレ境界条件が有効かどうかを返す"""
        return self.dirichlet_values is not None
    
    @property
    def is_neumann(self) -> bool:
        """ノイマン境界条件が有効かどうかを返す"""
        return self.neumann_values is not None


class MatrixBuilder(Protocol):
    """行列ビルダーのプロトコル定義"""

    def build_matrix(
        self, grid_config: GridConfig, coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None, neumann_enabled: bool = None
    ) -> jnp.ndarray:
        """行列を構築するメソッド"""
        ...


class VectorBuilder(Protocol):
    """ベクトルビルダーのプロトコル定義"""

    def build_vector(
        self, grid_config: GridConfig, values: jnp.ndarray, coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None, neumann_enabled: bool = None
    ) -> jnp.ndarray:
        """ベクトルを構築するメソッド"""
        ...


class ResultExtractor(Protocol):
    """結果抽出のプロトコル定義"""

    def extract_components(
        self, grid_config: GridConfig, solution: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """解ベクトルから各成分を抽出するメソッド"""
        ...


class SystemBuilder(ABC):
    """CCD方程式系の構築を抽象化する基底クラス"""

    @abstractmethod
    def build_system(
        self,
        grid_config: GridConfig,
        values: jnp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        線形方程式系 Lx = b を構築する

        Args:
            grid_config: グリッド設定
            values: 入力関数値
            coeffs: 微分係数 [a, b, c, d]
            dirichlet_enabled: ディリクレ境界条件を有効にするか
            neumann_enabled: ノイマン境界条件を有効にするか

        Returns:
            L: 左辺行列
            b: 右辺ベクトル
        """
        pass

    @abstractmethod
    def extract_results(
        self, grid_config: GridConfig, solution: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出する

        Args:
            grid_config: グリッド設定
            solution: 線形方程式系の解

        Returns:
            (psi, psi', psi'', psi''')のタプル
        """
        pass


class CCDLeftHandBuilder:
    """左辺ブロック行列を生成するクラス"""

    def _build_interior_blocks(
        self, coeffs: Optional[List[float]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """内部点のブロック行列A, B, Cを生成

        Args:
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用

        Returns:
            A: 左側のブロック行列
            B: 中央のブロック行列
            C: 右側のブロック行列
        """
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs

        # 左ブロック行列 A
        A = jnp.array(
            [
                [0, 0, 0, 0],
                [35 / 32, 19 / 32, 1 / 8, 1 / 96],
                [-4, -(29 / 16), -(5 / 16), -(1 / 48)],
                [-(105 / 16), -(105 / 16), -(15 / 8), -(3 / 16)],
            ]
        )

        # 中央ブロック行列 B - 第1行を係数で置き換え
        B = jnp.array([[a, b, c, d], [0, 1, 0, 0], [8, 0, 1, 0], [0, 0, 0, 1]])

        # 右ブロック行列 C - Aに対して反対称的な構造
        C = jnp.array(
            [
                [0, 0, 0, 0],
                [-(35 / 32), 19 / 32, -(1 / 8), 1 / 96],
                [-4, 29 / 16, -(5 / 16), 1 / 48],
                [105 / 16, -(105 / 16), 15 / 8, -(3 / 16)],
            ]
        )

        return A, B, C

    def _build_boundary_blocks(
        self, 
        grid_config: GridConfig, 
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """境界点のブロック行列を生成

        Args:
            grid_config: グリッド設定
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用
            dirichlet_enabled: ディリクレ境界条件を有効にするか
            neumann_enabled: ノイマン境界条件を有効にするか

        Returns:
            B0: 左境界の主ブロック
            C0: 左境界の第2ブロック
            D0: 左境界の第3ブロック
            ZR: 右境界の第1ブロック
            AR: 右境界の第2ブロック
            BR: 右境界の主ブロック
        """
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]

        a, b, c, d = coeffs
        
        # 境界条件の有効/無効状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet
        
        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann
        
        # 左境界 - 主ブロック B0
        B0 = jnp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [11 / 2, 1, 0, 0],  # ノイマン境界では[0, 1, 0, 0]に変更
                [-(51 / 2), 0, 1, 0],
                [387 / 4, 0, 0, 1]  # ディリクレ境界では[1, 0, 0, 0]に変更
            ]
        )
        
        # 左境界 - 第2ブロック C0
        C0 = jnp.array(
            [
                [0, 0, 0, 0],
                [24, 24, 4, 4 / 3],  # ノイマン境界では[0, 0, 0, 0]に変更
                [-264, -216, -44, -12],
                [1644, 1236, 282, 66]  # ディリクレ境界では[0, 0, 0, 0]に変更
            ]
        )
        
        # 左境界 - 第3ブロック D0
        D0 = jnp.array(
            [
                [0, 0, 0, 0],
                [-(59 / 2), 10, -1, 0],  # ノイマン境界では[0, 0, 0, 0]に変更
                [579 / 2, -99, 10, 0],
                [-(6963 / 4), 1203 / 2, -(123 / 2), 0]  # ディリクレ境界では[0, 0, 0, 0]に変更
            ]
        )
        
        # 右境界 - 第1ブロック ZR
        ZR = jnp.array(
            [
                [0, 0, 0, 0],
                [59 / 2, 10, 1, 0],  # ノイマン境界では[0, 0, 0, 0]に変更
                [579 / 2, 99, 10, 0],
                [6963 / 4, 1203 / 2, 123 / 2, 0]  # ディリクレ境界では[0, 0, 0, 0]に変更
            ]
        )
        
        # 右境界 - 第2ブロック AR
        AR = jnp.array(
            [
                [0, 0, 0, 0],
                [-24, 24, -4, 4 / 3],  # ノイマン境界では[0, 0, 0, 0]に変更
                [-264, 216, -44, 12],
                [-1644, 1236, -282, 66]  # ディリクレ境界では[0, 0, 0, 0]に変更
            ]
        )
        
        # 右境界 - 主ブロック BR
        BR = jnp.array(
            [
                [a, b, c, d],  # 第1行を係数で置き換え
                [-(11 / 2), 1, 0, 0],  # ノイマン境界では[0, 1, 0, 0]に変更
                [-(51 / 2), 0, 1, 0],
                [-(387 / 4), 0, 0, 1]  # ディリクレ境界では[1, 0, 0, 0]に変更
            ]
        )
        
        # 境界条件に応じて必要な行を更新
        # ディリクレ境界条件が有効な場合
        if dirichlet_enabled:
            # 左端の第4行
            B0 = B0.at[3].set([1, 0, 0, 0])
            C0 = C0.at[3].set([0, 0, 0, 0])
            D0 = D0.at[3].set([0, 0, 0, 0])
            
            # 右端の第4行
            BR = BR.at[3].set([1, 0, 0, 0])
            AR = AR.at[3].set([0, 0, 0, 0])
            ZR = ZR.at[3].set([0, 0, 0, 0])
        
        # ノイマン境界条件が有効な場合
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
        neumann_enabled: bool = None
    ) -> jnp.ndarray:
        """左辺のブロック行列全体を生成

        Args:
            grid_config: グリッド設定
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用
            dirichlet_enabled: ディリクレ境界条件を有効にするか (Noneの場合はgrid_configから判断)
            neumann_enabled: ノイマン境界条件を有効にするか (Noneの場合はgrid_configから判断)

        Returns:
            生成されたブロック行列（JAX配列）
        """
        n, h = grid_config.n_points, grid_config.h
        
        # 境界条件の有効/無効状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet
        
        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann

        # 係数を使用してブロック行列を生成
        A, B, C = self._build_interior_blocks(coeffs)
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks(
            grid_config,
            coeffs,
            dirichlet_enabled=dirichlet_enabled,
            neumann_enabled=neumann_enabled
        )

        # 次数行列の定義
        DEGREE = jnp.array(
            [
                [1, 1, 1, 1],
                [h**-1, h**0, h**1, h**2],
                [h**-2, h**-1, h**0, h**1],
                [h**-3, h**-2, h**1, h**0],
            ]
        )

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


class CCDRightHandBuilder:
    """右辺ベクトルを生成するクラス"""

    def build_vector(
        self, 
        grid_config: GridConfig, 
        values: jnp.ndarray, 
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None
    ) -> jnp.ndarray:
        """
        関数値から右辺ベクトルを生成

        Args:
            grid_config: グリッド設定
            values: グリッド点での関数値
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用
            dirichlet_enabled: ディリクレ境界条件を有効にするか (Noneの場合はgrid_configから判断)
            neumann_enabled: ノイマン境界条件を有効にするか (Noneの場合はgrid_configから判断)

        Returns:
            パターン[f[0],0,0,0,f[1],0,0,0,...]の右辺ベクトル
        """
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = [1.0, 0.0, 0.0, 0.0]
        
        n = grid_config.n_points
        depth = 4
        
        # 境界条件の有効/無効状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet
        
        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann
        
        # 境界値
        dirichlet_values = grid_config.dirichlet_values if grid_config.is_dirichlet else [0.0, 0.0]
        neumann_values = grid_config.neumann_values if grid_config.is_neumann else [0.0, 0.0]

        # 右辺ベクトルを効率的に生成
        rhs = jnp.zeros(n * depth)

        # 各点のf値を設定
        indices = jnp.arange(0, n * depth, depth)
        rhs = rhs.at[indices].set(values)

        # 境界条件用のインデックス
        left_neu_idx = 1    # 左端ノイマン条件（ψ'）
        left_dir_idx = 3    # 左端ディリクレ条件（ψ）
        right_neu_idx = (n-1)*depth+1  # 右端ノイマン条件
        right_dir_idx = n*depth-1      # 右端ディリクレ条件

        # 境界条件を設定 - 有効な場合のみ
        if dirichlet_enabled:
            rhs = rhs.at[left_dir_idx].set(dirichlet_values[0])   # 左端
            rhs = rhs.at[right_dir_idx].set(dirichlet_values[1])  # 右端
        
        if neumann_enabled:
            rhs = rhs.at[left_neu_idx].set(neumann_values[0])    # 左端
            rhs = rhs.at[right_neu_idx].set(neumann_values[1])   # 右端
        
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
            solution: 方程式系の解

        Returns:
            (psi, psi', psi'', psi''')のタプル
        """
        n = grid_config.n_points

        # 効率的な抽出方法
        indices0 = jnp.arange(0, n * 4, 4)
        indices1 = indices0 + 1
        indices2 = indices0 + 2
        indices3 = indices0 + 3

        psi0 = solution[indices0]
        psi1 = solution[indices1]
        psi2 = solution[indices2]
        psi3 = solution[indices3]

        return psi0, psi1, psi2, psi3


class CCDSystemBuilder(SystemBuilder):
    """CCD方程式系全体の構築を担当するクラス"""

    def __init__(
        self,
        matrix_builder: CCDLeftHandBuilder,
        vector_builder: CCDRightHandBuilder,
        result_extractor: CCDResultExtractor,
    ):
        """
        Args:
            matrix_builder: 左辺行列ビルダー
            vector_builder: 右辺ベクトルビルダー
            result_extractor: 結果抽出器
        """
        self.matrix_builder = matrix_builder
        self.vector_builder = vector_builder
        self.result_extractor = result_extractor

    def build_system(
        self,
        grid_config: GridConfig,
        values: jnp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        線形方程式系 Lx = b を構築する

        Args:
            grid_config: グリッド設定
            values: 入力関数値
            coeffs: 微分係数 [a, b, c, d]
            dirichlet_enabled: ディリクレ境界条件を有効にするか (Noneの場合はgrid_configから判断)
            neumann_enabled: ノイマン境界条件を有効にするか (Noneの場合はgrid_configから判断)

        Returns:
            L: 左辺行列
            b: 右辺ベクトル
        """
        # 境界条件のデフォルト値を設定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet
        
        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann
            
        # 左辺行列の構築 - 境界条件を明示的に指定
        L = self.matrix_builder.build_matrix(
            grid_config, 
            coeffs,
            dirichlet_enabled=dirichlet_enabled,
            neumann_enabled=neumann_enabled
        )

        # 右辺ベクトルの構築 - 境界条件を明示的に指定
        b = self.vector_builder.build_vector(
            grid_config, 
            values, 
            coeffs,
            dirichlet_enabled=dirichlet_enabled,
            neumann_enabled=neumann_enabled
        )

        return L, b

    def extract_results(
        self, grid_config: GridConfig, solution: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出する

        Args:
            grid_config: グリッド設定
            solution: 線形方程式系の解

        Returns:
            (psi, psi', psi'', psi''')のタプル
        """
        return self.result_extractor.extract_components(grid_config, solution)