import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Protocol


@dataclass
class GridConfig:
    """グリッド設定を保持するデータクラス"""
    n_points: int  # グリッド点の数
    h: float  # グリッド幅


class MatrixBuilder(Protocol):
    """行列ビルダーのプロトコル定義"""
    
    def build_matrix(self, grid_config: GridConfig, coeffs: Optional[List[float]] = None) -> jnp.ndarray:
        """行列を構築するメソッド"""
        ...


class VectorBuilder(Protocol):
    """ベクトルビルダーのプロトコル定義"""
    
    def build_vector(self, values: jnp.ndarray) -> jnp.ndarray:
        """ベクトルを構築するメソッド"""
        ...


class ResultExtractor(Protocol):
    """結果抽出のプロトコル定義"""
    
    def extract_components(self, solution: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """解ベクトルから各成分を抽出するメソッド"""
        ...


class SystemBuilder(ABC):
    """CCD方程式系の構築を抽象化する基底クラス"""
    
    @abstractmethod
    def build_system(
        self, grid_config: GridConfig, values: jnp.ndarray, coeffs: Optional[List[float]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        線形方程式系 Lx = b を構築する
        
        Args:
            grid_config: グリッド設定
            values: 入力関数値
            coeffs: 微分係数 [a, b, c, d]
            
        Returns:
            L: 左辺行列
            b: 右辺ベクトル
        """
        pass
    
    @abstractmethod
    def extract_results(
        self, solution: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出する
        
        Args:
            solution: 線形方程式系の解
            
        Returns:
            (psi, psi', psi'', psi''')のタプル
        """
        pass


class CCDLeftHandBuilder:
    """左辺ブロック行列を生成するクラス"""
    
    def _build_interior_blocks(self, coeffs: Optional[List[float]] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        A = jnp.array([
            #      f,      f',     f'',    f'''
            [      0,       0,       0,       0],
            [  35/32,   19/32,     1/8,    1/96],  # 左側ブロックの1行目
            [   -7/9,   29/72,    7/24,   5/108],  # 左側ブロックの2行目
            [-105/16, -105/16,   -15/8,   -3/16],  # 左側ブロックの3行目
        ])

        # 中央ブロック行列 B - 第1行を係数で置き換え
        B = jnp.array([
            #      f,      f',     f'',    f'''
            [      a,       b,       c,       d],  # 係数[a,b,c,d]を設定
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

    def _build_boundary_blocks(self, coeffs: Optional[List[float]] = None) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """境界点のブロック行列を生成
        
        Args:
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用
            
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

        # 左境界ブロック - 第1行を係数で置き換え
        B0 = jnp.array([
            #    f,     f',    f'',   f'''
            [    a,      b,      c,      d],  # 係数[a,b,c,d]を設定
            [ -9/2,      1,      0,      0],
            [-33/2,      0,      1,      0],
            [-189/4,     0,      0,      1],
        ])

        C0 = jnp.array([
            #    f,     f',    f'',   f'''
            [    0,      0,      0,      0],
            [   24,     -8,      4,      0],
            [  168,    -72,     28,      0],
            [  732,   -348,    114,      0],
        ])

        D0 = jnp.array([
            #     f,     f',    f'',   f'''
            [     0,      0,      0,      0],
            [ -39/2,     -8,     -1,      0],
            [-303/2,    -63,     -8,      0],
            [-2739/4, -579/2,  -75/2,      0],
        ])

        # 右境界ブロック - 第1行を係数で置き換え
        BR = jnp.array([
            #    f,     f',    f'',   f'''
            [    a,      b,      c,      d],  # 係数[a,b,c,d]を設定
            [  9/2,      1,      0,      0],
            [-33/2,      0,      1,      0],
            [189/4,      0,      0,      1],
        ])

        AR = jnp.array([
            #    f,     f',    f'',   f'''
            [    0,      0,      0,      0],
            [  -24,     -8,     -4,      0],
            [  168,     72,     28,      0],
            [ -732,   -348,   -114,      0],
        ])

        ZR = jnp.array([
            #     f,     f',    f'',   f'''
            [     0,      0,      0,      0],
            [  39/2,     -8,      1,      0],
            [-303/2,     63,     -8,      0],
            [2739/4, -579/2,   75/2,      0],
        ])

        return B0, C0, D0, ZR, AR, BR

    def build_matrix(self, grid_config: GridConfig, coeffs: Optional[List[float]] = None) -> jnp.ndarray:
        """左辺のブロック行列全体を生成
        
        Args:
            grid_config: グリッド設定
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用
            
        Returns:
            生成されたブロック行列（JAX配列）
        """
        n, h = grid_config.n_points, grid_config.h
        
        # 係数を使用してブロック行列を生成
        A, B, C = self._build_interior_blocks(coeffs)
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks(coeffs)

        # 次数行列の定義
        DEGREE = jnp.array([
            [     1,      1,      1,      1],
            [1/h**1,      1,   h**1,   h**2],
            [1/h**2, 1/h**1,      1,   h**1],
            [1/h**3, 1/h**2, 1/h**1,      1],
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


class CCDRightHandBuilder:
    """右辺ベクトルを生成するクラス"""
    
    def build_vector(self, grid_config: GridConfig, values: jnp.ndarray) -> jnp.ndarray:
        """
        関数値から右辺ベクトルを生成
        
        Args:
            grid_config: グリッド設定
            values: グリッド点での関数値
            
        Returns:
            パターン[f[0],0,0,0,f[1],0,0,0,...]の右辺ベクトル
        """
        n = grid_config.n_points
        depth = 4
        
        # 右辺ベクトルを効率的に生成
        rhs = jnp.zeros(n * depth)
        
        # 全てのインデックスを一度に更新
        indices = jnp.arange(0, n * depth, depth)
        rhs = rhs.at[indices].set(values)
        
        return rhs


class CCDResultExtractor:
    """CCDソルバーの結果から各成分を抽出するクラス"""
    
    def extract_components(self, grid_config: GridConfig, solution: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        
        f = solution[indices0]
        f_prime = solution[indices1]
        f_second = solution[indices2]
        f_third = solution[indices3]
        
        return f, f_prime, f_second, f_third


class CCDSystemBuilder(SystemBuilder):
    """CCD方程式系全体の構築を担当するクラス"""
    
    def __init__(
        self, 
        matrix_builder: CCDLeftHandBuilder, 
        vector_builder: CCDRightHandBuilder, 
        result_extractor: CCDResultExtractor
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
        self, grid_config: GridConfig, values: jnp.ndarray, coeffs: Optional[List[float]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        線形方程式系 Lx = b を構築する
        
        Args:
            grid_config: グリッド設定
            values: 入力関数値
            coeffs: 微分係数 [a, b, c, d]
            
        Returns:
            L: 左辺行列
            b: 右辺ベクトル
        """
        # 左辺行列の構築
        L = self.matrix_builder.build_matrix(grid_config, coeffs)
        
        # 右辺ベクトルの構築
        b = self.vector_builder.build_vector(grid_config, values)
        
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