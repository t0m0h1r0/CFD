"""
システムビルダーモジュール

CCD法の方程式系全体の構築を担当するクラスを提供します。
GridConfigの境界条件管理機能を利用します。
"""

import jax.numpy as jnp
from typing import List, Optional, Tuple

from grid_config import GridConfig
from matrix_builder import CCDLeftHandBuilder
from vector_builder import CCDRightHandBuilder
from result_extractor import CCDResultExtractor


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
        # coeffsが指定されていない場合はgrid_configから取得
        if coeffs is None:
            coeffs = grid_config.coeffs
            
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