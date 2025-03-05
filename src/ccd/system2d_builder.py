"""
2次元システムビルダーモジュール

2次元CCD法の方程式系全体の構築を担当するクラス
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Dict, Tuple, Optional

from grid2d_config import Grid2DConfig
from matrix2d_builder import CCD2DLeftHandBuilder
from vector2d_builder import CCD2DRightHandBuilder
from result2d_extractor import CCD2DResultExtractor


class CCD2DSystemBuilder:
    """2次元CCD方程式系全体の構築を担当するクラス（CuPy対応）"""

    def __init__(
        self,
        matrix_builder: CCD2DLeftHandBuilder,
        vector_builder: CCD2DRightHandBuilder,
        result_extractor: CCD2DResultExtractor,
    ):
        """
        コンポーネントの初期化

        Args:
            matrix_builder: 2次元左辺行列ビルダー
            vector_builder: 2次元右辺ベクトルビルダー
            result_extractor: 2次元結果抽出器
        """
        self.matrix_builder = matrix_builder
        self.vector_builder = vector_builder
        self.result_extractor = result_extractor

    def build_system(
        self,
        grid_config: Grid2DConfig,
        values: cp.ndarray,
        coeffs: Optional[Dict[str, float]] = None,
        dirichlet_enabled_x: bool = None,
        dirichlet_enabled_y: bool = None,
        neumann_enabled_x: bool = None,
        neumann_enabled_y: bool = None,
    ) -> Tuple[cpx_sparse.spmatrix, cp.ndarray]:
        """
        2次元線形方程式系 Lx = b を構築

        Args:
            grid_config: 2次元グリッド設定
            values: 関数値 (2次元配列またはフラット化された配列)
            coeffs: 係数辞書
            dirichlet_enabled_x: x方向ディリクレ境界条件の有効/無効
            dirichlet_enabled_y: y方向ディリクレ境界条件の有効/無効
            neumann_enabled_x: x方向ノイマン境界条件の有効/無効
            neumann_enabled_y: y方向ノイマン境界条件の有効/無効

        Returns:
            (左辺行列, 右辺ベクトル)のタプル
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
        use_coeffs = grid_config.coeffs if coeffs is None else coeffs

        # 左辺行列と右辺ベクトルを構築
        L = self.matrix_builder.build_matrix(
            grid_config,
            use_coeffs,
            dirichlet_enabled_x=use_dirichlet_x,
            dirichlet_enabled_y=use_dirichlet_y,
            neumann_enabled_x=use_neumann_x,
            neumann_enabled_y=use_neumann_y,
        )

        b = self.vector_builder.build_vector(
            grid_config,
            values,
            use_coeffs,
            dirichlet_enabled_x=use_dirichlet_x,
            dirichlet_enabled_y=use_dirichlet_y,
            neumann_enabled_x=use_neumann_x,
            neumann_enabled_y=use_neumann_y,
        )

        return L, b

    def extract_results(
        self, grid_config: Grid2DConfig, solution: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル

        Returns:
            各成分を含む辞書
            {
                "f": 関数値(nx, ny),
                "f_x": x方向1階微分(nx, ny),
                "f_y": y方向1階微分(nx, ny),
                "f_xx": x方向2階微分(nx, ny),
                ...
            }
        """
        components = self.result_extractor.extract_components(grid_config, solution)
        
        # 必要に応じて混合微分も抽出
        mixed_derivatives = self.result_extractor.extract_mixed_derivatives(grid_config, solution)
        
        # 辞書を結合
        for key, value in mixed_derivatives.items():
            components[key] = value
            
        return components
