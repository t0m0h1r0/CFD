"""
CuPy対応システムビルダーモジュール

CCD法の方程式系全体の構築を担当するクラス
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Tuple, Union

from grid_config import GridConfig
from matrix_builder import CCDLeftHandBuilder
from vector_builder import CCDRightHandBuilder
from result_extractor import CCDResultExtractor


class CCDSystemBuilder:
    """CCD方程式系全体の構築を担当するクラス（CuPy疎行列対応）"""

    def __init__(
        self,
        matrix_builder: CCDLeftHandBuilder,
        vector_builder: CCDRightHandBuilder,
        result_extractor: CCDResultExtractor,
    ):
        """
        コンポーネントの初期化

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
        values: cp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], cp.ndarray]:
        """
        線形方程式系 Lx = b を構築する

        Args:
            grid_config: グリッド設定
            values: 関数値
            coeffs: 係数（省略時はgrid_configから取得）
            dirichlet_enabled: ディリクレ境界条件の有効/無効（省略時はgrid_configから判断）
            neumann_enabled: ノイマン境界条件の有効/無効（省略時はgrid_configから判断）

        Returns:
            (左辺行列, 右辺ベクトル)のタプル
        """
        # 境界条件と係数の状態を決定
        use_dirichlet = (
            grid_config.is_dirichlet if dirichlet_enabled is None else dirichlet_enabled
        )
        use_neumann = (
            grid_config.is_neumann if neumann_enabled is None else neumann_enabled
        )
        use_coeffs = grid_config.coeffs if coeffs is None else coeffs

        try:
            # 左辺行列と右辺ベクトルを構築
            L = self.matrix_builder.build_matrix(
                grid_config,
                use_coeffs,
                dirichlet_enabled=use_dirichlet,
                neumann_enabled=use_neumann,
            )

            b = self.vector_builder.build_vector(
                grid_config,
                values,
                use_coeffs,
                dirichlet_enabled=use_dirichlet,
                neumann_enabled=use_neumann,
            )

            return L, b
            
        except Exception as e:
            # エラー情報をより詳細に出力
            print(f"Error in build_system: {e}")
            # エラーを再送出して呼び出し元に伝播
            raise

    def extract_results(
        self, grid_config: GridConfig, solution: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出する

        Args:
            grid_config: グリッド設定
            solution: 解ベクトル

        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        return self.result_extractor.extract_components(grid_config, solution)