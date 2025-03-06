"""
2次元システムビルダーモジュール

2次元CCDの方程式系全体の構築を担当するクラス
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Optional, List, Tuple, Union

from grid_config_2d import GridConfig2D
from matrix_builder_2d import CCDLeftHandBuilder2D
from vector_builder_2d import CCDRightHandBuilder2D
from result_extractor_2d import CCDResultExtractor2D


class CCDSystemBuilder2D:
    """2次元CCD方程式系全体の構築を担当するクラス"""

    def __init__(
        self,
        matrix_builder: CCDLeftHandBuilder2D,
        vector_builder: CCDRightHandBuilder2D,
        result_extractor: CCDResultExtractor2D,
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
        grid_config: GridConfig2D,
        values: cp.ndarray,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], cp.ndarray]:
        """
        2次元線形方程式系 Lx = b を構築する

        Args:
            grid_config: 2次元グリッド設定
            values: 関数値の2次元配列
            coeffs: 係数（省略時はgrid_configから取得）
            dirichlet_enabled: ディリクレ境界条件の有効/無効（省略時はgrid_configから判断）
            neumann_enabled: ノイマン境界条件の有効/無効（省略時はgrid_configから判断）

        Returns:
            (左辺行列, 右辺ベクトル)のタプル
        """
        # 境界条件と係数の状態を決定
        use_dirichlet = (
            grid_config.is_dirichlet() if dirichlet_enabled is None else dirichlet_enabled
        )
        use_neumann = (
            grid_config.is_neumann() if neumann_enabled is None else neumann_enabled
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
        self, grid_config: GridConfig2D, solution: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        解ベクトルから関数値と各偏導関数を抽出する

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル

        Returns:
            (f, f_x, f_y, f_xx, f_xy, f_yy)のタプル
        """
        return self.result_extractor.extract_components(grid_config, solution)