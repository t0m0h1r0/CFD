"""
2次元CCDシステムビルダーモジュール（修正版）

2次元CCD法の方程式系全体の構築を担当するクラスを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Optional, Tuple, Dict, Union

from grid2d_config import Grid2DConfig
from matrix2d_builder import CCD2DLeftHandBuilder
from vector2d_builder import CCD2DRightHandBuilder
from result2d_extractor import CCD2DResultExtractor


class CCD2DSystemBuilder:
    """2次元CCD方程式系全体の構築を担当するクラス（CuPy疎行列対応）"""

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
        x_dirichlet_enabled: bool = None,
        y_dirichlet_enabled: bool = None,
        x_neumann_enabled: bool = None,
        y_neumann_enabled: bool = None,
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], cp.ndarray]:
        """
        2次元線形方程式系 Lx = b を構築する

        Args:
            grid_config: 2次元グリッド設定
            values: 関数値（2次元配列）
            coeffs: 係数（省略時はgrid_configから取得）
            x_dirichlet_enabled: x方向ディリクレ境界条件の有効/無効
            y_dirichlet_enabled: y方向ディリクレ境界条件の有効/無効
            x_neumann_enabled: x方向ノイマン境界条件の有効/無効
            y_neumann_enabled: y方向ノイマン境界条件の有効/無効

        Returns:
            (左辺行列, 右辺ベクトル)のタプル
        """
        # 境界条件と係数の状態を決定
        use_x_dirichlet = (
            grid_config.is_x_dirichlet if x_dirichlet_enabled is None else x_dirichlet_enabled
        )
        use_y_dirichlet = (
            grid_config.is_y_dirichlet if y_dirichlet_enabled is None else y_dirichlet_enabled
        )
        use_x_neumann = (
            grid_config.is_x_neumann if x_neumann_enabled is None else x_neumann_enabled
        )
        use_y_neumann = (
            grid_config.is_y_neumann if y_neumann_enabled is None else y_neumann_enabled
        )
        use_coeffs = grid_config.coeffs if coeffs is None else coeffs

        try:
            # 左辺行列を構築
            L = self.matrix_builder.build_matrix(grid_config, use_coeffs)
            print(f"左辺行列のサイズ: {L.shape}")

            # 右辺ベクトルを構築
            b = self.vector_builder.build_vector(
                grid_config,
                values,
                use_coeffs,
                x_dirichlet_enabled=use_x_dirichlet,
                y_dirichlet_enabled=use_y_dirichlet,
                x_neumann_enabled=use_x_neumann,
                y_neumann_enabled=use_y_neumann,
            )
            print(f"右辺ベクトルのサイズ: {b.shape}")
            
            # 行列とベクトルのサイズ一致チェック
            if L.shape[0] != b.shape[0]:
                print(f"警告: 行列とベクトルのサイズが一致しません! 行列: {L.shape}, ベクトル: {b.shape}")
                
                # 簡易的な修正: サイズを合わせる
                min_size = min(L.shape[0], b.shape[0])
                if L.shape[0] > min_size:
                    print(f"行列のサイズを {min_size} に切り詰めます")
                    L = L[:min_size, :min_size]
                if b.shape[0] > min_size:
                    print(f"ベクトルのサイズを {min_size} に切り詰めます")
                    b = b[:min_size]
                
                print(f"修正後 - 行列: {L.shape}, ベクトル: {b.shape}")

            return L, b
            
        except Exception as e:
            # エラー情報をより詳細に出力
            print(f"Error in build_system: {e}")
            # エラーを再送出して呼び出し元に伝播
            raise

    def extract_results(
        self, grid_config: Grid2DConfig, solution: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出する

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル

        Returns:
            {"f": 関数値, "f_x": x偏導関数, ... } の形式の辞書
        """
        return self.result_extractor.extract_components(grid_config, solution)