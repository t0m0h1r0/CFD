"""
変換パイプライン

CuPy対応のスケーリングと正則化を順に適用するパイプラインを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Optional, Tuple, Callable, Dict, Any, List

from strategy_interface import MatrixTransformer


class TransformationPipeline:
    """
    変換パイプライン

    複数の変換を順次適用して、結果を結合します（CuPy対応）
    """

    def __init__(self, transformers: Optional[List[MatrixTransformer]] = None):
        """
        初期化

        Args:
            transformers: 適用する変換のリスト
        """
        self.transformers = transformers or []
        self.original_matrix = None
        self.transformed_matrix = None
        self.transformation_history = []

    def transform_matrix(
        self, matrix
    ) -> Tuple[cp.ndarray, Callable[[cp.ndarray], cp.ndarray]]:
        """
        全変換を順に適用

        Args:
            matrix: 変換する行列

        Returns:
            (変換された行列, 逆変換関数)
        """
        # スパース行列の場合、密行列に変換
        if cpx_sparse.issparse(matrix):
            self.original_matrix = matrix.toarray()
        else:
            self.original_matrix = cp.asarray(matrix)

        transformed = self.original_matrix
        inverse_funcs = []
        self.transformation_history = []

        # 各変換を順に適用
        for transformer in self.transformers:
            transformed, inverse_func = transformer.transform_matrix(transformed)
            inverse_funcs.append(inverse_func)
            self.transformation_history.append(transformed)

        self.transformed_matrix = transformed

        # 逆変換関数の合成
        def composite_inverse(x):
            result = x
            for inverse_func in reversed(inverse_funcs):
                result = inverse_func(result)
            return result

        return transformed, composite_inverse

    def transform_rhs(self, rhs: cp.ndarray) -> cp.ndarray:
        """
        右辺ベクトルに全変換を順に適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        transformed_rhs = cp.asarray(rhs)

        # 各変換を順に適用
        for transformer in self.transformers:
            transformed_rhs = transformer.transform_rhs(transformed_rhs)

        return transformed_rhs

    def add_transformer(self, transformer: MatrixTransformer) -> None:
        """
        パイプラインに変換を追加

        Args:
            transformer: 追加する変換
        """
        self.transformers.append(transformer)


class TransformerFactory:
    """
    変換のファクトリークラス

    スケーリングと正則化の戦略を生成します（CuPy対応）
    """

    @staticmethod
    def create_scaling_strategy(
        name: str, matrix, params: Optional[Dict[str, Any]] = None
    ) -> MatrixTransformer:
        """
        スケーリング戦略を作成

        Args:
            name: スケーリング戦略名
            matrix: スケーリングする行列
            params: スケーリングパラメータ

        Returns:
            スケーリング戦略

        Raises:
            KeyError: 指定した名前の戦略が見つからない場合
        """
        from scaling_strategy import scaling_registry

        params = params or {}

        # スパース行列の場合は密行列に変換
        import cupyx.scipy.sparse as cpx_sparse

        if cpx_sparse.issparse(matrix):
            matrix = matrix.toarray()

        strategy_class = scaling_registry.get(name)
        return strategy_class(matrix, **params)

    @staticmethod
    def create_regularization_strategy(
        name: str, matrix, params: Optional[Dict[str, Any]] = None
    ) -> MatrixTransformer:
        """
        正則化戦略を作成

        Args:
            name: 正則化戦略名
            matrix: 正則化する行列
            params: 正則化パラメータ

        Returns:
            正則化戦略

        Raises:
            KeyError: 指定した名前の戦略が見つからない場合
        """
        from regularization_strategy import regularization_registry

        params = params or {}

        # スパース行列の場合は密行列に変換
        import cupyx.scipy.sparse as cpx_sparse

        if cpx_sparse.issparse(matrix):
            matrix = matrix.toarray()

        strategy_class = regularization_registry.get(name)
        return strategy_class(matrix, **params)

    @staticmethod
    def create_transformation_pipeline(
        matrix,
        scaling: str = "none",
        regularization: str = "none",
        scaling_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None,
    ) -> TransformationPipeline:
        """
        スケーリングと正則化を組み合わせたパイプラインを作成

        Args:
            matrix: 変換する行列
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            scaling_params: スケーリングパラメータ
            regularization_params: 正則化パラメータ

        Returns:
            変換パイプライン
        """
        pipeline = TransformationPipeline()

        # スケーリングを追加
        if scaling.lower() != "none":
            scaling_strategy = TransformerFactory.create_scaling_strategy(
                scaling, matrix, scaling_params
            )
            pipeline.add_transformer(scaling_strategy)

        # 正則化を追加
        if regularization.lower() != "none":
            regularization_strategy = TransformerFactory.create_regularization_strategy(
                regularization, matrix, regularization_params
            )
            pipeline.add_transformer(regularization_strategy)

        return pipeline
