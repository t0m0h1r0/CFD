"""
変換パイプライン

スケーリングと正則化を順に適用するパイプラインを提供します。
"""

import jax.numpy as jnp
from typing import Optional, Tuple, Callable, Dict, Any, List

from strategy_interface import MatrixTransformer


class TransformationPipeline:
    """
    変換パイプライン

    複数の変換を順次適用して、結果を結合します
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
        self, matrix: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        全変換を順に適用

        Args:
            matrix: 変換する行列

        Returns:
            (変換された行列, 逆変換関数)
        """
        self.original_matrix = matrix
        transformed = matrix
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

    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルに全変換を順に適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        transformed_rhs = rhs

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

    スケーリングと正則化の戦略を生成します
    """

    @staticmethod
    def create_scaling_strategy(
        name: str, matrix: jnp.ndarray, params: Optional[Dict[str, Any]] = None
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
        strategy_class = scaling_registry.get(name)
        return strategy_class(matrix, **params)

    @staticmethod
    def create_regularization_strategy(
        name: str, matrix: jnp.ndarray, params: Optional[Dict[str, Any]] = None
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
        strategy_class = regularization_registry.get(name)
        return strategy_class(matrix, **params)

    @staticmethod
    def create_transformation_pipeline(
        matrix: jnp.ndarray,
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
        scaling_strategy = TransformerFactory.create_scaling_strategy(
            scaling, matrix, scaling_params
        )
        pipeline.add_transformer(scaling_strategy)

        # 正則化を追加
        regularization_strategy = TransformerFactory.create_regularization_strategy(
            regularization, matrix, regularization_params
        )
        pipeline.add_transformer(regularization_strategy)

        return pipeline
