"""
変換戦略インターフェース

スケーリングと正則化のための共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Callable, Protocol, Optional, Union
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse


class MatrixTransformer(Protocol):
    """
    行列変換プロトコル

    行列と右辺ベクトルの変換に関する共通インターフェース（CuPy疎行列対応）
    """

    def transform_matrix(
        self, matrix: Optional[Union[cp.ndarray, cpx_sparse.spmatrix]] = None
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        行列を変換し、逆変換関数を返す

        Args:
            matrix: 変換する行列（Noneの場合はクラスに保持されている行列を使用）

        Returns:
            (変換された行列, 逆変換関数)
        """
        ...

    def transform_rhs(self, rhs: cp.ndarray) -> cp.ndarray:
        """
        右辺ベクトルを変換する

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        ...


class TransformationStrategy(ABC):
    """
    変換戦略の抽象基底クラス

    スケーリングと正則化の両方に使用される基本機能を提供します（CuPy疎行列対応）
    """

    def __init__(self, matrix: Union[cp.ndarray, cpx_sparse.spmatrix], **kwargs):
        """
        初期化

        Args:
            matrix: 変換する行列
            **kwargs: 戦略固有のパラメータ
        """
        # 疎行列かどうかを記録
        self.is_sparse = cpx_sparse.issparse(matrix)
        # 元の行列を保存
        self.matrix = matrix
        self._init_params(**kwargs)

    def _init_params(self, **kwargs):
        """
        パラメータの初期化 - サブクラスでオーバーライド可能

        Args:
            **kwargs: 初期化パラメータ
        """
        pass

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        戦略のパラメータ情報を返す

        Returns:
            パラメータ名をキー、パラメータ情報を値とする辞書
            {パラメータ名: {'type': 型, 'default': デフォルト値, 'help': ヘルプテキスト}}
        """
        return {}

    @abstractmethod
    def transform_matrix(
        self, matrix: Optional[Union[cp.ndarray, cpx_sparse.spmatrix]] = None
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        行列を変換し、逆変換関数を返す

        Args:
            matrix: 変換する行列（指定がない場合は初期化時の行列を使用）

        Returns:
            (変換された行列, 逆変換関数)
        """
        pass

    def transform_rhs(self, rhs: cp.ndarray) -> cp.ndarray:
        """
        右辺ベクトルを変換する

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        # デフォルトでは何もしない
        return rhs


class NoneTransformation(TransformationStrategy):
    """
    変換なし - 恒等変換（CuPy疎行列対応）
    """

    def transform_matrix(
        self, matrix: Optional[Union[cp.ndarray, cpx_sparse.spmatrix]] = None
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        行列をそのまま返し、恒等関数を逆変換として返す

        Args:
            matrix: 変換する行列（指定がない場合は初期化時の行列を使用）

        Returns:
            (元の行列, 恒等関数)
        """
        if matrix is not None:
            self.matrix = matrix
        return self.matrix, lambda x: x