"""
正規化スケーリング戦略

行と列のノルムに基づくスケーリングを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Tuple, Dict, Any, Callable, Union

from scaling_strategy import ScalingStrategy, scaling_registry


class NormalizationScaling(ScalingStrategy):
    """
    行と列のL2ノルムによるスケーリング

    行と列の要素をそれぞれのL2ノルムで正規化します
    """

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {}

    def apply_scaling(self) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        行と列のL2ノルムでスケーリングを適用

        各要素を行と列のL2ノルムの平方根で割ることでスケーリング

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 疎行列か密行列かで処理を分ける
        if self.is_sparse:
            # 疎行列の場合
            csr_matrix = self.matrix.tocsr() if not isinstance(self.matrix, cpx_sparse.csr_matrix) else self.matrix
            
            # 行と列のL2ノルムを計算（疎行列効率的な方法）
            row_norms = cp.zeros(csr_matrix.shape[0])
            for i in range(csr_matrix.shape[0]):
                row_start, row_end = csr_matrix.indptr[i], csr_matrix.indptr[i+1]
                if row_start < row_end:  # 行に非ゼロ要素がある場合
                    row_data = csr_matrix.data[row_start:row_end]
                    row_norms[i] = cp.sqrt(cp.sum(row_data * row_data))
            
            # 列の計算用に転置
            csc_matrix = csr_matrix.tocsc()
            col_norms = cp.zeros(csc_matrix.shape[1])
            for j in range(csc_matrix.shape[1]):
                col_start, col_end = csc_matrix.indptr[j], csc_matrix.indptr[j+1]
                if col_start < col_end:  # 列に非ゼロ要素がある場合
                    col_data = csc_matrix.data[col_start:col_end]
                    col_norms[j] = cp.sqrt(cp.sum(col_data * col_data))
        else:
            # 密行列の場合
            row_norms = cp.sqrt(cp.sum(self.matrix * self.matrix, axis=1))
            col_norms = cp.sqrt(cp.sum(self.matrix * self.matrix, axis=0))

        # 0除算を防ぐため、非常に小さい値をクリップ
        row_norms = cp.maximum(row_norms, 1e-10)
        col_norms = cp.maximum(col_norms, 1e-10)

        # スケーリング行列を作成（疎対角行列）
        d_row = 1.0 / cp.sqrt(row_norms)
        d_col = 1.0 / cp.sqrt(col_norms)
        D_row = cpx_sparse.diags(d_row)
        D_col = cpx_sparse.diags(d_col)

        # スケーリング行列を保存（右辺ベクトル変換用）
        self.scaling_matrix_row = D_row
        self.scaling_matrix_col = D_col

        # スケーリングを適用（疎行列演算）
        if self.is_sparse:
            L_scaled = D_row @ self.matrix @ D_col
        else:
            # 密行列の場合も疎対角行列との積
            L_scaled = D_row @ self.matrix @ D_col

        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled

        return L_scaled, inverse_scaling

    def transform_rhs(self, rhs: cp.ndarray) -> cp.ndarray:
        """
        右辺ベクトルにスケーリングを適用

        Args:
            rhs: 変換する右辺ベクトル

        Returns:
            変換された右辺ベクトル
        """
        # 行方向のスケーリングのみ適用（列方向は解に影響）
        if hasattr(self, "scaling_matrix_row"):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("normalization", NormalizationScaling)