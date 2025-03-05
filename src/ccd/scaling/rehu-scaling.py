"""
Rehu スケーリング戦略

行と列の最大絶対値によるスケーリングを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Tuple, Dict, Any, Callable, Union

from scaling_strategy import ScalingStrategy, scaling_registry


class RehuScaling(ScalingStrategy):
    """
    Rehu法によるスケーリング

    各行と列の最大絶対値の平方根でスケーリングする効果的な方法
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
        Rehu法（行と列の最大絶対値）によるスケーリングを適用

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        try:
            # 疎行列かどうかで処理を分ける
            if self.is_sparse:
                # 疎行列の場合
                csr_matrix = self.matrix.tocsr() if not isinstance(self.matrix, cpx_sparse.csr_matrix) else self.matrix
                
                # 行の最大絶対値を計算
                max_values_row = cp.zeros(csr_matrix.shape[0], dtype=cp.float64)
                for i in range(csr_matrix.shape[0]):
                    row_start, row_end = csr_matrix.indptr[i], csr_matrix.indptr[i+1]
                    if row_start < row_end:  # 行に非ゼロ要素がある場合
                        row_data = csr_matrix.data[row_start:row_end]
                        max_values_row[i] = cp.max(cp.abs(row_data))
                
                # 列の最大絶対値を計算
                csc_matrix = csr_matrix.tocsc()
                max_values_col = cp.zeros(csc_matrix.shape[1], dtype=cp.float64)
                for j in range(csc_matrix.shape[1]):
                    col_start, col_end = csc_matrix.indptr[j], csc_matrix.indptr[j+1]
                    if col_start < col_end:  # 列に非ゼロ要素がある場合
                        col_data = csc_matrix.data[col_start:col_end]
                        max_values_col[j] = cp.max(cp.abs(col_data))
            else:
                # 密行列の場合
                max_values_row = cp.max(cp.abs(self.matrix), axis=1)
                max_values_col = cp.max(cp.abs(self.matrix), axis=0)

            # 0除算を防ぐため、非常に小さい値をクリップ
            max_values_row = cp.maximum(max_values_row, 1e-10)
            max_values_col = cp.maximum(max_values_col, 1e-10)

            # スケーリング係数の計算
            d_row = 1.0 / cp.sqrt(max_values_row)
            d_col = 1.0 / cp.sqrt(max_values_col)

            # 疎行列のスケーリング行列を作成
            D_row = cpx_sparse.diags(d_row)
            D_col = cpx_sparse.diags(d_col)

            # スケーリング行列を保存（右辺ベクトル変換用）
            self.scaling_matrix_row = D_row
            self.scaling_matrix_col = D_col

            # スケーリングを適用
            L_scaled = D_row @ self.matrix @ D_col

            # 逆変換関数
            def inverse_scaling(X_scaled):
                return D_col @ X_scaled

            return L_scaled, inverse_scaling
            
        except Exception as e:
            print(f"Error in Rehu scaling: {e}")
            print("Falling back to no scaling")
            return self.matrix, lambda x: x

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
            try:
                return self.scaling_matrix_row @ rhs
            except Exception as e:
                print(f"Error transforming RHS: {e}")
                return rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("rehu", RehuScaling)