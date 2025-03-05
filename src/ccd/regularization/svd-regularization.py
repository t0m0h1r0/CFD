"""
SVD正則化戦略

特異値分解に基づく正則化戦略を提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Tuple, Dict, Any, Callable, Union

from regularization_strategy import RegularizationStrategy, regularization_registry


class SVDRegularization(RegularizationStrategy):
    """
    SVD切断法による正則化

    特異値分解を使用し、閾値以下の特異値を切断値に置き換える
    """

    def _init_params(self, **kwargs):
        """
        パラメータの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        self.threshold = kwargs.get("threshold", 1e-10)

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {
            "threshold": {
                "type": float,
                "default": 1e-10,
                "help": "特異値の切断閾値（この値より小さい特異値は切断値に置き換える）",
            }
        }

    def apply_regularization(
        self,
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        SVD切断法による正則化を適用

        Returns:
            (正則化された行列, 逆変換関数)
        """
        try:
            # 疎行列の場合は密行列に変換（SVDには密行列が必要）
            if self.is_sparse:
                matrix_dense = self.matrix.toarray()
                # 行列のスケールを確認
                matrix_norm = cp.linalg.norm(matrix_dense, ord=2)
                
                # 行列のスケールが大きい場合はスケーリング
                if matrix_norm > 1.0:
                    self.reg_factor = 1.0 / matrix_norm
                    L_scaled = matrix_dense * self.reg_factor
                else:
                    self.reg_factor = 1.0
                    L_scaled = matrix_dense
            else:
                # 密行列の場合
                matrix_norm = cp.linalg.norm(self.matrix, ord=2)
                
                if matrix_norm > 1.0:
                    self.reg_factor = 1.0 / matrix_norm
                    L_scaled = self.matrix * self.reg_factor
                else:
                    self.reg_factor = 1.0
                    L_scaled = self.matrix

            # 特異値分解を実行
            U, s, Vh = cp.linalg.svd(L_scaled, full_matrices=False)

            # 特異値のフィルタリング
            s_filtered = cp.maximum(s, self.threshold)

            # 正則化された行列を計算
            L_reg = U @ cp.diag(s_filtered) @ Vh

            # 疎行列の場合、結果も疎行列に変換
            if self.is_sparse:
                # しきい値を設定して疎行列化（小さい値を0に）
                sparse_threshold = 1e-12
                L_reg_sparse = cp.where(cp.abs(L_reg) > sparse_threshold, L_reg, 0)
                L_reg = cpx_sparse.csr_matrix(L_reg_sparse)

            # 逆変換関数
            def inverse_transform(x_reg):
                return x_reg / self.reg_factor

            return L_reg, inverse_transform
            
        except Exception as e:
            print(f"Error in SVD regularization: {e}")
            print("Falling back to no regularization")
            return self.matrix, lambda x: x


# 正則化戦略をレジストリに登録
regularization_registry.register("svd", SVDRegularization)