"""
切断特異値分解（TSVD）正則化戦略

特定のランク以上の特異値を完全に切り捨てる正則化戦略を提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Tuple, Dict, Any, Callable, Union

from regularization_strategy import RegularizationStrategy, regularization_registry


class TSVDRegularization(RegularizationStrategy):
    """
    切断特異値分解（Truncated SVD）による正則化

    指定したランク以上の特異値を完全に切り捨てる
    """

    def _init_params(self, **kwargs):
        """
        パラメータの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        self.rank = kwargs.get("rank", None)
        self.threshold_ratio = kwargs.get("threshold_ratio", 1e-5)

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {
            "rank": {
                "type": int,
                "default": None,
                "help": "保持する特異値の数（Noneの場合は閾値比率で決定）",
            },
            "threshold_ratio": {
                "type": float,
                "default": 1e-5,
                "help": "最大特異値との比率による閾値（rank=Noneの場合のみ使用）",
            },
        }

    def apply_regularization(
        self,
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        切断特異値分解（TSVD）による正則化を適用

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

            # 使用するランクを決定
            if self.rank is None:
                # 閾値比率に基づいてランクを決定
                threshold = cp.max(s) * self.threshold_ratio
                # 閾値以上の特異値の数をカウント
                rank = cp.sum(s > threshold)
                # 整数値に変換
                rank = int(rank.item())
            else:
                # ランクが行列の最小次元を超えないようにする
                min_dim = min(L_scaled.shape[0], L_scaled.shape[1])
                rank = min(self.rank, min_dim)

            # 特異値をトランケート
            s_truncated = cp.zeros_like(s)
            s_truncated[:rank] = s[:rank]

            # 正則化された行列を計算
            L_reg = U @ cp.diag(s_truncated) @ Vh

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
            print(f"Error in TSVD regularization: {e}")
            print("Falling back to no regularization")
            return self.matrix, lambda x: x


# 正則化戦略をレジストリに登録
regularization_registry.register("tsvd", TSVDRegularization)