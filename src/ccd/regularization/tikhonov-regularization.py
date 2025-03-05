"""
ティホノフ正則化戦略

Tikhonov正則化（L2正則化）の実装を提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Tuple, Dict, Any, Callable, Union

from regularization_strategy import RegularizationStrategy, regularization_registry


class TikhonovRegularization(RegularizationStrategy):
    """
    Tikhonov正則化（L2正則化）

    行列に単位行列の定数倍を加算する正則化手法
    """

    def _init_params(self, **kwargs):
        """
        パラメータの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        self.alpha = kwargs.get("alpha", 1e-6)

    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す

        Returns:
            パラメータ情報の辞書
        """
        return {
            "alpha": {
                "type": float,
                "default": 1e-6,
                "help": "正則化パラメータ（大きいほど正則化の効果が強くなる）",
            }
        }

    def apply_regularization(
        self,
    ) -> Tuple[Union[cp.ndarray, cpx_sparse.spmatrix], Callable[[cp.ndarray], cp.ndarray]]:
        """
        Tikhonov正則化を適用

        Returns:
            (正則化された行列, 逆変換関数)
        """
        # 行列のスケールを確認（疎行列でも計算可能）
        if self.is_sparse:
            # 疎行列の場合はSVDの代わりに行列ノルムを計算
            matrix_norm = cpx_sparse.linalg.norm(self.matrix, ord=2)
        else:
            matrix_norm = cp.linalg.norm(self.matrix, ord=2)

        # 行列のスケールが大きい場合はスケーリング
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.matrix * self.reg_factor
            alpha_scaled = self.alpha * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.matrix
            alpha_scaled = self.alpha

        # L2正則化を考慮した行列の計算
        n = L_scaled.shape[0]
        
        # 疎行列の場合は効率的に単位行列を加算
        if self.is_sparse:
            # 疎行列に対する対角成分の加算
            # 各対角要素にalpha_scaledを加算
            if isinstance(L_scaled, cpx_sparse.csr_matrix) or isinstance(L_scaled, cpx_sparse.csc_matrix):
                # 対角要素のインデックスを取得
                diag_indices = cp.arange(n)
                # 対角要素にアクセスするための行列
                eye = cpx_sparse.eye(n, format=L_scaled.format, dtype=L_scaled.dtype)
                # 正則化された行列 = 元の行列 + alpha * 単位行列
                L_reg = L_scaled + alpha_scaled * eye
            else:
                # その他の形式の場合はCSRに変換
                L_scaled_csr = L_scaled.tocsr()
                eye = cpx_sparse.eye(n, format='csr', dtype=L_scaled_csr.dtype)
                L_reg = L_scaled_csr + alpha_scaled * eye
        else:
            # 密行列の場合は単位行列を生成して加算
            I = cp.eye(n)
            L_reg = L_scaled + alpha_scaled * I

        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor

        return L_reg, inverse_transform


# 正則化戦略をレジストリに登録
regularization_registry.register("tikhonov", TikhonovRegularization)