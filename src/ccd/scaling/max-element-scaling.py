"""
最大成分スケーリング戦略

行列全体の最大絶対値が1になるようスケーリングする手法を提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Tuple, Dict, Any, Callable, Union

from scaling_strategy import ScalingStrategy, scaling_registry


class MaxElementScaling(ScalingStrategy):
    """
    最大成分スケーリング

    行列全体の最大絶対値が1になるようスケーリングします
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
        最大成分スケーリングを適用

        行列全体の最大絶対値が1になるようスケーリングします。
        非常にシンプルなスケーリング手法です。

        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        try:
            # 疎行列の場合
            if self.is_sparse:
                # 疎行列の最大絶対値を計算 - data配列の最大値
                if hasattr(self.matrix, 'data') and len(self.matrix.data) > 0:
                    max_abs_value = cp.max(cp.abs(self.matrix.data))
                else:
                    # データがない場合は0（ただし0除算を避けるため1e-10を設定）
                    max_abs_value = 1e-10
            else:
                # 密行列の場合
                max_abs_value = cp.max(cp.abs(self.matrix))

            # 0除算を防ぐため、非常に小さい値をクリップ
            max_abs_value = cp.maximum(max_abs_value, 1e-10)

            # スケーリング係数を保存
            self.scale_factor = float(max_abs_value)  # CuPy配列からスカラー値に変換

            # スケーリングを適用
            if self.is_sparse:
                # 疎行列のスケーリング - 新しい疎行列を作成
                if isinstance(self.matrix, cpx_sparse.csr_matrix):
                    L_scaled = cpx_sparse.csr_matrix(self.matrix)
                    L_scaled.data = L_scaled.data / self.scale_factor
                elif isinstance(self.matrix, cpx_sparse.csc_matrix):
                    L_scaled = cpx_sparse.csc_matrix(self.matrix)
                    L_scaled.data = L_scaled.data / self.scale_factor
                else:
                    # その他の形式は一旦CSRに変換
                    csr_matrix = self.matrix.tocsr()
                    L_scaled = cpx_sparse.csr_matrix(csr_matrix)
                    L_scaled.data = L_scaled.data / self.scale_factor
            else:
                # 密行列のスケーリング
                L_scaled = self.matrix / self.scale_factor

            # 逆変換関数 - この場合はスケーリングが一様なので、追加の変換は不要
            def inverse_scaling(X_scaled):
                return X_scaled

            return L_scaled, inverse_scaling
            
        except Exception as e:
            print(f"Error in MaxElement scaling: {e}")
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
        # 行列と同じスケールで右辺ベクトルもスケーリング
        if hasattr(self, "scale_factor"):
            try:
                return rhs / self.scale_factor
            except Exception as e:
                print(f"Error transforming RHS: {e}")
                return rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("max_element", MaxElementScaling)