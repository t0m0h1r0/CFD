"""
対角優位スケーリング戦略

CCD法の対角優位スケーリング戦略を提供します。
対角要素が1になるようスケーリングします。
右辺ベクトルのスケーリングをサポートするように修正しました。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class DiagonalDominanceScaling(ScalingStrategy):
    """対角優位スケーリング"""
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        対角優位スケーリングを適用
        
        対角要素が1になるようスケーリングし、他の要素は相対的に小さくなります。
        数値解法の収束特性の改善に役立ちます。
        
        Returns:
            スケーリングされた行列L、逆変換関数
        """
        n = self.L.shape[0]
        diag_elements = jnp.diag(self.L)
        
        # 対角要素が0の場合に備えて小さな値を加える
        diag_elements = jnp.where(jnp.abs(diag_elements) < 1e-10, 1e-10, diag_elements)
        
        # スケーリング行列を作成
        D = jnp.diag(1.0 / diag_elements)
        
        # 行と列のスケーリング行列として同じ行列を使用
        self.scaling_matrix_row = D
        self.scaling_matrix_col = D
        
        # スケーリングを適用
        L_scaled = D @ self.L @ D
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D @ X_scaled
        
        return L_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("diagonal_dominance", DiagonalDominanceScaling)