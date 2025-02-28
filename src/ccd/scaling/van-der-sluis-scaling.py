"""
Van der Sluis スケーリング戦略

各列を2-ノルムで正規化するスケーリング手法を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class VanDerSluisScaling(ScalingStrategy):
    """
    Van der Sluis スケーリング
    
    各列を2-ノルムで正規化するスケーリング手法
    """
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        
        Returns:
            パラメータ情報の辞書
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        Van der Sluis スケーリングを適用
        
        各列を2-ノルムで正規化するスケーリング手法です。
        行列の列間でスケールが大きく異なる場合に特に効果的です。
        
        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 各列の2-ノルムを計算
        col_norms = jnp.sqrt(jnp.sum(self.matrix * self.matrix, axis=0))
        
        # 0除算を防ぐため、非常に小さい値をクリップ
        col_norms = jnp.maximum(col_norms, 1e-10)
        
        # スケーリング行列を作成（列のみスケーリング）
        D_col = jnp.diag(1.0 / col_norms)
        
        # スケーリング行列を保存
        self.scaling_matrix_row = jnp.eye(self.matrix.shape[0])  # 行方向のスケーリングなし
        self.scaling_matrix_col = D_col
        
        # スケーリングを適用
        L_scaled = self.matrix @ D_col
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, inverse_scaling
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルにスケーリングを適用
        
        Args:
            rhs: 変換する右辺ベクトル
            
        Returns:
            変換された右辺ベクトル
        """
        # 行方向のスケーリングのみ適用（この場合は単位行列なので変化なし）
        if hasattr(self, 'scaling_matrix_row'):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("van_der_sluis", VanDerSluisScaling)
