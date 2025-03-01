"""
反復的スケーリング戦略

Sinkhorn-Knopp法による反復的スケーリングを提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class IterativeScaling(ScalingStrategy):
    """
    反復的スケーリング（Sinkhorn-Knopp法）
    
    行と列の和を反復的に均等化するスケーリング手法
    """
    
    def _init_params(self, **kwargs):
        """
        パラメータの初期化
        
        Args:
            **kwargs: 初期化パラメータ
        """
        self.max_iter = kwargs.get('max_iter', 10)
        self.tol = kwargs.get('tol', 1e-8)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        
        Returns:
            パラメータ情報の辞書
        """
        return {
            'max_iter': {
                'type': int,
                'default': 10,
                'help': '最大反復回数'
            },
            'tol': {
                'type': float,
                'default': 1e-8,
                'help': '収束判定閾値'
            }
        }
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        Sinkhorn-Knopp法による反復的スケーリングを適用
        
        行と列の和を反復的に均等化する手法で、数値的に安定したスケーリングを実現します。
        
        Returns:
            (スケーリングされた行列, 逆変換関数)
        """
        # 行列の絶対値を取得
        A = jnp.abs(self.matrix)
        
        # 行と列のスケーリングベクトルを初期化
        d_row = jnp.ones(A.shape[0])
        d_col = jnp.ones(A.shape[1])
        
        D_row = jnp.diag(d_row)
        D_col = jnp.diag(d_col)
        
        # JAX非互換のループを避けるため、固定回数実行
        for _ in range(self.max_iter):
            # 行のスケーリング
            row_sums = jnp.sum(D_row @ A @ D_col, axis=1)
            row_sums = jnp.maximum(row_sums, 1e-10)  # 0除算防止
            d_row_new = 1.0 / jnp.sqrt(row_sums)
            D_row_new = jnp.diag(d_row_new)
            
            # 列のスケーリング
            col_sums = jnp.sum(D_row_new @ A @ D_col, axis=0)
            col_sums = jnp.maximum(col_sums, 1e-10)  # 0除算防止
            d_col_new = 1.0 / jnp.sqrt(col_sums)
            D_col_new = jnp.diag(d_col_new)
            
            # 更新
            d_row, d_col = d_row_new, d_col_new
            D_row, D_col = D_row_new, D_col_new
        
        # スケーリング行列を保存
        self.scaling_matrix_row = D_row
        self.scaling_matrix_col = D_col
        
        # スケーリングを適用
        L_scaled = D_row @ self.matrix @ D_col
        
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
        # 行方向のスケーリングのみ適用
        if hasattr(self, 'scaling_matrix_row'):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("iterative", IterativeScaling)
