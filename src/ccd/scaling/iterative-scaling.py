"""
反復的スケーリング戦略

CCD法の反復的スケーリング戦略（Sinkhorn-Knopp法）を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class IterativeScaling(ScalingStrategy):
    """反復的スケーリング（Sinkhorn-Knopp法）"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.max_iter = kwargs.get('max_iter', 10)
        self.tol = kwargs.get('tol', 1e-8)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
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
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        Sinkhorn-Knopp法による反復的スケーリングを適用
        
        行と列の和を反復的に均等化する手法で、数値的に安定したスケーリングを実現します。
        
        Returns:
            スケーリングされた行列L、逆変換関数
        """
        # 行列の絶対値を取得
        A = jnp.abs(self.L)
        
        # 行と列のスケーリングベクトルを初期化
        d_row = jnp.ones(A.shape[0])
        d_col = jnp.ones(A.shape[1])
        
        D_row = jnp.diag(d_row)
        D_col = jnp.diag(d_col)
        
        # 反復的にスケーリングを適用
        for _ in range(self.max_iter):
            # 行のスケーリング
            row_sums = jnp.sum(D_row @ A @ D_col, axis=1)
            d_row_new = 1.0 / jnp.sqrt(row_sums)
            D_row_new = jnp.diag(d_row_new)
            
            # 列のスケーリング
            col_sums = jnp.sum(D_row_new @ A @ D_col, axis=0)
            d_col_new = 1.0 / jnp.sqrt(col_sums)
            D_col_new = jnp.diag(d_col_new)
            
            # 収束判定
            if (jnp.max(jnp.abs(d_row_new - d_row)) < self.tol and 
                jnp.max(jnp.abs(d_col_new - d_col)) < self.tol):
                break
            
            d_row, d_col = d_row_new, d_col_new
            D_row, D_col = D_row_new, D_col_new
        
        # 最終的なスケーリング行列を保存
        self.D_row = D_row
        self.D_col = D_col
        
        # スケーリングを適用
        L_scaled = self.D_row @ self.L @ self.D_col
        
        # 修正：右辺ベクトル変換用の関数を内部関数として定義
        def scale_rhs(rhs):
            # 右辺ベクトルをスケーリング
            return self.D_row @ rhs
        
        # 修正：スケーリングに対応した逆変換関数
        def inverse_scaling(X_scaled):
            return self.D_col @ X_scaled
        
        # スケーリング情報を保存
        self.scale_rhs = scale_rhs
        self.inverse_transform = inverse_scaling
        
        return L_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("iterative", IterativeScaling)