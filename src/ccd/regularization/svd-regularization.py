"""
SVD切断法による正則化戦略

CCD法のSVD切断法による正則化戦略を提供します。
右辺ベクトルの変換と解の逆変換に対応しました。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategies_base import RegularizationStrategy, regularization_registry


class SVDRegularization(RegularizationStrategy):
    """SVD切断法による正則化"""
    
    def _init_params(self, **kwargs):
        """パラメータの初期化"""
        self.threshold = kwargs.get('threshold', 1e-10)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {
            'threshold': {
                'type': float,
                'default': 1e-10,
                'help': '特異値の切断閾値（この値より小さい特異値は切断値に置き換える）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable]:
        """
        SVD切断法による正則化を適用
        
        特異値分解を使用し、一定の閾値より小さい特異値を切断値に置き換えて
        数値的な安定性を向上させる正則化手法です。
        
        Returns:
            正則化された行列L、逆変換関数
        """
        # 行列のスケールを確認
        matrix_norm = jnp.linalg.norm(self.L, ord=2)
        
        # 行列のスケールが大きい場合はスケーリング
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.L * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.L
        
        # 特異値分解を実行
        U, s, Vh = jnp.linalg.svd(L_scaled, full_matrices=False)
        
        # 特異値のフィルタリング（JAX互換）
        s_filtered = jnp.maximum(s, self.threshold)
        
        # 正則化された行列を計算
        L_reg = U @ jnp.diag(s_filtered) @ Vh
        
        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor
        
        return L_reg, inverse_transform
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """右辺ベクトルに正則化の変換を適用"""
        # 行列と同じスケーリングを右辺ベクトルにも適用
        return rhs * self.reg_factor


# 正則化戦略をレジストリに登録
regularization_registry.register("svd", SVDRegularization)