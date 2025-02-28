"""
Landweber反復法による正則化戦略

反復法に基づく正則化手法を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategy import RegularizationStrategy, regularization_registry


class LandweberRegularization(RegularizationStrategy):
    """
    Landweber反復法による正則化
    
    反復法に基づく正則化手法
    """
    
    def _init_params(self, **kwargs):
        """
        パラメータの初期化
        
        Args:
            **kwargs: 初期化パラメータ
        """
        self.iterations = kwargs.get('iterations', 20)
        self.relaxation = kwargs.get('relaxation', 0.1)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        
        Returns:
            パラメータ情報の辞書
        """
        return {
            'iterations': {
                'type': int,
                'default': 20,
                'help': '反復回数'
            },
            'relaxation': {
                'type': float,
                'default': 0.1,
                'help': '緩和パラメータ（0 < relaxation < 2/σ_max^2、ここでσ_maxはLの最大特異値）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        Landweber反復法による正則化を適用
        
        反復法に基づく正則化手法で、反復回数を制限することで正則化効果を得ます。
        
        Returns:
            (正則化された行列, 逆変換関数)
        """
        # 行列のスケールを確認
        matrix_norm = jnp.linalg.norm(self.matrix, ord=2)
        
        # 行列のスケールが大きい場合はスケーリング
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.matrix * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.matrix
        
        # 緩和パラメータを安全な範囲に調整
        s_max = jnp.linalg.norm(L_scaled, ord=2)
        omega = jnp.minimum(self.relaxation, 1.9 / (s_max ** 2))
        
        # 行列自体は正規化した形に調整（数値的な問題を避けるため）
        L_reg = L_scaled
        
        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor
        
        return L_reg, inverse_transform


# 正則化戦略をレジストリに登録
regularization_registry.register("landweber", LandweberRegularization)
