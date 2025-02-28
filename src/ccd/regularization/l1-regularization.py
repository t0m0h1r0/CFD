"""
L1正則化戦略（LASSO）

解のL1ノルムに対するペナルティを課すL1正則化を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from regularization_strategy import RegularizationStrategy, regularization_registry


class L1Regularization(RegularizationStrategy):
    """
    L1正則化（LASSO）
    
    解のL1ノルムに対するペナルティを課す正則化手法
    """
    
    def _init_params(self, **kwargs):
        """
        パラメータの初期化
        
        Args:
            **kwargs: 初期化パラメータ
        """
        self.alpha = kwargs.get('alpha', 1e-4)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        
        Returns:
            パラメータ情報の辞書
        """
        return {
            'alpha': {
                'type': float,
                'default': 1e-4,
                'help': '正則化パラメータ'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        L1正則化（LASSO）を適用
        
        解のL1ノルムに対するペナルティを課す正則化手法で、
        スパース性（多くの要素がゼロ）を持つ解を生成します。
        
        Returns:
            (正則化された行列, 逆変換関数)
        """
        # 行列のスケールを確認
        matrix_norm = jnp.linalg.norm(self.matrix, ord=2)
        
        # 行列のスケールが大きい場合はスケーリング
        if matrix_norm > 1.0:
            self.reg_factor = 1.0 / matrix_norm
            L_scaled = self.matrix * self.reg_factor
            alpha_scaled = self.alpha * self.reg_factor
        else:
            self.reg_factor = 1.0
            L_scaled = self.matrix
            alpha_scaled = self.alpha
        
        # L1正則化では対角成分に正則化パラメータを加算
        n = L_scaled.shape[1]
        # 単位行列に正則化パラメータをスケールして加算
        L_reg = L_scaled + alpha_scaled * jnp.eye(n)
        
        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor
        
        return L_reg, inverse_transform


# 正則化戦略をレジストリに登録
regularization_registry.register("l1", L1Regularization)
