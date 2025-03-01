"""
ティホノフ正則化戦略

Tikhonov正則化（L2正則化）の実装を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

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
        self.alpha = kwargs.get('alpha', 1e-6)
    
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
                'default': 1e-6,
                'help': '正則化パラメータ（大きいほど正則化の効果が強くなる）'
            }
        }
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        Tikhonov正則化を適用
        
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
        
        # L2正則化を考慮した行列の計算
        n = L_scaled.shape[0]
        # 単位行列を生成
        I = jnp.eye(n)
        # 行列を正則化
        L_reg = L_scaled + alpha_scaled * I
        
        # 逆変換関数
        def inverse_transform(x_reg):
            return x_reg / self.reg_factor
        
        return L_reg, inverse_transform


# 正則化戦略をレジストリに登録
regularization_registry.register("tikhonov", TikhonovRegularization)
