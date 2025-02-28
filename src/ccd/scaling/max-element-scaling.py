"""
最大成分スケーリング戦略（Max Element Scaling）

CCD法の最大成分スケーリング戦略を提供します。
行列全体の最大絶対値が1になるようスケーリングします。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategies_base import ScalingStrategy, scaling_registry


class MaxElementScaling(ScalingStrategy):
    """最大成分スケーリング（Max Element Scaling）"""
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        最大成分スケーリングを適用
        
        行列全体の最大絶対値が1になるようスケーリングします。
        非常にシンプルなスケーリング手法です。
        
        Returns:
            スケーリングされた行列L、逆変換関数
        """
        # 行列全体の最大絶対値を取得
        max_abs_value = jnp.max(jnp.abs(self.L))
        
        # 0除算を防ぐため、非常に小さい値をクリップ
        max_abs_value = jnp.maximum(max_abs_value, 1e-10)
        
        # スケーリング係数を保存
        self.scale_factor = max_abs_value
        
        # スケーリングを適用
        L_scaled = self.L / max_abs_value
        
        # 右辺ベクトルスケーリング関数を定義
        def scale_rhs(rhs):
            return rhs / max_abs_value
        
        # 逆変換関数 - この場合はスケーリングが一様なので、何もしない
        def inverse_scaling(X_scaled):
            return X_scaled
        
        # スケーリング情報を保存
        self.scale_rhs = scale_rhs
        
        return L_scaled, inverse_scaling


# スケーリング戦略をレジストリに登録
scaling_registry.register("max_element", MaxElementScaling)