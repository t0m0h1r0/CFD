"""
スケーリング戦略の基底クラスモジュール

CCD法のスケーリング戦略の基底クラスを提供します。
右辺ベクトルのスケーリングをサポートするように修正しました。
"""

import jax.numpy as jnp
from typing import Callable, Tuple, Dict, Any
from abc import ABC, abstractmethod

# スケーリング戦略のレジストリ
from plugin_manager import PluginRegistry


class ScalingStrategy(ABC):
    """スケーリング戦略の基底クラス"""
    
    def __init__(self, L: jnp.ndarray, **kwargs):
        """
        Args:
            L: スケーリングする行列
            **kwargs: その他のパラメータ
        """
        self.L = L
        self._init_params(**kwargs)
        
        # 変換情報の初期化
        self.scaling_matrix_row = None
        self.scaling_matrix_col = None
    
    def _init_params(self, **kwargs):
        """
        パラメータの初期化 - サブクラスでオーバーライド可能
        
        Args:
            **kwargs: 初期化パラメータ
        """
        pass
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        戦略のパラメータ情報を返す
        
        Returns:
            パラメータ名をキー、パラメータ情報を値とする辞書
            {パラメータ名: {'type': 型, 'default': デフォルト値, 'help': ヘルプテキスト}}
        """
        return {}
    
    @abstractmethod
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        スケーリングを適用し、スケーリングされた行列と逆変換関数を返す
        
        Returns:
            スケーリングされた行列L、逆変換関数
        """
        pass
    
    def scale_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルにスケーリングを適用
        
        線形方程式 L * x = rhs において、L をスケーリングした場合、
        右辺ベクトル rhs も対応してスケーリングする必要があります。
        
        Args:
            rhs: 右辺ベクトル
            
        Returns:
            スケーリングされた右辺ベクトル
        """
        # 行方向のスケーリングがあれば適用
        if self.scaling_matrix_row is not None:
            return self.scaling_matrix_row @ rhs
        
        # デフォルトでは何もしない
        return rhs


class NoneScaling(ScalingStrategy):
    """スケーリングなし"""
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable]:
        """
        スケーリングなし - 元の行列をそのまま返す
        
        Returns:
            元の行列L、恒等関数
        """
        return self.L, lambda x: x


# スケーリング戦略のレジストリを作成
scaling_registry = PluginRegistry(ScalingStrategy, "スケーリング戦略")

# デフォルトのスケーリング戦略を登録
scaling_registry.register("none", NoneScaling)