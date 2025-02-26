"""
正則化戦略の基底クラスモジュール

CCD法の正則化戦略の基底クラスを提供します。
"""

import jax.numpy as jnp
from typing import Callable, Tuple, Dict, Any
from abc import ABC, abstractmethod

# 正則化戦略のレジストリ
from plugin_manager import PluginRegistry


class RegularizationStrategy(ABC):
    """正則化戦略の基底クラス"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, **kwargs):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
            **kwargs: その他のパラメータ
        """
        self.L = L
        self.K = K
        self._init_params(**kwargs)
    
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
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        正則化を適用し、正則化された行列とソルバー関数を返す
        
        Returns:
            正則化された行列L、正則化された行列K、ソルバー関数
        """
        pass


class NoneRegularization(RegularizationStrategy):
    """正則化なし"""
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """
        正則化なし - 標準的な行列の解法
        
        Returns:
            元の行列L、K、標準ソルバー関数
        """
        def solver_func(rhs):
            return jnp.linalg.solve(self.L, rhs)
        
        return self.L, self.K, solver_func


# 正則化戦略のレジストリを作成
regularization_registry = PluginRegistry(RegularizationStrategy, "正則化戦略")

# デフォルトの正則化戦略を登録
regularization_registry.register("none", NoneRegularization)
