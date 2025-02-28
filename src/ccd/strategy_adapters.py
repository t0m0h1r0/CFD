"""
スケーリングと正則化のアダプタークラス

既存のスケーリング戦略と正則化戦略をMatrixTransformerプロトコルに適応させるアダプタークラスを提供します。
これにより、既存のコードに変更を加えることなく、新しいインターフェースで利用できます。
"""

import jax.numpy as jnp
from typing import Tuple, Callable

from scaling_strategies_base import ScalingStrategy
from regularization_strategies_base import RegularizationStrategy


class ScalingStrategyAdapter:
    """既存のScalingStrategyをMatrixTransformerプロトコルに適応させるアダプター"""
    
    def __init__(self, strategy: ScalingStrategy):
        """
        Args:
            strategy: 適応させるScalingStrategy
        """
        self.strategy = strategy
    
    def transform_matrix(self, matrix: jnp.ndarray) -> Tuple[jnp.ndarray, Callable]:
        """
        スケーリングを適用し、逆変換関数を返す
        
        Args:
            matrix: 元の行列
            
        Returns:
            スケーリングされた行列、逆変換関数
        """
        return self.strategy.apply_scaling()
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルにスケーリングを適用
        
        Args:
            rhs: 右辺ベクトル
            
        Returns:
            スケーリングされた右辺ベクトル
        """
        return self.strategy.scale_rhs(rhs)


class RegularizationStrategyAdapter:
    """既存のRegularizationStrategyをMatrixTransformerプロトコルに適応させるアダプター"""
    
    def __init__(self, strategy: RegularizationStrategy):
        """
        Args:
            strategy: 適応させるRegularizationStrategy
        """
        self.strategy = strategy
    
    def transform_matrix(self, matrix: jnp.ndarray) -> Tuple[jnp.ndarray, Callable]:
        """
        正則化を適用し、逆変換関数を返す
        
        Args:
            matrix: 元の行列
            
        Returns:
            正則化された行列、逆変換関数
        """
        return self.strategy.apply_regularization()
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルに正則化の変換を適用
        
        Args:
            rhs: 右辺ベクトル
            
        Returns:
            変換された右辺ベクトル
        """
        return self.strategy.transform_rhs(rhs)


class ScalingStrategyFactory:
    """スケーリング戦略のファクトリークラス"""
    
    @staticmethod
    def create_adapter(strategy_name: str, matrix: jnp.ndarray, **kwargs) -> ScalingStrategyAdapter:
        """
        名前からスケーリング戦略アダプターを作成
        
        Args:
            strategy_name: スケーリング戦略名
            matrix: 行列
            **kwargs: スケーリングパラメータ
            
        Returns:
            スケーリング戦略アダプター
        """
        from scaling_strategies_base import scaling_registry
        
        strategy_class = scaling_registry.get(strategy_name)
        strategy = strategy_class(matrix, **kwargs)
        return ScalingStrategyAdapter(strategy)


class RegularizationStrategyFactory:
    """正則化戦略のファクトリークラス"""
    
    @staticmethod
    def create_adapter(strategy_name: str, matrix: jnp.ndarray, **kwargs) -> RegularizationStrategyAdapter:
        """
        名前から正則化戦略アダプターを作成
        
        Args:
            strategy_name: 正則化戦略名
            matrix: 行列
            **kwargs: 正則化パラメータ
            
        Returns:
            正則化戦略アダプター
        """
        from regularization_strategies_base import regularization_registry
        
        strategy_class = regularization_registry.get(strategy_name)
        strategy = strategy_class(matrix, **kwargs)
        return RegularizationStrategyAdapter(strategy)