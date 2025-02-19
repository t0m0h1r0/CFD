from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

@dataclass
class TimeIntegrationConfig:
    """時間発展の設定"""
    dt: float  # 時間ステップ幅
    check_stability: bool = True  # 安定性チェックを行うか
    adaptive_dt: bool = False  # 適応的な時間ステップ制御を行うか
    safety_factor: float = 0.9  # 安定性のための安全係数
    
    def validate(self) -> None:
        """設定の妥当性確認"""
        if self.dt <= 0:
            raise ValueError(f"無効な時間ステップ幅: {self.dt}")
        if not (0 < self.safety_factor < 1):
            raise ValueError(f"無効な安全係数: {self.safety_factor}")

class TimeIntegratorBase(ABC):
    """時間発展スキームの基底クラス"""
    
    def __init__(self, config: TimeIntegrationConfig):
        """
        時間発展スキームの初期化
        
        Args:
            config: 時間発展の設定
        """
        self.config = config
        self.config.validate()
    
    @abstractmethod
    def step(self,
            dfield: ArrayLike,
            t: float,
            field: ArrayLike) -> ArrayLike:
        """
        1ステップの時間発展を実行
        
        Args:
            dfield: 場の時間微分
            t: 現在時刻
            field: 現在の場
            
        Returns:
            時間発展後の場
        """
        pass
    
    def check_stability(self,
                       dfield: ArrayLike,
                       field: ArrayLike,
                       t: float) -> jnp.ndarray:
        """
        時間発展の安定性をチェック
        
        Args:
            dfield: 場の時間微分
            field: 現在の場
            t: 現在時刻
            
        Returns:
            jnp.ndarray: 安定性条件を満たすかどうか (0.0 or 1.0)
        """
        if not self.config.check_stability:
            return jnp.array(1.0)
            
        # フォン・ノイマンの安定性解析に基づく条件
        dt = self.config.dt
        stability_number = dt * jnp.linalg.norm(dfield) / (jnp.linalg.norm(field) + 1e-10)
        
        # JAX対応のため、booleanではなくfloat32で返す
        return jnp.array(stability_number <= self.config.safety_factor, dtype=jnp.float32)
    
    @staticmethod
    def get_order() -> int:
        """
        スキームの精度次数を取得
        
        Returns:
            精度次数
        """
        return 1  # デフォルトは1次精度