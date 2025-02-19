from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

@dataclass
class TimeIntegrationConfig:
    """時間発展の設定 - GPU最適化版"""
    dt: float
    check_stability: bool = True
    adaptive_dt: bool = False
    safety_factor: float = 0.9
    vectorized: bool = True  # ベクトル化フラグ追加
    vmap_strategy: str = 'scan'  # scan or map
    
    def validate(self):
        """設定の拡張バリデーション"""
        if self.dt <= 0:
            raise ValueError(f"無効な時間ステップ幅: {self.dt}")
        if not (0 < self.safety_factor < 1):
            raise ValueError(f"無効な安全係数: {self.safety_factor}")
        
        if self.vmap_strategy not in ['scan', 'map']:
            raise ValueError("Invalid vmap_strategy. Choose 'scan' or 'map'")

class TimeIntegratorBase(ABC):
    """GPU最適化された時間発展スキーム基底クラス"""
    
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
    def vectorize_step(
        integrator: Callable, 
        batch_dims: Union[int, Tuple[int, ...]] = 0
    ):
        """ベクトル化のためのラッパー"""
        return jax.vmap(integrator, in_axes=batch_dims, out_axes=batch_dims)
    
    @staticmethod
    def scan_vectorize_step(
        integrator: Callable, 
        batch_initial_state: ArrayLike,
        time_steps: ArrayLike
    ):
        """scanによるベクトル化"""
        def scan_step(state, time_step):
            # 時間発展のための関数を適切に定義
            next_state = integrator(state, time_step)
            return next_state, next_state
        
        return jax.lax.scan(scan_step, batch_initial_state, time_steps)
    
    @staticmethod
    def get_order() -> int:
        """
        スキームの精度次数を取得
        
        Returns:
            精度次数
        """
        return 1  # デフォルトは1次精度