from functools import partial

import jax
from jax.typing import ArrayLike
import jax.numpy as jnp

from .base import TimeIntegratorBase, TimeIntegrationConfig

class ExplicitEuler(TimeIntegratorBase):
    """前進オイラー法による時間発展"""
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            dfield: ArrayLike,
            t: float,
            field: ArrayLike) -> ArrayLike:
        """
        オイラー法による1ステップの時間発展
        
        Args:
            dfield: 場の時間微分
            t: 現在時刻
            field: 現在の場
            
        Returns:
            時間発展後の場
        """
        dt = self.config.dt
        
        # 安定性チェック
        if self.config.check_stability:
            is_stable = self.check_stability(dfield, field, t)
            if not is_stable and self.config.adaptive_dt:
                # 時間ステップを半分にして再試行
                self.config.dt *= 0.5
                return self.step(dfield, t, field)
        
        # 場の更新
        return field + dt * dfield

class ImplicitEuler(TimeIntegratorBase):
    """後退オイラー法による時間発展"""
    
    def __init__(self,
                 config: TimeIntegrationConfig,
                 newton_tol: float = 1e-6,
                 max_newton_iter: int = 10):
        """
        後退オイラー法の初期化
        
        Args:
            config: 時間発展の設定
            newton_tol: ニュートン法の収束判定閾値
            max_newton_iter: ニュートン法の最大反復回数
        """
        super().__init__(config)
        self.newton_tol = newton_tol
        self.max_newton_iter = max_newton_iter
    
    def step(self,
            dfield: ArrayLike,
            t: float,
            field: ArrayLike) -> ArrayLike:
        """
        後退オイラー法による1ステップの時間発展
        
        Args:
            dfield: 場の時間微分
            t: 現在時刻
            field: 現在の場
            
        Returns:
            時間発展後の場
        """
        dt = self.config.dt
        t_next = t + dt
        
        # ニュートン法による非線形方程式の求解
        field_next = field  # 初期推定値
        
        for _ in range(self.max_newton_iter):
            # 残差の計算
            residual = field_next - field - dt * dfield
            residual_norm = jnp.linalg.norm(residual)
            
            # 収束判定
            if residual_norm < self.newton_tol:
                break
            
            # 更新
            field_next = field_next - residual
        
        return field_next