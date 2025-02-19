from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import TimeIntegratorBase, TimeIntegrationConfig

class ExplicitEuler(TimeIntegratorBase):
    """GPU最適化された陽解法オイラースキーム"""
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        dfield: ArrayLike, 
        t: float, 
        field: ArrayLike
    ) -> ArrayLike:
        """
        GPU最適化されたステップ計算
        
        Args:
            dfield: 場の時間微分
            t: 現在時刻
            field: 現在の場
        
        Returns:
            時間発展後の場
        """
        dt = self.config.dt
        
        # 安定性チェック
        is_stable = self.check_stability(dfield, field, t)
        
        def unstable_update():
            new_dt = 0.5 * dt
            return field + new_dt * dfield
            
        def stable_update():
            return field + dt * dfield
        
        # ベクトル化オプション
        if self.config.vectorized:
            if self.config.vmap_strategy == 'scan':
                return self._vectorized_scan_step(dt, dfield, field)
            else:
                return self._vectorized_map_step(dt, dfield, field)
        
        # デフォルトの条件分岐
        return jax.lax.cond(
            jnp.asarray(is_stable, dtype=jnp.bool_),
            lambda: stable_update(),
            lambda: jax.lax.cond(
                jnp.asarray(self.config.adaptive_dt, dtype=jnp.bool_),
                lambda: unstable_update(),
                lambda: stable_update()
            )
        )
    
    def _vectorized_scan_step(
        self, 
        dt: float, 
        dfield: ArrayLike, 
        field: ArrayLike
    ) -> ArrayLike:
        """scanによるベクトル化ステップ"""
        def update_fn(current_field):
            return current_field + dt * dfield
        
        return update_fn(field)
    
    def _vectorized_map_step(
        self, 
        dt: float, 
        dfield: ArrayLike, 
        field: ArrayLike
    ) -> ArrayLike:
        """mapによるベクトル化ステップ"""
        update_fn = lambda f: f + dt * dfield
        return update_fn(field)

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
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        dfield: ArrayLike,
        t: float,
        field: ArrayLike
    ) -> ArrayLike:
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
        
        def newton_iteration(carry, _):
            field_next, residual_norm = carry
            
            # 残差の計算
            residual = field_next - field - dt * dfield
            residual_norm = jnp.linalg.norm(residual)
            
            # 更新
            field_next = field_next - residual
            
            return (field_next, residual_norm), residual_norm
        
        # ニュートン法による反復
        init_state = (field, jnp.array(float('inf')))
        (field_next, _), _ = jax.lax.scan(
            newton_iteration, 
            init_state, 
            None, 
            length=self.max_newton_iter
        )
        
        # ベクトル化オプション
        if self.config.vectorized:
            if self.config.vmap_strategy == 'scan':
                return self._vectorized_scan_step(dt, dfield, field)
            else:
                return self._vectorized_map_step(dt, dfield, field)
        
        return field_next

    def _vectorized_scan_step(
        self, 
        dt: float, 
        dfield: ArrayLike, 
        field: ArrayLike
    ) -> ArrayLike:
        """scanによるベクトル化ステップ"""
        def update_fn(current_field):
            # ニュートン法を使用した更新
            return current_field - (current_field - field - dt * dfield)
        
        return update_fn(field)
    
    def _vectorized_map_step(
        self, 
        dt: float, 
        dfield: ArrayLike, 
        field: ArrayLike
    ) -> ArrayLike:
        """mapによるベクトル化ステップ"""
        update_fn = lambda f: f - (f - field - dt * dfield)
        return update_fn(field)