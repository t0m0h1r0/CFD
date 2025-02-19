from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from dataclasses import dataclass

from .base import TimeIntegratorBase, TimeIntegrationConfig

@dataclass
class ButcherTableau:
    """ルンゲクッタ法のブッチャー表"""
    a: jnp.ndarray  # ルンゲクッタ行列
    b: jnp.ndarray  # 重み係数
    c: jnp.ndarray  # 時間配分係数
    order: int      # 精度次数
    
    @staticmethod
    def rk4() -> 'ButcherTableau':
        """4次のルンゲクッタ法の係数を生成"""
        a = jnp.array([
            [0., 0., 0., 0.],
            [0.5, 0., 0., 0.],
            [0., 0.5, 0., 0.],
            [0., 0., 1., 0.]
        ])
        b = jnp.array([1/6, 1/3, 1/3, 1/6])
        c = jnp.array([0., 0.5, 0.5, 1.])
        return ButcherTableau(a=a, b=b, c=c, order=4)

class RungeKutta4(TimeIntegratorBase):
    """GPU最適化された4次ルンゲクッタ法"""
    
    def __init__(self, config: TimeIntegrationConfig):
        """
        4次ルンゲクッタ法の初期化
        
        Args:
            config: 時間発展の設定
        """
        super().__init__(config)
        self.tableau = ButcherTableau.rk4()
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        stage_derivatives: Tuple[ArrayLike, ...], 
        t: float, 
        field: ArrayLike
    ) -> ArrayLike:
        """
        GPU最適化されたステップ計算
        
        Args:
            stage_derivatives: ステージごとの微分
            t: 現在時刻
            field: 現在の場
        
        Returns:
            時間発展後の場
        """
        dt = self.config.dt
        k1, k2, k3, k4 = stage_derivatives
        
        # 安定性チェック
        is_stable = self.check_stability(k1, field, t)
        
        def unstable_update():
            new_dt = 0.5 * dt
            return field + (new_dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
        def stable_update():
            return field + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # ベクトル化オプション
        if self.config.vectorized:
            if self.config.vmap_strategy == 'scan':
                return self._vectorized_scan_step(dt, k1, k2, k3, k4, field)
            else:
                return self._vectorized_map_step(dt, k1, k2, k3, k4, field)
        
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
        k1: ArrayLike, 
        k2: ArrayLike, 
        k3: ArrayLike, 
        k4: ArrayLike, 
        field: ArrayLike
    ) -> ArrayLike:
        """scanによるベクトル化ステップ"""
        def update_fn(current_field):
            return current_field + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return update_fn(field)
    
    def _vectorized_map_step(
        self, 
        dt: float, 
        k1: ArrayLike, 
        k2: ArrayLike, 
        k3: ArrayLike, 
        k4: ArrayLike, 
        field: ArrayLike
    ) -> ArrayLike:
        """mapによるベクトル化ステップ"""
        update_fn = lambda f: f + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return update_fn(field)
    
    @staticmethod
    def get_order() -> int:
        """精度次数の取得"""
        return 4

class AdaptiveRungeKutta4(RungeKutta4):
    """適応的な時間ステップ制御を行う4次のルンゲクッタ法"""
    
    def __init__(self,
                 config: TimeIntegrationConfig,
                 relative_tolerance: float = 1e-6,
                 absolute_tolerance: float = 1e-8):
        """
        適応的RK4の初期化
        
        Args:
            config: 時間発展の設定
            relative_tolerance: 相対誤差の許容値
            absolute_tolerance: 絶対誤差の許容値
        """
        super().__init__(config)
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
    
    @partial(jax.jit, static_argnums=(0,))
    def estimate_error(self,
                      stage_derivatives: Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
                      field: ArrayLike) -> ArrayLike:
        """
        誤差推定（組み込み法による）
        
        Args:
            stage_derivatives: 4つのステージの時間微分値
            field: 現在の場
            
        Returns:
            推定された局所誤差
        """
        dt = self.config.dt
        k1, k2, k3, k4 = stage_derivatives
        
        # 5次の解との比較による誤差推定（組み込み係数）
        b_star = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        b = jnp.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        
        # 誤差の推定
        error = dt * jnp.abs(
            (b_star[0] - b[0])*k1 + (b_star[2] - b[2])*k2 + 
            (b_star[3] - b[3])*k3 + (b_star[4] - b[4])*k4
        )
        
        return error
    
    @partial(jax.jit, static_argnums=(0,))
    def adjust_timestep(self,
                       error: ArrayLike,
                       field: ArrayLike) -> ArrayLike:
        """
        時間ステップ幅の調整
        
        Args:
            error: 推定された誤差
            field: 現在の場
            
        Returns:
            新しい時間ステップ幅
        """
        dt = self.config.dt
        scale = self.absolute_tolerance + self.relative_tolerance * jnp.abs(field)
        error_ratio = jnp.max(error / scale)
        
        def reduce_dt():
            return jnp.maximum(
                jnp.array(0.1 * dt),
                jnp.array(0.9 * dt * (1/error_ratio)**(1/4))
            )
            
        def increase_dt():
            return jnp.minimum(
                jnp.array(10.0 * dt),
                jnp.array(0.9 * dt * (1/error_ratio)**(1/5))
            )
        
        return jax.lax.cond(
            error_ratio > 1.0,
            reduce_dt,
            increase_dt
        )