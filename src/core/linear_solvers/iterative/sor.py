from typing import Tuple, Optional, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import IterativeSolverBase

class SORSolver(IterativeSolverBase):
    """Matrix-free実装のSOR法"""
    
    def __init__(self,
                 omega: float = 1.5,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 record_history: bool = False):
        """
        SOR法の初期化
        
        Args:
            omega: 緩和係数 (1 < omega < 2)
            max_iterations: 最大反復回数
            tolerance: 収束判定閾値
            record_history: 収束履歴を記録するかどうか
        """
        super().__init__(max_iterations, tolerance, record_history)
        self.omega = omega
        
    @partial(jax.jit, static_argnums=(0,))
    def solve(self,
             operator: Callable,
             b: ArrayLike,
             x0: Optional[ArrayLike] = None,
             preconditioner: Optional[Callable] = None) -> Tuple[ArrayLike, dict]:
        """
        線形システムAx = bをSOR法で解く
        
        Args:
            operator: 行列ベクトル積を計算する関数 (Ax)
            b: 右辺ベクトル
            x0: 初期推定値 (オプション)
            preconditioner: プリコンディショナ関数 (オプション, SORでは通常不使用)
            
        Returns:
            Tuple (解ベクトル, 収束情報)
        """
        # 初期値の設定
        x = x0 if x0 is not None else jnp.zeros_like(b)
        history = self.initialize_history()
        
        # 対角成分の抽出 (行列フリー演算用)
        n = len(b)
        diag = jnp.array([operator(jnp.eye(n)[i])[i] for i in range(n)])
        
        def sor_step(state, _):
            x, residual_norm = state
            
            # 新しい反復の開始
            x_new = x.copy()
            
            # 各成分の更新
            for i in range(n):
                # i番目の単位ベクトル
                e_i = jnp.zeros(n).at[i].set(1.0)
                
                # 残差の計算
                r_i = b[i] - operator(x)[i]
                
                # 解の更新
                x_new = x_new.at[i].set(
                    x[i] + self.omega * r_i / diag[i]
                )
            
            # 新しい残差の計算
            new_residual = b - operator(x_new)
            new_residual_norm = jnp.linalg.norm(new_residual)
            
            return (x_new, new_residual_norm), new_residual_norm
            
        # 初期状態の設定
        init_state = (x, jnp.linalg.norm(b - operator(x)))
        
        # 反復計算の実行
        (x, residual_norm), residual_norms = jax.lax.scan(
            sor_step, init_state, None, length=self.max_iterations
        )
        
        # 収束判定と履歴の更新
        converged = self.check_convergence(residual_norm, self.max_iterations)
        history = self.update_history(
            history, residual_norm, self.max_iterations, converged
        )
        
        if self.record_history:
            history['residual_norms'] = residual_norms
            
        return x, history
    
    def optimize_omega(self,
                      operator: Callable,
                      b: ArrayLike,
                      x0: Optional[ArrayLike] = None,
                      omega_range: Tuple[float, float] = (1.0, 1.9),
                      n_points: int = 10) -> float:
        """
        最適な緩和係数を探索
        
        Args:
            operator: 行列ベクトル積を計算する関数
            b: 右辺ベクトル
            x0: 初期推定値 (オプション)
            omega_range: 探索する緩和係数の範囲
            n_points: 探索点の数
            
        Returns:
            最適な緩和係数
        """
        def test_omega(omega):
            solver = SORSolver(
                omega=omega,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance
            )
            _, history = solver.solve(operator, b, x0)
            return history['iteration_count']
        
        # 異なるomega値でテスト
        omegas = jnp.linspace(omega_range[0], omega_range[1], n_points)
        iterations = jnp.array([test_omega(omega) for omega in omegas])
        
        # 最小反復回数となるomegaを返す
        return omegas[jnp.argmin(iterations)]