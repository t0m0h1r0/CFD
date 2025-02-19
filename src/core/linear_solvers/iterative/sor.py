from typing import Optional, Tuple, Dict, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import LinearSolverBase


class SORSolver(LinearSolverBase):
    """逐次過緩和法（SOR）による線形ソルバー"""
    
    def __init__(
        self, 
        discretization=None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        record_history: bool = False,
        omega: float = 1.5
    ):
        """
        SORソルバーの初期化
        
        Args:
            discretization: 空間離散化スキーム
            max_iterations: 最大反復回数
            tolerance: 収束判定許容誤差
            record_history: 収束履歴の記録フラグ
            omega: 緩和パラメータ（1 < omega < 2）
        """
        super().__init__(
            discretization, 
            max_iterations, 
            tolerance, 
            record_history
        )
        self.omega = omega
    
    @partial(jax.jit, static_argnums=(0,))
    def solve(
        self, 
        operator: ArrayLike, 
        b: ArrayLike, 
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """
        SOR法による線形システムの解法
        
        Args:
            operator: 線形作用素（行列）
            b: 右辺ベクトル
            x0: 初期推定解
        
        Returns:
            解のタプル（解ベクトル、収束情報辞書）
        """
        # 初期値設定
        x = x0 if x0 is not None else jnp.zeros_like(b)
        history = self.create_history_dict()
        
        # 対角成分と対角成分以外の成分を抽出
        diag = jnp.diag(operator)
        
        def sor_step(carry, _):
            x, iteration, residual_norm = carry
            
            # SOR法による更新
            x_new = x.copy()
            for i in range(len(x)):
                # 残差の計算
                r_i = b[i] - operator[i] @ x_new
                
                # SOR更新
                x_new = x_new.at[i].set(
                    x_new[i] + self.omega * r_i / diag[i]
                )
            
            # 残差の計算
            residual = b - operator @ x_new
            new_residual_norm = jnp.linalg.norm(residual)
            
            return (x_new, iteration + 1, new_residual_norm), (x_new, new_residual_norm)
        
        # スキャンを使用した反復計算
        init_state = (x, 0, jnp.linalg.norm(b - operator @ x))
        (x, n_iter, residuals), (solutions, individual_residuals) = jax.lax.scan(
            sor_step, init_state, None, length=self.max_iterations
        )
        
        # 収束履歴の更新
        history.update({
            'converged': float(residuals[-1]) < self.tolerance,
            'iterations': int(n_iter),
            'final_residual': float(residuals[-1])
        })
        
        # 履歴記録が有効な場合
        if self.record_history:
            history['residual_history'] = individual_residuals
        
        return x, history
    
    def optimize_relaxation_parameter(
        self, 
        operator: ArrayLike, 
        b: ArrayLike, 
        x0: Optional[ArrayLike] = None,
        omega_range: Tuple[float, float] = (1.0, 2.0),
        n_points: int = 10
    ) -> float:
        """
        最適な緩和パラメータを探索
        
        Args:
            operator: 線形作用素
            b: 右辺ベクトル
            x0: 初期推定解
            omega_range: 探索する緩和パラメータの範囲
            n_points: 探索点数
        
        Returns:
            最適な緩和パラメータ
        """
        def test_omega(omega):
            solver = SORSolver(
                omega=float(omega),
                max_iterations=self.max_iterations,
                tolerance=self.tolerance
            )
            _, history = solver.solve(operator, b, x0)
            return history['iterations']
        
        # 異なるomegaでテスト
        omegas = jnp.linspace(omega_range[0], omega_range[1], n_points)
        iterations = jnp.array([test_omega(omega) for omega in omegas])
        
        # 最小反復回数のomegaを返す
        return float(omegas[jnp.argmin(iterations)])