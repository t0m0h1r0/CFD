from typing import Tuple, Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import IterativeSolverBase

class ConjugateGradient(IterativeSolverBase):
    """Matrix-free実装の共役勾配法"""
    
    def __init__(self,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 record_history: bool = False):
        """
        CG法の初期化
        
        Args:
            max_iterations: 最大反復回数
            tolerance: 収束判定閾値
            record_history: 収束履歴を記録するかどうか
        """
        super().__init__(max_iterations, tolerance, record_history)
        
    @partial(jax.jit, static_argnums=(0,))
    def solve(self,
             operator: Callable,
             b: ArrayLike,
             x0: Optional[ArrayLike] = None,
             preconditioner: Optional[Callable] = None) -> Tuple[ArrayLike, dict]:
        """
        線形システムAx = bをCG法で解く
        
        Args:
            operator: 行列ベクトル積を計算する関数 (Ax)
            b: 右辺ベクトル
            x0: 初期推定値 (オプション)
            preconditioner: プリコンディショナ関数 (オプション)
            
        Returns:
            Tuple (解ベクトル, 収束情報)
        """
        # 初期値の設定
        x = x0 if x0 is not None else jnp.zeros_like(b)
        history = self.initialize_history()
        
        # プリコンディショナの設定
        M = preconditioner if preconditioner is not None else lambda x: x
        
        # 初期残差の計算
        r = b - operator(x)
        z = M(r)  # プリコンディショニングされた残差
        p = z
        rz_old = jnp.sum(r * z)
        
        def cg_step(carry, _):
            x, r, p, rz_old = carry
            
            # 探索方向に対する行列ベクトル積
            Ap = operator(p)
            alpha = rz_old / (jnp.sum(p * Ap) + 1e-10)
            
            # 解と残差の更新
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            
            # プリコンディショニングと内積の計算
            z_new = M(r_new)
            rz_new = jnp.sum(r_new * z_new)
            
            # 探索方向の更新
            beta = rz_new / (rz_old + 1e-10)
            p_new = z_new + beta * p
            
            return (x_new, r_new, p_new, rz_new), jnp.sqrt(rz_new)
            
        # 反復計算の実行
        init_carry = (x, r, p, rz_old)
        (x, r, p, rz), residual_norms = jax.lax.scan(
            cg_step, init_carry, None, length=self.max_iterations
        )
        
        # 収束判定と履歴の更新
        converged = self.check_convergence(residual_norms[-1], self.max_iterations)
        history = self.update_history(
            history, residual_norms[-1], self.max_iterations, converged
        )
        
        if self.record_history:
            history['residual_norms'] = residual_norms
            
        return x, history