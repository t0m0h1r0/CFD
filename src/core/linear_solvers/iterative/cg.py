from typing import Optional, Tuple, Dict, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..base import LinearSolverBase


class ConjugateGradientSolver(LinearSolverBase):
    """共役勾配法による線形ソルバー"""
    
    def __init__(
        self, 
        discretization=None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        record_history: bool = False,
        preconditioner: Optional[Callable[[ArrayLike], ArrayLike]] = None
    ):
        """
        共役勾配法ソルバーの初期化
        
        Args:
            discretization: 空間離散化スキーム
            max_iterations: 最大反復回数
            tolerance: 収束判定許容誤差
            record_history: 収束履歴の記録フラグ
            preconditioner: 前処理関数
        """
        super().__init__(
            discretization, 
            max_iterations, 
            tolerance, 
            record_history
        )
        self.preconditioner = preconditioner or (lambda x: x)
    
    @partial(jax.jit, static_argnums=(0,))
    def solve(
        self, 
        operator: ArrayLike, 
        b: ArrayLike, 
        x0: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict[str, Union[bool, float, list]]]:
        """
        共役勾配法による線形システムの解法
        
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
        
        # 初期残差と前処理付き残差
        r = b - operator @ x
        z = self.preconditioner(r)
        p = z
        rz_old = jnp.sum(r * z)
        
        def cg_step(carry, _):
            """共役勾配法のステップ関数"""
            x, r, p, rz_old = carry
            
            # 行列-ベクトル積
            Ap = operator @ p
            alpha = rz_old / jnp.sum(p * Ap)
            
            # 解と残差の更新
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            
            # 前処理
            z_new = self.preconditioner(r_new)
            rz_new = jnp.sum(r_new * z_new)
            
            # 探索方向の更新
            beta = rz_new / rz_old
            p_new = z_new + beta * p
            
            return (x_new, r_new, p_new, rz_new), jnp.sqrt(rz_new)
        
        # JAX scan を用いた反復計算
        init_carry = (x, r, p, rz_old)
        (x, r, p, rz), residual_norms = jax.lax.scan(
            cg_step, init_carry, None, length=self.max_iterations
        )
        
        # 収束判定と履歴更新
        converged = self.check_convergence(residual_norms[-1], self.max_iterations)
        
        history.update({
            'converged': converged,
            'iterations': self.max_iterations,
            'final_residual': float(residual_norms[-1])
        })
        
        if self.record_history:
            history['residual_history'] = residual_norms
        
        return x, history
    
    @staticmethod
    def diagonal_preconditioner(operator: ArrayLike) -> Callable[[ArrayLike], ArrayLike]:
        """
        対角前処理を生成
        
        Args:
            operator: 線形作用素
        
        Returns:
            対角前処理関数
        """
        diag = jnp.diag(operator)
        
        def preconditioner(x):
            return x / diag
        
        return preconditioner
    
    @staticmethod
    def symmetric_sor_preconditioner(
        operator: ArrayLike, 
        omega: float = 1.5
    ) -> Callable[[ArrayLike], ArrayLike]:
        """
        対称SOR前処理を生成
        
        Args:
            operator: 線形作用素
            omega: 緩和パラメータ
        
        Returns:
            対称SOR前処理関数
        """
        def preconditioner(x):
            # 前向きスイープ
            y = jnp.zeros_like(x)
            for i in range(len(x)):
                y = y.at[i].set(
                    (x[i] - jnp.sum(operator[i] @ y)) / operator[i, i]
                )
            y = omega * y
            
            # 後ろ向きスイープ
            z = jnp.zeros_like(x)
            for i in reversed(range(len(x))):
                z = z.at[i].set(
                    (x[i] - jnp.sum(operator[i] @ z)) / operator[i, i]
                )
            z = omega * z
            
            return y + z - omega * x
        
        return preconditioner