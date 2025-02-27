import jax.numpy as jnp
from typing import Tuple, List, Optional
from functools import partial
import jax

from ccd_core import GridConfig, LeftHandBlockBuilder


class CCDSolver:
    """結合コンパクト差分ソルバー"""
    
    def __init__(self, grid_config: GridConfig, coeffs: Optional[List[float]] = None):
        """
        CCDソルバーの初期化
        
        Args:
            grid_config: グリッド設定
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用 (f = psi)
        """
        self.grid_config = grid_config
        self.left_hand_builder = LeftHandBlockBuilder()
        
        # 係数を設定
        self.coeffs = coeffs if coeffs is not None else [1.0, 0.0, 0.0, 0.0]
        
        # 左辺行列を係数を含めて構築
        self.L = self.left_hand_builder.build_block(grid_config, self.coeffs)
        
    def _build_right_hand_vector(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        関数値fを組み込んだ右辺ベクトルを生成
        
        Args:
            f: グリッド点での関数値
            
        Returns:
            パターン[f[0],0,0,0,f[1],0,0,0,...]の右辺ベクトル
        """
        n = self.grid_config.n_points
        depth = 4
        
        # 右辺ベクトルを初期化
        K = jnp.zeros(n * depth)
        
        # パターン[f[0],0,0,0,f[1],0,0,0,...]でベクトルを設定
        for i in range(n):
            K = K.at[i * depth].set(f[i])
            # 他の3つの要素はゼロのまま
        
        return K
        
    def _extract_derivatives(self, solution: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        解から関数値と各階微分を抽出
        
        Args:
            solution: 方程式系の解
            
        Returns:
            (関数値, 1階微分, 2階微分, 3階微分)のタプル
        """
        n = self.grid_config.n_points
        
        # 解ベクトルから各成分を抽出
        f = jnp.array([solution[i * 4] for i in range(n)])
        f_prime = jnp.array([solution[i * 4 + 1] for i in range(n)])
        f_second = jnp.array([solution[i * 4 + 2] for i in range(n)])
        f_third = jnp.array([solution[i * 4 + 3] for i in range(n)])
        
        return f, f_prime, f_second, f_third
        
    @partial(jax.jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        関数値fに対するCCD方程式系を解き、ψとその各階微分を返す
        
        Args:
            f: グリッド点での関数値
            
        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # 右辺ベクトルを構築（パターン[f[0],0,0,0,f[1],0,0,0,...]）
        K = self._build_right_hand_vector(f)
        
        # 線形方程式系 L * u = K を解く
        solution = jnp.linalg.solve(self.L, K)
        
        # 解から関数値と各階微分を抽出して返す
        return self._extract_derivatives(solution)