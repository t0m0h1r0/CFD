import jax.numpy as jnp
from typing import Tuple
from functools import partial
import jax

from ccd_core import GridConfig, LeftHandBlockBuilder


class CCDSolver:
    """結合コンパクト差分ソルバー"""
    
    def __init__(self, grid_config: GridConfig):
        """
        CCDソルバーの初期化
        
        Args:
            grid_config: グリッド設定
        """
        self.grid_config = grid_config
        self.left_hand_builder = LeftHandBlockBuilder()
        
        # 左辺行列を事前に構築（不変）
        self.L = self.left_hand_builder.build_block(grid_config)
        
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
        関数値fに対するCCD方程式系を解き、関数値と各階微分を返す
        
        Args:
            f: グリッド点での関数値
            
        Returns:
            (関数値, 1階微分, 2階微分, 3階微分)のタプル
        """
        # 右辺ベクトルを構築（パターン[f[0],0,0,0,f[1],0,0,0,...]）
        K = self._build_right_hand_vector(f)
        
        # 線形方程式系 L * u = K を解く
        solution = jnp.linalg.solve(self.L, K)
        
        # 解から関数値と各階微分を抽出して返す
        return self._extract_derivatives(solution)