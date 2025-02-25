from jax import jit
import jax.numpy as jnp
from jax.scipy import linalg
from functools import partial

from ccd_core import GridConfig, LeftHandBlockBuilder, RightHandBlockBuilder


class CCDSolver:
    """CCD法による導関数計算の基本ソルバー（スケーリングなし）"""

    def __init__(self, grid_config: GridConfig):
        self.grid_config = grid_config
        self.left_builder = LeftHandBlockBuilder()
        self.right_builder = RightHandBlockBuilder()
        self._initialize_solver()

    def _initialize_solver(self):
        """ソルバーの初期化 - 行列の構築"""
        self.L = self.left_builder.build_block(self.grid_config)
        self.K = self.right_builder.build_block(self.grid_config)

    @partial(jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> jnp.ndarray:
        """導関数を計算

        Args:
            f: 関数値ベクトル (n,)

        Returns:
            X: 導関数ベクトル (3n,) - [f'_0, f''_0, f'''_0, f'_1, f''_1, f'''_1, ...]
        """
        # 右辺ベクトルを計算
        rhs = self.K @ f
        
        # 連立方程式を解く
        X = linalg.solve(self.L, rhs, assume_a='sym')
        
        return X