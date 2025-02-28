import jax.numpy as jnp
import jax
from functools import partial
from typing import Tuple, List, Optional
import jax.scipy.sparse.linalg as jspl

from ccd_core import GridConfig, LeftHandBlockBuilder



class CCDSolver:
    """結合コンパクト差分ソルバー（JAX最適化版）"""
    
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
        
        # 行列の特性を分析して最適なソルバーを選択
        self._analyze_matrix()
        
    def _analyze_matrix(self):
        """行列の特性を分析し、最適なソルバーを選択"""
        # この行列は非常に特殊な構造を持っているため、
        # 直接解法と反復解法の両方をテストして最適化
        
        # 行列のサイズを取得
        self.matrix_size = self.L.shape[0]
        
        # 現時点では直接法を使用
        # (将来的にはサイズや条件数に応じて最適な手法を選択する)
        self.use_iterative = False
        
        # バンド行列としての特性を活かした最適化は将来実装
        self.band_optimized = False
    
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
        
        # 右辺ベクトルを効率的に生成
        K = jnp.zeros(n * depth)
        
        # 全てのインデックスを一度に更新
        indices = jnp.arange(0, n * depth, depth)
        K = K.at[indices].set(f)
        
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
        
        # 効率的な抽出方法
        indices0 = jnp.arange(0, n * 4, 4)
        indices1 = indices0 + 1
        indices2 = indices0 + 2
        indices3 = indices0 + 3
        
        f = solution[indices0]
        f_prime = solution[indices1]
        f_second = solution[indices2]
        f_third = solution[indices3]
        
        return f, f_prime, f_second, f_third
    
    def _solve_direct(self, K: jnp.ndarray) -> jnp.ndarray:
        """直接法によるソルバー"""
        return jnp.linalg.solve(self.L, K)
    
    def _solve_iterative(self, K: jnp.ndarray) -> jnp.ndarray:
        """反復法によるソルバー"""
        # JAXのGMRESを使用
        solution, info = jspl.gmres(
            self.L, K, 
            tol=1e-10, 
            atol=1e-10,
            restart=50,
            maxiter=1000
        )
        return solution
        
    @partial(jax.jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        関数値fに対するCCD方程式系を解き、ψとその各階微分を返す
        
        Args:
            f: グリッド点での関数値
            
        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # 右辺ベクトルを構築
        K = self._build_right_hand_vector(f)
        
        # 選択したソルバーで方程式を解く
        if self.use_iterative:
            solution = self._solve_iterative(K)
        else:
            solution = self._solve_direct(K)
        
        # 解から関数値と各階微分を抽出して返す
        return self._extract_derivatives(solution)


# ベクトル化対応版のCCDSolverクラス
class VectorizedCCDSolver(CCDSolver):
    """複数の関数に同時に適用可能なCCDソルバー"""

    @partial(jax.jit, static_argnums=(0,))
    def solve_batch(self, fs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        複数の関数値セットに対してCCD方程式系を一度に解く
        
        Args:
            fs: 関数値の配列 (batch_size, n_points)
            
        Returns:
            (ψs, ψ's, ψ''s, ψ'''s) 各要素は (batch_size, n_points) の形状
        """
        # バッチサイズとグリッド点数を取得
        batch_size, n_points = fs.shape
        
        # 各バッチについて個別に計算する関数を定義
        def solve_one(f):
            return self.solve(f)
        
        # バッチ全体に対して適用 (vmap)
        return jax.vmap(solve_one)(fs)