import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class GridConfig:
    """グリッド設定を保持するデータクラス"""
    n_points: int  # グリッド点の数
    h: float      # グリッド幅
    
class BlockMatrixBuilder(ABC):
    """ブロック行列生成の抽象基底クラス"""
    @abstractmethod
    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """ブロック行列を生成する抽象メソッド"""
        pass

class LeftHandBlockBuilder(BlockMatrixBuilder):
    """左辺のブロック行列を生成するクラス"""
    def _build_interior_blocks(self, h: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """内部点のブロック行列A, B, Cを生成"""
        A = jnp.array([
            [7/16, h/16, h**2/32],
            [-9/(8*h), -1/8, h/16],
            [1/h**2, 1/(4*h), 0]
        ])
        
        B = jnp.eye(3)
        
        C = jnp.array([
            [7/16, -h/16, -h**2/32],
            [9/(8*h), -1/8, -h/16],
            [-1/h**2, -1/(4*h), 0]
        ])
        
        return A, B, C
    
    def _build_boundary_blocks(self, h: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """境界点のブロック行列を生成"""
        B0 = jnp.array([
            [1, 0, 0],
            [0, h, 0],
            [0, 0, h**2]
        ])
        
        C0 = jnp.array([
            [2, -h/2, h**2/4],
            [-6/h, 5/2, -h/4],
            [1/h**2, -1/(4*h), 0]
        ])
        
        D0 = jnp.array([
            [1/16, -h/16, -h**2/32],
            [9/(16*h), -1/16, h/16],
            [-1/h**2, 1/(4*h), 0]
        ])
        
        BR = jnp.array([
            [1, 0, 0],
            [0, h, 0],
            [0, 0, h**2]
        ])
        
        AR = jnp.array([
            [2, h/2, h**2/4],
            [6/h, 5/2, h/4],
            [-1/h**2, 1/(4*h), 0]
        ])
        
        ZR = jnp.array([
            [1/16, h/16, h**2/32],
            [-9/(16*h), 1/16, -h/16],
            [1/h**2, -1/(4*h), 0]
        ])
        return B0, C0, D0, ZR, AR, BR

    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """左辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        A, B, C = self._build_interior_blocks(h)
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks(h)
        
        # 全体行列のサイズを計算
        matrix_size = 3 * n
        L = jnp.zeros((matrix_size, matrix_size))
        
        # 左境界条件を設定
        L = L.at[0:3, 0:3].set(B0)
        L = L.at[0:3, 3:6].set(C0)
        L = L.at[0:3, 6:9].set(D0)
        
        # 内部点を設定
        for i in range(1, n-1):
            row_start = 3 * i
            L = L.at[row_start:row_start+3, row_start-3:row_start].set(A)
            L = L.at[row_start:row_start+3, row_start:row_start+3].set(B)
            L = L.at[row_start:row_start+3, row_start+3:row_start+6].set(C)
            
        # 右境界条件を設定
        row_start = 3 * (n-1)
        L = L.at[row_start:row_start+3, row_start-6:row_start-3].set(ZR)
        L = L.at[row_start:row_start+3, row_start-3:row_start].set(AR)
        L = L.at[row_start:row_start+3, row_start:row_start+3].set(BR)
        
        return L

class RightHandBlockBuilder(BlockMatrixBuilder):
    """右辺のブロック行列を生成するクラス"""
    def _build_interior_block(self, h: float) -> jnp.ndarray:
        """内部点のブロック行列を生成"""
        return jnp.array([
            [-15/(16*h), 0, 15/(16*h)],
            [3/h**2, -6/h**2, 3/h**2],
            [-1/(2*h**3), 0, 1/(2*h**3)]
        ])

    def _build_boundary_blocks(self, h: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """境界点のブロック行列を生成"""
        K0 = (1/h) * jnp.array([
            [-7/2, 4, -1/2],
            [9, -12, 3],
            [-2, 3, -1]
        ])
        
        Kn = (1/h) * jnp.array([
            [1/2, -4, 7/2],
            [3, -12, 9],
            [1, -3, 2]
        ])
        
        return K0, Kn

    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """右辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        K_interior = self._build_interior_block(h)
        K0, Kn = self._build_boundary_blocks(h)
        
        matrix_size = 3 * n
        vector_size = n
        K = jnp.zeros((matrix_size, vector_size))
        
        # 左境界条件を設定
        K = K.at[0:3, 0:3].set(K0)
        
        # 内部点を設定
        for i in range(1, n-1):
            row_start = 3 * i
            K = K.at[row_start:row_start+3, i-1:i+2].set(K_interior)
            
        # 右境界条件を設定
        row_start = 3 * (n-1)
        K = K.at[row_start:row_start+3, n-3:n].set(Kn)
        
        return K

class CCDSolver:
    """CCD法による導関数計算ソルバー"""
    def __init__(self, grid_config: GridConfig):
        self.grid_config = grid_config
        self.left_builder = LeftHandBlockBuilder()
        self.right_builder = RightHandBlockBuilder()
        self._initialize_solver()
        
    def _initialize_solver(self):
        """ソルバーの初期化: 左辺の逆行列と右辺行列の積を計算"""
        L = self.left_builder.build_block(self.grid_config)
        K = self.right_builder.build_block(self.grid_config)
        
        # 左辺の逆行列を計算
        L_inv = jnp.linalg.inv(L)
        
        # ソルバー行列を計算 (L^{-1}K)
        self.solver_matrix = L_inv @ K
        
    def solve(self, f: jnp.ndarray) -> jnp.ndarray:
        """導関数を計算
        
        Args:
            f: 関数値ベクトル (n,)
            
        Returns:
            X: 導関数ベクトル (3n,) - [f'_0, f''_0, f'''_0, f'_1, f''_1, f'''_1, ...]
        """
        return self.solver_matrix @ f

# 使用例
def example_usage():
    # グリッド設定
    grid_config = GridConfig(n_points=5, h=0.1)
    
    # ソルバーの初期化
    solver = CCDSolver(grid_config)
    
    # テスト用の関数値ベクトル
    f = jnp.array([0.0, 0.1, 0.4, 0.9, 1.6])
    
    # 導関数を計算
    derivatives = solver.solve(f)
    
    # 結果を整形して表示
    n = grid_config.n_points
    for i in range(n):
        print(f"Point {i}:")
        print(f"  f'  = {derivatives[3*i]:.6f}")
        print(f"  f'' = {derivatives[3*i+1]:.6f}")
        print(f"  f'''= {derivatives[3*i+2]:.6f}")

if __name__ == "__main__":
    example_usage()