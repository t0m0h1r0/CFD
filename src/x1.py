import jax
import jax.numpy as jnp
from jax import jit
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional
from functools import partial
from jax.experimental import sparse

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
        
        KR = (1/h) * jnp.array([
            [1/2, -4, 7/2],
            [3, -12, 9],
            [1, -3, 2]
        ])
        
        return K0, KR

    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """右辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        K_interior = self._build_interior_block(h)
        K0, KR = self._build_boundary_blocks(h)
        
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
        K = K.at[row_start:row_start+3, n-3:n].set(KR)
        
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
        
    @partial(jit, static_argnums=(0,))
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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from dataclasses import dataclass

@dataclass
class TestFunction:
    """テスト関数とその導関数を保持するデータクラス"""
    name: str
    f: Callable[[float], float]
    df: Callable[[float], float]
    d2f: Callable[[float], float]
    d3f: Callable[[float], float]

class CCDMethodTester:
    """CCD法のテストを行うクラス"""
    def __init__(self, solver: CCDSolver, x_range: Tuple[float, float]):
        self.solver = solver
        self.x_range = x_range
        self._initialize_test_functions()
        
    def _initialize_test_functions(self):
        """テスト関数群の初期化"""
        self.test_functions = [
            TestFunction(
                name="Cubic",
                f=lambda x: x**3 - 2*x**2 + 3*x - 1,
                df=lambda x: 3*x**2 - 4*x + 3,
                d2f=lambda x: 6*x - 4,
                d3f=lambda x: 6
            ),
            TestFunction(
                name="Exponential",
                f=lambda x: jnp.exp(x),
                df=lambda x: jnp.exp(x),
                d2f=lambda x: jnp.exp(x),
                d3f=lambda x: jnp.exp(x)
            ),
            TestFunction(
                name="Sine",
                f=lambda x: jnp.sin(jnp.pi*x),
                df=lambda x: jnp.pi * jnp.cos(jnp.pi*x),
                d2f=lambda x: -jnp.pi**2 * jnp.sin(jnp.pi*x),
                d3f=lambda x: -jnp.pi**3 * jnp.cos(jnp.pi*x)
            ),
            TestFunction(
                name="Complex",
                f=lambda x: jnp.exp(-x**2) * jnp.sin(2*x),
                df=lambda x: jnp.exp(-x**2) * (2*jnp.cos(2*x) - 2*x*jnp.sin(2*x)),
                d2f=lambda x: jnp.exp(-x**2) * ((-4+4*x**2)*jnp.sin(2*x) - 8*x*jnp.cos(2*x)),
                d3f=lambda x: jnp.exp(-x**2) * ((-12*x+4*x**3)*jnp.sin(2*x) + (-12+8*x**2)*jnp.cos(2*x))
            )
        ]
    
    def compute_errors(self, test_func: TestFunction) -> Tuple[float, float, float]:
        """各導関数の誤差を計算"""
        n = self.solver.grid_config.n_points
        h = self.solver.grid_config.h
        x_start = self.x_range[0]
        
        # グリッド点での関数値を計算
        x_points = jnp.array([x_start + i*h for i in range(n)])
        f_values = jnp.array([test_func.f(x) for x in x_points])
        
        # 数値解の計算
        numerical_derivatives = self.solver.solve(f_values)
        
        # 解析解の計算
        analytical_derivatives = jnp.zeros(3*n)
        for i in range(n):
            x = x_points[i]
            analytical_derivatives = analytical_derivatives.at[3*i].set(test_func.df(x))
            analytical_derivatives = analytical_derivatives.at[3*i+1].set(test_func.d2f(x))
            analytical_derivatives = analytical_derivatives.at[3*i+2].set(test_func.d3f(x))
        
        # 誤差の計算 (L2ノルム)
        errors = []
        for i in range(3):
            numerical = numerical_derivatives[i::3]
            analytical = analytical_derivatives[i::3]
            error = jnp.sqrt(jnp.mean((numerical - analytical)**2))
            errors.append(float(error))
            
        return tuple(errors)
    
    def visualize_results(self, test_func: TestFunction, save_path: str = None):
        """結果の可視化"""
        n = self.solver.grid_config.n_points
        h = self.solver.grid_config.h
        x_start = self.x_range[0]
        
        # グリッド点での計算
        x_points = jnp.array([x_start + i*h for i in range(n)])
        f_values = jnp.array([test_func.f(x) for x in x_points])
        numerical_derivatives = self.solver.solve(f_values)
        
        # 高解像度の点での解析解
        x_fine = jnp.linspace(self.x_range[0], self.x_range[1], 200)
        analytical_df = jnp.array([test_func.df(x) for x in x_fine])
        analytical_d2f = jnp.array([test_func.d2f(x) for x in x_fine])
        analytical_d3f = jnp.array([test_func.d3f(x) for x in x_fine])
        
        # プロット
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Test Results for {test_func.name} Function')
        
        # 元関数
        axes[0, 0].plot(x_fine, [test_func.f(x) for x in x_fine], 'b-', label='f(x)')
        axes[0, 0].plot(x_points, f_values, color='red', label='Grid Points')
        axes[0, 0].set_title('Original Function')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 1階導関数
        axes[0, 1].plot(x_fine, analytical_df, 'b-', label='Analytical')
        axes[0, 1].plot(x_points, numerical_derivatives[::3], color='red', label='Numerical')
        axes[0, 1].set_title('First Derivative')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 2階導関数
        axes[1, 0].plot(x_fine, analytical_d2f, 'b-', label='Analytical')
        axes[1, 0].plot(x_points, numerical_derivatives[1::3], color='red', label='Numerical')
        axes[1, 0].set_title('Second Derivative')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 3階導関数
        axes[1, 1].plot(x_fine, analytical_d3f, 'b-', label='Analytical')
        axes[1, 1].plot(x_points, numerical_derivatives[2::3], color='red', label='Numerical')
        axes[1, 1].set_title('Third Derivative')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

def run_tests():
    """テストの実行"""
    # グリッド設定
    n = 256
    L = 2.0
    grid_config = GridConfig(n_points=n, h=L/(n-1))
    solver = CCDSolver(grid_config)
    
    # テスターの初期化
    x_range = (0.0, L)
    tester = CCDMethodTester(solver, x_range)
    
    # 各テスト関数に対してテストを実行
    print("Error Analysis Results:")
    print("-" * 60)
    print(f"{'Function':<12} {'1st Der.':<12} {'2nd Der.':<12} {'3rd Der.':<12}")
    print("-" * 60)
    
    for test_func in tester.test_functions:
        errors = tester.compute_errors(test_func)
        print(f"{test_func.name:<12} {errors[0]:<12.2e} {errors[1]:<12.2e} {errors[2]:<12.2e}")
        
        # 結果の可視化と保存
        tester.visualize_results(test_func, f"{test_func.name.lower()}_results.png")
    
    print("-" * 60)

if __name__ == "__main__":
    run_tests()