import jax.numpy as jnp
from jax import jit
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
from functools import partial


@dataclass
class GridConfig:
    """グリッド設定を保持するデータクラス"""

    n_points: int  # グリッド点の数
    h: float  # グリッド幅


class BlockMatrixBuilder(ABC):
    """ブロック行列生成の抽象基底クラス"""

    @abstractmethod
    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """ブロック行列を生成する抽象メソッド"""
        pass


class LeftHandBlockBuilder(BlockMatrixBuilder):
    """左辺のブロック行列を生成するクラス"""

    def _build_interior_blocks(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """内部点のブロック行列A, B, Cを生成
        
        Returns:
            A: 左側のブロック行列
            B: 中央のブロック行列
            C: 右側のブロック行列
        """
        # 左ブロック行列 A
        A = jnp.array([
            # f',    f'',   f'''
            [ 19/32,  1/8,   1/96],  # 左側ブロックの1行目
            [-29/16, -5/16, -1/48],  # 左側ブロックの2行目
            [-105/16,-15/8, -3/16]   # 左側ブロックの3行目
        ])

        # 中央ブロック行列 B - 単位行列
        B = jnp.eye(3)

        # 右ブロック行列 C - Aに対して反対称的な構造
        C = jnp.array([
            # f',    f'',    f'''
            [ 19/32, -1/8,   1/96],  # 右側ブロックの1行目
            [ 29/16, -5/16,  1/48],  # 右側ブロックの2行目
            [-105/16, 15/8, -3/16]   # 右側ブロックの3行目
        ])

        return A, B, C

    def _build_boundary_blocks(self) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """境界点のブロック行列を生成
        
        Returns:
            B0: 左境界の主ブロック
            C0: 左境界の第2ブロック
            D0: 左境界の第3ブロック
            ZR: 右境界の第1ブロック
            AR: 右境界の第2ブロック
            BR: 右境界の主ブロック
        """
        # 左境界の行列群
        B0 = jnp.array([
            # f',  f'',  f'''
            [311,  90,    8],  # Pattern 3: 高次の境界条件
            [ 34,  10,  4/3],  # Pattern 2: 中間の境界条件
            [  0,  16,    3]   # Pattern 1: 基本の境界条件
        ])

        C0 = jnp.array([
            # f',  f'',  f'''
            [144,   0,    0],  # Pattern 3
            [ 16,   0,    0],  # Pattern 2
            [-140, 44, 31/3]   # Pattern 1
        ])

        D0 = jnp.array([
            # f',  f'',  f'''
            [ -3,   0,    0],  # Pattern 3
            [  0,   0,    0],  # Pattern 2
            [  0,   0,    0]   # Pattern 1
        ])

        # 右境界の行列群 - 左境界に対して対称的な構造
        BR = jnp.array([
            # f',   f'',   f'''
            [-311,   90,   -8],  # Pattern 3
            [ -34,   10, -4/3],  # Pattern 2
            [   0,   16,   -3]   # Pattern 1
        ])

        ZR = jnp.array([
            # f',  f'',   f'''
            [-144,   0,    0],  # Pattern 3
            [ -16,   0,    0],  # Pattern 2
            [ 140,  44, -31/3]  # Pattern 1
        ])

        AR = jnp.array([
            # f',  f'',  f'''
            [  3,   0,    0],  # Pattern 3
            [  0,   0,    0],  # Pattern 2
            [  0,   0,    0]   # Pattern 1
        ])

        return B0, C0, D0, ZR, AR, BR

    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """左辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        A, B, C = self._build_interior_blocks()
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks()

        # スケーリング行列の定義
        S = jnp.array([[h, h**2, h**3]])

        # スケーリングを適用
        A = A * S
        B = B * S
        C = C * S
        B0 = B0 * S
        C0 = C0 * S
        D0 = D0 * S
        ZR = ZR * S
        AR = AR * S
        BR = BR * S

        matrix_size = 3 * n
        L = jnp.zeros((matrix_size, matrix_size))

        # 左境界条件を設定
        L = L.at[0:3, 0:3].set(B0)
        L = L.at[0:3, 3:6].set(C0)
        L = L.at[0:3, 6:9].set(D0)

        # 内部点を設定
        for i in range(1, n - 1):
            row_start = 3 * i
            L = L.at[row_start : row_start + 3, row_start - 3 : row_start].set(A)
            L = L.at[row_start : row_start + 3, row_start : row_start + 3].set(B)
            L = L.at[row_start : row_start + 3, row_start + 3 : row_start + 6].set(C)

        # 右境界条件を設定
        row_start = 3 * (n - 1)
        L = L.at[row_start : row_start + 3, row_start - 6 : row_start - 3].set(ZR)
        L = L.at[row_start : row_start + 3, row_start - 3 : row_start].set(AR)
        L = L.at[row_start : row_start + 3, row_start : row_start + 3].set(BR)

        return L



class RightHandBlockBuilder(BlockMatrixBuilder):
    """右辺のブロック行列を生成するクラス"""

    def _build_interior_block(self) -> jnp.ndarray:
        """右辺のブロック行列Kを生成"""
        K = jnp.array([
            # 左点,  中点,  右点
            [-35/32,    0,  35/32],  # 1階導関数の係数
            [     4,   -8,      4],  # 2階導関数の係数
            [105/16,    0, -105/16]  # 3階導関数の係数
        ])

        return K

    def _build_boundary_blocks(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """境界点のブロック行列を生成
        
        Returns:
            K0: 左境界用の行列
            KR: 右境界用の行列（K0と対称的な構造）
        """
        # 左境界用の行列
        K0 = jnp.array([
            # 左点,  中点,  右点
            [-450,    448,     2],  # 1階導関数の係数
            [ -49,     48,     1],  # 2階導関数の係数
            [ 130,   -120,   -10]   # 3階導関数の係数
        ])

        # 右境界用の行列 - K0と対称的なパターン
        KR = jnp.array([
            # 左点,  中点,   右点
            [    2,   448,  -450],  # 1階導関数の係数
            [    1,    48,   -49],  # 2階導関数の係数
            [  -10,  -120,   130]   # 3階導関数の係数
        ])

        return K0, KR
    
    def build_block(self, grid_config: GridConfig) -> jnp.ndarray:
        """右辺のブロック行列全体を生成"""
        n, h = grid_config.n_points, grid_config.h
        K_interior = self._build_interior_block()
        K0, KR = self._build_boundary_blocks()

        matrix_size = 3 * n
        vector_size = n
        K = jnp.zeros((matrix_size, vector_size))

        # スケーリング行列は不要 - 右辺の行列はスケーリングしない

        # 左境界条件を設定
        K = K.at[0:3, 0:3].set(K0)

        # 内部点を設定
        for i in range(1, n - 1):
            row_start = 3 * i
            K = K.at[row_start : row_start + 3, i - 1 : i + 2].set(K_interior)

        # 右境界条件を設定
        row_start = 3 * (n - 1)
        K = K.at[row_start : row_start + 3, n - 3 : n].set(KR)

        return K

from jax import jit
import jax.numpy as jnp
from jax.scipy import linalg
from functools import partial

class CCDSolver:
    """CCD法による導関数計算ソルバー"""

    def __init__(self, grid_config: GridConfig):
        self.grid_config = grid_config
        self.left_builder = LeftHandBlockBuilder()
        self.right_builder = RightHandBlockBuilder()
        self._initialize_solver()

    def _initialize_solver(self):
        """ソルバーの初期化: スケーリングされた左辺行列と右辺行列を準備"""
        # 左辺行列と右辺行列の生成
        L = self.left_builder.build_block(self.grid_config)
        K = self.right_builder.build_block(self.grid_config)

        # スケーリング行列の計算
        D = jnp.diag(1.0 / jnp.sqrt(jnp.abs(jnp.diag(L))))
        
        # スケーリングされた行列を保存
        self.L_scaled = D @ L @ D
        self.K_scaled = D @ K
        self.D = D

    @partial(jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> jnp.ndarray:
        """導関数を計算

        Args:
            f: 関数値ベクトル (n,)

        Returns:
            X: 導関数ベクトル (3n,) - [f'_0, f''_0, f'''_0, f'_1, f''_1, f'''_1, ...]
        """
        # K_scaled @ f を計算
        rhs = self.K_scaled @ f
        
        # L_scaled @ X = rhs を解く
        X_scaled = linalg.solve(self.L_scaled, rhs)
        
        # スケーリングを戻す
        return self.D @ X_scaled


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
        print(f"  f'  = {derivatives[3 * i]:.6f}")
        print(f"  f'' = {derivatives[3 * i + 1]:.6f}")
        print(f"  f'''= {derivatives[3 * i + 2]:.6f}")


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
                    name="Zero",
                    f=lambda x: 0.0,  # すでに両端でゼロ
                    df=lambda x: 0.0,
                    d2f=lambda x: 0.0,
                    d3f=lambda x: 0.0
                ),
                TestFunction(
                    name="QuadPoly",
                    f=lambda x: (1 - x**2),  # シンプルな2次関数
                    df=lambda x: -2*x,
                    d2f=lambda x: -2,
                    d3f=lambda x: 0
                ),
                TestFunction(
                    name="CubicPoly",
                    f=lambda x: (1 - x)*(1 + x)*(x + 0.5),  # 3次関数
                    df=lambda x: -(2*x)*(x + 0.5) + (1 - x**2),
                    d2f=lambda x: -2*(x + 0.5) - 4*x,
                    d3f=lambda x: -6
                ),
                TestFunction(
                    name="Sine",
                    f=lambda x: jnp.sin(jnp.pi*x),  # 両端でゼロ
                    df=lambda x: jnp.pi*jnp.cos(jnp.pi*x),
                    d2f=lambda x: -(jnp.pi**2)*jnp.sin(jnp.pi*x),
                    d3f=lambda x: -(jnp.pi**3)*jnp.cos(jnp.pi*x)
                ),
                TestFunction(
                    name="Cosine",
                    f=lambda x: jnp.cos(2*jnp.pi*x),  # 平行移動で両端でゼロ
                    df=lambda x: -2*jnp.pi*jnp.sin(2*jnp.pi*x),
                    d2f=lambda x: -4*(jnp.pi**2)*jnp.cos(2*jnp.pi*x),
                    d3f=lambda x: 8*jnp.pi**3*jnp.sin(2*jnp.pi*x)
                ),
                TestFunction(
                    name="ExpMod",
                    f=lambda x: jnp.exp(-x**2) - jnp.exp(-1),  # 平行移動で両端でゼロ
                    df=lambda x: -2*x*jnp.exp(-x**2),
                    d2f=lambda x: (-2 + 4*x**2)*jnp.exp(-x**2),
                    d3f=lambda x: (12*x - 8*x**3)*jnp.exp(-x**2)
                ),
                TestFunction(
                    name="HigherPoly",
                    f=lambda x: x**4 - x**2,  # 4次関数
                    df=lambda x: 4*x**3 - 2*x,
                    d2f=lambda x: 12*x**2 - 2,
                    d3f=lambda x: 24*x
                ),
                TestFunction(
                    name="CompoundPoly",
                    f=lambda x: x**2 * (1 - x**2),  # 両端でゼロの4次関数
                    df=lambda x: 2*x*(1 - x**2) - 2*x**3,
                    d2f=lambda x: 2*(1 - x**2) - 8*x**2,
                    d3f=lambda x: -12*x
                )
            ]

    def compute_errors(self, test_func: TestFunction) -> Tuple[float, float, float]:
        """各導関数の誤差を計算"""
        n = self.solver.grid_config.n_points
        h = self.solver.grid_config.h
        x_start = self.x_range[0]

        # グリッド点での関数値を計算
        x_points = jnp.array([x_start + i * h for i in range(n)])
        f_values = jnp.array([test_func.f(x) for x in x_points])

        # 数値解の計算
        numerical_derivatives = self.solver.solve(f_values)

        # 解析解の計算
        analytical_derivatives = jnp.zeros(3 * n)
        for i in range(n):
            x = x_points[i]
            analytical_derivatives = analytical_derivatives.at[3 * i].set(
                test_func.df(x)
            )
            analytical_derivatives = analytical_derivatives.at[3 * i + 1].set(
                test_func.d2f(x)
            )
            analytical_derivatives = analytical_derivatives.at[3 * i + 2].set(
                test_func.d3f(x)
            )

        # 誤差の計算 (L2ノルム)
        errors = []
        for i in range(3):
            numerical = numerical_derivatives[i::3]
            analytical = analytical_derivatives[i::3]
            error = jnp.sqrt(jnp.mean((numerical - analytical) ** 2))
            errors.append(float(error))

        return tuple(errors)

    def visualize_results(self, test_func: TestFunction, save_path: str = None):
        """結果の可視化"""
        n = self.solver.grid_config.n_points
        h = self.solver.grid_config.h
        x_start = self.x_range[0]

        # グリッド点での計算
        x_points = jnp.array([x_start + i * h for i in range(n)])
        f_values = jnp.array([test_func.f(x) for x in x_points])
        numerical_derivatives = self.solver.solve(f_values)

        # 高解像度の点での解析解
        x_fine = jnp.linspace(self.x_range[0], self.x_range[1], 200)
        analytical_df = jnp.array([test_func.df(x) for x in x_fine])
        analytical_d2f = jnp.array([test_func.d2f(x) for x in x_fine])
        analytical_d3f = jnp.array([test_func.d3f(x) for x in x_fine])

        # プロット
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Test Results for {test_func.name} Function")

        # 元関数
        axes[0, 0].plot(x_fine, [test_func.f(x) for x in x_fine], "b-", label="f(x)")
        axes[0, 0].plot(x_points, f_values, color="red", label="Grid Points")
        axes[0, 0].set_title("Original Function")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 1階導関数
        axes[0, 1].plot(x_fine, analytical_df, "b-", label="Analytical")
        axes[0, 1].plot(
            x_points, numerical_derivatives[::3], color="red", label="Numerical"
        )
        axes[0, 1].set_title("First Derivative")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 2階導関数
        axes[1, 0].plot(x_fine, analytical_d2f, "b-", label="Analytical")
        axes[1, 0].plot(
            x_points, numerical_derivatives[1::3], color="red", label="Numerical"
        )
        axes[1, 0].set_title("Second Derivative")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 3階導関数
        axes[1, 1].plot(x_fine, analytical_d3f, "b-", label="Analytical")
        axes[1, 1].plot(
            x_points, numerical_derivatives[2::3], color="red", label="Numerical"
        )
        axes[1, 1].set_title("Third Derivative")
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
    L = 2.0  # 区間の長さ（-1から1まで）
    grid_config = GridConfig(n_points=n, h=L / (n - 1))
    solver = CCDSolver(grid_config)

    # テスターの初期化
    x_range = (-1.0, 1.0)  # 区間を[-1,1]に変更
    tester = CCDMethodTester(solver, x_range)

    # 各テスト関数に対してテストを実行
    print("Error Analysis Results:")
    print("-" * 60)
    print(f"{'Function':<15} {'1st Der.':<12} {'2nd Der.':<12} {'3rd Der.':<12}")
    print("-" * 60)

    for test_func in tester.test_functions:
        errors = tester.compute_errors(test_func)
        print(
            f"{test_func.name:<15} {errors[0]:<12.2e} {errors[1]:<12.2e} {errors[2]:<12.2e}"
        )

        # 結果の可視化と保存
        tester.visualize_results(test_func, f"{test_func.name.lower()}_results.png")

    print("-" * 60)


class CCDSolverDiagnostics:
    """CCD法ソルバーの診断を行うクラス"""
    
    def __init__(self, grid_config: GridConfig):
        self.grid_config = grid_config
        self.left_builder = LeftHandBlockBuilder()
        self.right_builder = RightHandBlockBuilder()
        
    def check_boundary_blocks(self):
        """境界ブロック行列の確認"""
        n, h = self.grid_config.n_points, self.grid_config.h
        
        # 左境界のブロック行列を取得
        B0, C0, D0, ZR, AR, BR = self.left_builder._build_boundary_blocks()
        
        print("=== 境界ブロック行列の確認 ===")
        print("\n左境界:")
        print("B0 (左端):\n", B0)
        print("\nC0 (左端の次):\n", C0)
        print("\nD0 (左端の次の次):\n", D0)
        
        print("\n右境界:")
        print("BR (右端):\n", BR)
        print("\nAR (右端の手前):\n", AR)
        print("\nZR (右端の手前の手前):\n", ZR)
        
        # 対称性の確認
        print("\n=== 対称性の確認 ===")
        print("B0とBRの対応する要素の比:")
        print(B0 / (-BR))  # 対称なら絶対値が近い値になるはず
        
        print("\nC0とZRの対応する要素の比:")
        print(C0 / (-ZR))  # 対称なら絶対値が近い値になるはず

    def check_full_matrix(self):
        """全体の行列構造の確認"""
        L = self.left_builder.build_block(self.grid_config)
        K = self.right_builder.build_block(self.grid_config)
        
        print("=== 全体行列の確認 ===")
        print("\n左辺行列の条件数:", jnp.linalg.cond(L))
        
        # 左端と右端の3×9ブロックを抽出
        left_boundary = L[:3, :9]
        right_boundary = L[-3:, -9:]
        
        print("\n左端3×9ブロック:\n", left_boundary)
        print("\n右端3×9ブロック:\n", right_boundary)
        
        # 行列の対称性を確認
        print("\n行列の対称性 (L + L.T)の最大絶対値:", 
              jnp.max(jnp.abs(L + L.T)))

    def check_scaling(self):
        """スケーリングの影響確認"""
        L = self.left_builder.build_block(self.grid_config)
        
        # オリジナルのスケーリング行列
        D = jnp.diag(1.0 / jnp.sqrt(jnp.abs(jnp.diag(L))))
        
        print("=== スケーリングの確認 ===")
        print("\nスケーリング係数 (先頭10個):", D.diagonal()[:10])
        print("\nスケーリング係数 (末尾10個):", D.diagonal()[-10:])
        
        # スケーリング後の行列
        L_scaled = D @ L @ D
        print("\nスケーリング後の条件数:", jnp.linalg.cond(L_scaled))
        
        # スケーリング後の境界部分
        print("\nスケーリング後の左端3×9ブロック:\n", L_scaled[:3, :9])
        print("\nスケーリング後の右端3×9ブロック:\n", L_scaled[-3:, -9:])

def run_diagnostics():
    """診断の実行"""
    # グリッド設定
    n = 256
    L = 2.0
    grid_config = GridConfig(n_points=n, h=L / (n - 1))
    
    # 診断の実行
    diagnostics = CCDSolverDiagnostics(grid_config)
    
    print("\n=== 境界ブロックの確認 ===")
    diagnostics.check_boundary_blocks()
    
    print("\n=== 全体行列の確認 ===")
    diagnostics.check_full_matrix()
    
    print("\n=== スケーリングの確認 ===")
    diagnostics.check_scaling()


if __name__ == "__main__":
    run_diagnostics()
    #run_tests()