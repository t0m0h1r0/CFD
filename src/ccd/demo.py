# performance_comparison.py
"""
疎行列最適化の効果を測定するためのデモスクリプト
"""

import cupy as cp
import time
import matplotlib.pyplot as plt
from grid import Grid
from test_functions import TestFunctionFactory
from equation_system import EquationSystem
from solver import CCDSolver
from equation.poisson import PoissonEquation
from equation.boundary import DirichletBoundaryEquation, NeumannBoundaryEquation
from equation.compact_internal import (
    Internal1stDerivativeEquation,
    Internal2ndDerivativeEquation,
    Internal3rdDerivativeEquation,
)
from equation.compact_left_boundary import (
    LeftBoundary1stDerivativeEquation,
    LeftBoundary2ndDerivativeEquation,
    LeftBoundary3rdDerivativeEquation,
)
from equation.compact_right_boundary import (
    RightBoundary1stDerivativeEquation,
    RightBoundary2ndDerivativeEquation,
    RightBoundary3rdDerivativeEquation,
)


def setup_system(grid, test_func):
    """
    方程式システムをセットアップ
    
    Args:
        grid: 計算格子
        test_func: テスト関数
        
    Returns:
        EquationSystem: セットアップされた方程式システム
    """
    system = EquationSystem(grid)
    
    # グリッド情報
    x_min = grid.x_min
    x_max = grid.x_max
    
    # 内部点の方程式
    system.add_interior_equation(PoissonEquation(test_func.d2f))
    system.add_interior_equation(Internal1stDerivativeEquation())
    system.add_interior_equation(Internal2ndDerivativeEquation())
    system.add_interior_equation(Internal3rdDerivativeEquation())
    
    # 左境界の方程式
    system.add_left_boundary_equation(PoissonEquation(test_func.d2f))
    system.add_left_boundary_equation(
        DirichletBoundaryEquation(test_func.f(x_min), is_left=True)
    )
    system.add_left_boundary_equation(
        NeumannBoundaryEquation(test_func.df(x_min), is_left=True)
    )
    system.add_left_boundary_equation(
        LeftBoundary1stDerivativeEquation()
        + LeftBoundary2ndDerivativeEquation()
        + LeftBoundary3rdDerivativeEquation()
    )
    
    # 右境界の方程式
    system.add_right_boundary_equation(PoissonEquation(test_func.d2f))
    system.add_right_boundary_equation(
        DirichletBoundaryEquation(test_func.f(x_max), is_left=False)
    )
    system.add_right_boundary_equation(
        NeumannBoundaryEquation(test_func.df(x_max), is_left=False)
    )
    system.add_right_boundary_equation(
        RightBoundary1stDerivativeEquation()
        + RightBoundary2ndDerivativeEquation()
        + RightBoundary3rdDerivativeEquation()
    )
    
    return system


def run_performance_test(grid_sizes, x_range=(-1.0, 1.0), solvers=None):
    """
    パフォーマンステストを実行
    
    Args:
        grid_sizes: グリッドサイズのリスト
        x_range: x座標の範囲
        solvers: テストするソルバーのリスト (デフォルトは ["dense_direct", "sparse_direct", "sparse_gmres"])
        
    Returns:
        dict: ソルバーごとの実行時間とメモリ使用量
    """
    if solvers is None:
        solvers = ["dense_direct", "sparse_direct", "sparse_gmres"]
    
    # テスト関数
    test_funcs = TestFunctionFactory.create_standard_functions()
    test_func = test_funcs[3]  # Sin関数
    
    results = {
        solver: {
            "grid_sizes": grid_sizes,
            "build_times": [],
            "solve_times": [],
            "total_times": [],
            "memory_usage": [],
            "errors": []
        } for solver in solvers
    }
    
    for n_points in grid_sizes:
        print(f"\n格子点数: {n_points}")
        
        # グリッドの作成
        grid = Grid(n_points, x_range)
        
        for solver_type in solvers:
            print(f"  ソルバー: {solver_type}")
            
            # システム構築
            system = setup_system(grid, test_func)
            
            # ソルバーの作成
            solver = CCDSolver(system, grid)
            
            # ソルバータイプに応じた設定
            if solver_type == "sparse_gmres":
                solver.set_solver(method="gmres", options={
                    "tol": 1e-10,
                    "maxiter": 1000,
                    "restart": 100,
                    "use_preconditioner": True
                })
            elif solver_type == "sparse_direct":
                solver.set_solver(method="direct")
            # dense_directは特に設定不要（デフォルト）
            
            # 行列構築時間の測定
            build_start = time.time()
            A, b = system.build_matrix_system()
            build_end = time.time()
            build_time = build_end - build_start
            
            # 疎性情報の取得
            if solver_type.startswith("sparse"):
                sparsity_info = system.analyze_sparsity()
                memory_usage = sparsity_info["memory_sparse_MB"]
            else:
                memory_usage = (n_points * 4)**2 * 8 / (1024 * 1024)  # 8 bytes per double
            
            # 求解時間の測定
            solve_start = time.time()
            psi, psi_prime, psi_second, psi_third = solver.solve(analyze_before_solve=False)
            solve_end = time.time()
            solve_time = solve_end - solve_start
            
            # 計算された解の精度評価
            x = grid.get_points()
            exact_psi = cp.array([test_func.f(xi) for xi in x])
            error = float(cp.max(cp.abs(psi - exact_psi)))
            
            # 結果を記録
            results[solver_type]["build_times"].append(build_time)
            results[solver_type]["solve_times"].append(solve_time)
            results[solver_type]["total_times"].append(build_time + solve_time)
            results[solver_type]["memory_usage"].append(memory_usage)
            results[solver_type]["errors"].append(error)
            
            print(f"    構築時間: {build_time:.4f}秒")
            print(f"    求解時間: {solve_time:.4f}秒")
            print(f"    合計時間: {build_time + solve_time:.4f}秒")
            print(f"    メモリ使用量: {memory_usage:.2f} MB")
            print(f"    誤差: {error:.6e}")
            
    return results


def plot_performance_results(results, output_file=None):
    """
    パフォーマンス結果をプロット
    
    Args:
        results: run_performance_test()の結果
        output_file: 出力ファイル名（Noneの場合は表示のみ）
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    solver_labels = {
        "dense_direct": "密行列 (直接法)",
        "sparse_direct": "疎行列 (直接法)",
        "sparse_gmres": "疎行列 (GMRES)",
    }
    
    colors = {
        "dense_direct": "blue",
        "sparse_direct": "red",
        "sparse_gmres": "green",
    }
    
    markers = {
        "dense_direct": "o",
        "sparse_direct": "s",
        "sparse_gmres": "^",
    }
    
    # 格子点数のラベル
    x = results[list(results.keys())[0]]["grid_sizes"]
    
    # 合計実行時間のプロット
    ax = axes[0, 0]
    for solver, data in results.items():
        ax.plot(x, data["total_times"], marker=markers[solver], label=solver_labels[solver], color=colors[solver])
    ax.set_xlabel("格子点数")
    ax.set_ylabel("合計実行時間 (秒)")
    ax.set_title("格子サイズに対する合計実行時間")
    ax.grid(True)
    ax.legend()
    
    # メモリ使用量のプロット
    ax = axes[0, 1]
    for solver, data in results.items():
        ax.plot(x, data["memory_usage"], marker=markers[solver], label=solver_labels[solver], color=colors[solver])
    ax.set_xlabel("格子点数")
    ax.set_ylabel("メモリ使用量 (MB)")
    ax.set_title("格子サイズに対するメモリ使用量")
    ax.grid(True)
    ax.legend()
    
    # 誤差のプロット
    ax = axes[1, 0]
    for solver, data in results.items():
        ax.plot(x, data["errors"], marker=markers[solver], label=solver_labels[solver], color=colors[solver])
    ax.set_xlabel("格子点数")
    ax.set_ylabel("最大誤差")
    ax.set_title("格子サイズに対する解の精度")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    
    # 求解時間の比較
    ax = axes[1, 1]
    for solver, data in results.items():
        ax.plot(x, data["solve_times"], marker=markers[solver], label=solver_labels[solver], color=colors[solver])
    ax.set_xlabel("格子点数")
    ax.set_ylabel("求解時間 (秒)")
    ax.set_title("格子サイズに対する求解時間")
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"パフォーマンス結果を {output_file} に保存しました")
    else:
        plt.show()


if __name__ == "__main__":
    # 小〜中規模の問題サイズでテスト
    grid_sizes = [21, 41, 81, 161, 321]
    
    print("パフォーマンス比較を開始...")
    results = run_performance_test(grid_sizes)
    
    print("\n結果をプロットしています...")
    plot_performance_results(results, "results/performance_comparison.png")