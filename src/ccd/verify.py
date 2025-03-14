#!/usr/bin/env python3
"""
CCD Matrix Structure Verification Tool
高精度コンパクト差分法(CCD)の行列構造を可視化・検証するツール
"""

import os
from typing import Dict, Tuple, Optional, Any, Union, List

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from grid import Grid
from equation_system import EquationSystem
from test_functions import TestFunctionFactory
from equation_sets import EquationSet
from solver import CCDSolver1D, CCDSolver2D
from scaling import plugin_manager


def visualize_system(A: Any, b: Any, x: Any, 
                     title: str, output_path: str) -> str:
    """
    システム Ax = b を可視化
    
    Args:
        A: システム行列
        b: 右辺ベクトル
        x: 解ベクトル
        title: 図のタイトル
        output_path: 出力ファイルのパス
        
    Returns:
        生成されたファイルのパス
    """
    # CuPy/SciPy配列をNumPy配列に変換
    def to_numpy(arr):
        if hasattr(arr, 'toarray'):
            return arr.toarray().get() if hasattr(arr, 'get') else arr.toarray()
        elif hasattr(arr, 'get'):
            return arr.get()
        return arr

    # 行列とベクトルを変換・整形
    A_np = to_numpy(A)
    b_np = to_numpy(b).reshape(-1, 1)
    x_np = to_numpy(x).reshape(-1, 1) if x is not None else None
    
    # A、x、bを結合
    if x_np is not None:
        combined = np.hstack([A_np, x_np, b_np])
        column_labels = ["A"] * A_np.shape[1] + ["x"] + ["b"]
    else:
        combined = np.hstack([A_np, b_np])
        column_labels = ["A"] * A_np.shape[1] + ["b"]
    
    # 可視化用の設定
    abs_combined = np.abs(combined)
    non_zero = abs_combined[abs_combined > 0]
    
    if len(non_zero) == 0:
        print(f"警告: システムに非ゼロの要素がありません")
        return ""
    
    # 対数スケールで表示
    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        abs_combined, 
        norm=LogNorm(vmin=non_zero.min(), vmax=abs_combined.max()),
        cmap='viridis', 
        aspect='auto', 
        interpolation='nearest'
    )
    
    plt.title(f"System Values (Ax = b): {title}")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.colorbar(im, label='Absolute Value (Log Scale)')
    
    # 保存して後片付け
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def setup_solver(equation_set: EquationSet, grid: Grid, 
                scaling_method: Optional[str] = None) -> Tuple[Any, Any]:
    """
    ソルバーのセットアップと行列Aの取得
    
    Args:
        equation_set: 方程式セット
        grid: グリッド
        scaling_method: スケーリング手法
        
    Returns:
        (ソルバーインスタンス, 行列A)
    """
    # グリッド次元に応じたソルバーを作成
    solver_class = CCDSolver2D if grid.is_2d else CCDSolver1D
    solver = solver_class(equation_set, grid)
    
    # スケーリング設定
    if scaling_method:
        solver.scaling_method = scaling_method
    
    return solver, solver.matrix_A


def build_rhs_vector(solver: Union[CCDSolver1D, CCDSolver2D], 
                     grid: Grid, test_func) -> cp.ndarray:
    """
    右辺ベクトルの構築
    
    Args:
        solver: ソルバーインスタンス
        grid: グリッド
        test_func: テスト関数
        
    Returns:
        右辺ベクトル
    """
    if grid.is_2d:
        # 2Dケース
        nx, ny = grid.nx_points, grid.ny_points
        x_min, x_max = grid.x_min, grid.x_max
        y_min, y_max = grid.y_min, grid.y_max
        
        # ソース項
        f_values = cp.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                x, y = grid.get_point(i, j)
                f_values[i, j] = test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)
        
        # 境界値
        boundary = {
            'left_dirichlet': cp.array([test_func.f(x_min, y) for y in grid.y]),
            'right_dirichlet': cp.array([test_func.f(x_max, y) for y in grid.y]),
            'bottom_dirichlet': cp.array([test_func.f(x, y_min) for x in grid.x]),
            'top_dirichlet': cp.array([test_func.f(x, y_max) for x in grid.x]),
            'left_neumann': cp.array([test_func.df_dx(x_min, y) for y in grid.y]),
            'right_neumann': cp.array([test_func.df_dx(x_max, y) for y in grid.y]),
            'bottom_neumann': cp.array([test_func.df_dy(x, y_min) for x in grid.x]),
            'top_neumann': cp.array([test_func.df_dy(x, y_max) for x in grid.x])
        }
        
        return solver._build_rhs_vector(f_values=f_values, **boundary)
    else:
        # 1Dケース
        x_min, x_max = grid.x_min, grid.x_max
        
        # ソース項とソリューション
        f_values = cp.array([test_func.d2f(x) for x in grid.x])
        
        # 境界値
        boundary = {
            'left_dirichlet': test_func.f(x_min),
            'right_dirichlet': test_func.f(x_max),
            'left_neumann': test_func.df(x_min),
            'right_neumann': test_func.df(x_max)
        }
        
        return solver._build_rhs_vector(f_values=f_values, **boundary)


def analyze_system(equation_set: EquationSet, grid: Grid, name: str,
                  test_func, scaling_method: Optional[str] = None,
                  output_dir: str = "results") -> None:
    """
    行列システムを分析・可視化
    
    Args:
        equation_set: 方程式セット
        grid: グリッド
        name: 識別名
        test_func: テスト関数
        scaling_method: スケーリング手法
        output_dir: 出力ディレクトリ
    """
    try:
        # ソルバーのセットアップ
        solver, A = setup_solver(equation_set, grid, scaling_method)
        
        # 右辺ベクトル構築
        b = build_rhs_vector(solver, grid, test_func)
        
        # システム分析
        total_size = A.shape[0]
        nnz = A.nnz if hasattr(A, 'nnz') else np.count_nonzero(A)
        sparsity = 1.0 - (nnz / (total_size * total_size))
        
        grid_info = f"{grid.nx_points}x{grid.ny_points}" if grid.is_2d else f"{grid.n_points}"
        
        print(f"\n{name} 行列分析:")
        print(f"  グリッド: {grid_info}")
        print(f"  行列サイズ: {total_size}, 非ゼロ要素: {nnz}, 疎性率: {sparsity:.6f}")
        
        # Ax=bを解いてxを計算
        print("  線形方程式を解いています...")
        x = None
        
        # スケーリングが適用されている場合
        A_vis = A
        b_vis = b
        
        if scaling_method is not None:
            print(f"  スケーリング: {scaling_method}")
            
            # スケーリングを試みる
            try:
                scaler = plugin_manager.get_plugin(scaling_method)
                if scaler:
                    A_vis, b_vis, scaling_info = scaler.scale(A, b)
                    
                    # 線形方程式を解く
                    try:
                        x_vis = solver.linear_solver.solve(A_vis, b_vis)
                        # スケーリングを元に戻す
                        x = scaler.unscale(x_vis, scaling_info)
                    except Exception as e:
                        print(f"  線形方程式の解法に失敗しました: {e}")
            except Exception as e:
                print(f"  スケーリングの適用に失敗しました: {e}")
        else:
            # スケーリングなしで解く
            try:
                x = solver.linear_solver.solve(A, b)
            except Exception as e:
                print(f"  線形方程式の解法に失敗しました: {e}")
        
        if x is not None:
            print("  線形方程式の解を計算しました")
        
        # 可視化用ファイル名
        prefix = f"{name.lower()}_{grid_info}"
        if scaling_method:
            prefix += f"_{scaling_method.lower()}"
        
        # システム可視化
        dimension = "2D" if grid.is_2d else "1D"
        title = f"{name} {dimension} System"
        output_path = os.path.join(output_dir, f"{prefix}_system_values.png")
        
        # 可視化（これでA, x, bが表示される）
        visualize_system(A_vis, b_vis, x, title, output_path)
        
    except Exception as e:
        print(f"行列分析でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def run_verification(dimension: int, output_dir: str = "results") -> None:
    """
    指定次元での検証を実行
    
    Args:
        dimension: 次元（1または2）
        output_dir: 出力ディレクトリ
    """
    equation_set_types = ["poisson", "derivative"]
    
    for eq_type in equation_set_types:
        print(f"\n--- {dimension}次元 {eq_type.capitalize()} 方程式システムの検証 ---")
        
        try:
            # グリッドと関数のセットアップ
            if dimension == 1:
                grid = Grid(16, x_range=(-1.0, 1.0))
                test_func = TestFunctionFactory.create_standard_1d_functions()[3]  # Sine
            else:
                grid = Grid(5, 5, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
                test_func = TestFunctionFactory.create_standard_2d_functions()[0]  # Sine2D
            
            # 方程式セット作成
            equation_set = EquationSet.create(eq_type, dimension=dimension)
            
            # スケーリングなしでの分析
            analyze_system(
                equation_set, 
                grid, 
                f"{eq_type.capitalize()}{dimension}D_{test_func.name}",
                test_func,
                output_dir=output_dir
            )
            
            # SymmetricScalingでの分析
            analyze_system(
                equation_set, 
                grid, 
                f"{eq_type.capitalize()}{dimension}D_{test_func.name}", 
                test_func,
                scaling_method="SymmetricScaling",
                output_dir=output_dir
            )
            
        except Exception as e:
            print(f"{dimension}D {eq_type} 検証でエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()


def main(output_dir: str = "results") -> None:
    """メイン実行関数"""
    print("==== CCD行列構造検証ツール ====")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 検証実行
    run_verification(1, output_dir)
    run_verification(2, output_dir)
    
    print(f"\n検証が完了しました。結果は {output_dir} に保存されています。")


if __name__ == "__main__":
    main()