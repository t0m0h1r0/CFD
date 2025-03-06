"""
2次元可視化モジュール

2次元CCDソルバーの結果を視覚化するためのユーティリティ関数を提供します。
"""

import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from typing import Tuple, Dict, List

from test_functions_2d import TestFunction2DExplicit
from grid_config_2d import GridConfig2D


def visualize_derivative_results_2d(
    test_func: TestFunction2DExplicit,
    f_values: cp.ndarray,
    numerical_derivatives: Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray],
    analytical_derivatives: Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray],
    grid_config: GridConfig2D,
    xy_range: Tuple[Tuple[float, float], Tuple[float, float]],
    solver_name: str,
    save_path: str = None,
):
    """
    2次元偏導関数の計算結果を可視化

    Args:
        test_func: 2次元テスト関数
        f_values: グリッド点での関数値の2次元配列
        numerical_derivatives: 数値計算された偏導関数のタプル (f, f_x, f_y, f_xx, f_xy, f_yy)
        analytical_derivatives: 解析解の偏導関数のタプル (f, f_x, f_y, f_xx, f_xy, f_yy)
        grid_config: 2次元グリッド設定
        xy_range: (x軸の範囲, y軸の範囲) のタプル
        solver_name: ソルバーの名前 (プロットタイトルに使用)
        save_path: 保存先のパス (Noneの場合は保存しない)
    """
    nx, ny = grid_config.nx_points, grid_config.ny_points
    hx, hy = grid_config.hx, grid_config.hy
    x_range, y_range = xy_range

    # グリッド点の計算
    x = cp.linspace(x_range[0], x_range[1], nx)
    y = cp.linspace(y_range[0], y_range[1], ny)
    X, Y = cp.meshgrid(x, y)

    # 数値解と解析解
    f_num, f_x_num, f_y_num, f_xx_num, f_xy_num, f_yy_num = numerical_derivatives
    f_ana, f_x_ana, f_y_ana, f_xx_ana, f_xy_ana, f_yy_ana = analytical_derivatives

    # 誤差計算
    error_f = cp.sqrt(cp.mean((f_num - f_ana) ** 2))
    error_fx = cp.sqrt(cp.mean((f_x_num - f_x_ana) ** 2))
    error_fy = cp.sqrt(cp.mean((f_y_num - f_y_ana) ** 2))
    error_fxx = cp.sqrt(cp.mean((f_xx_num - f_xx_ana) ** 2))
    error_fxy = cp.sqrt(cp.mean((f_xy_num - f_xy_ana) ** 2))
    error_fyy = cp.sqrt(cp.mean((f_yy_num - f_yy_ana) ** 2))

    # プロット
    fig = plt.figure(figsize=(15, 20))
    fig.suptitle(f"Test Results for {test_func.name} Function using {solver_name}", fontsize=16)

    # サブプロットの構造: 関数値とその偏導関数を比較
    # 1行目: 関数値
    plot_comparison_2d(fig, 1, "Function (f)", f_ana, f_num, error_f, X, Y, x_range, y_range)
    
    # 2行目: x方向一階偏導関数
    plot_comparison_2d(fig, 2, "∂f/∂x", f_x_ana, f_x_num, error_fx, X, Y, x_range, y_range)
    
    # 3行目: y方向一階偏導関数
    plot_comparison_2d(fig, 3, "∂f/∂y", f_y_ana, f_y_num, error_fy, X, Y, x_range, y_range)
    
    # 4行目: x方向二階偏導関数
    plot_comparison_2d(fig, 4, "∂²f/∂x²", f_xx_ana, f_xx_num, error_fxx, X, Y, x_range, y_range)
    
    # 5行目: 混合偏導関数
    plot_comparison_2d(fig, 5, "∂²f/∂x∂y", f_xy_ana, f_xy_num, error_fxy, X, Y, x_range, y_range)
    
    # 6行目: y方向二階偏導関数
    plot_comparison_2d(fig, 6, "∂²f/∂y²", f_yy_ana, f_yy_num, error_fyy, X, Y, x_range, y_range)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # タイトルのスペースを確保

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"2D偏導関数の可視化結果を保存: {save_path}")
    
    plt.close(fig)


def plot_comparison_2d(fig, row, title, analytical, numerical, error, X, Y, x_range, y_range):
    """サブプロットに解析解と数値解の比較を表示"""
    # 解析解
    ax1 = fig.add_subplot(6, 3, (row-1)*3 + 1)
    im1 = ax1.pcolormesh(X.get(), Y.get(), analytical.get().T, cmap='viridis', shading='auto')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(f"{title} (Analytical)")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # 数値解
    ax2 = fig.add_subplot(6, 3, (row-1)*3 + 2)
    im2 = ax2.pcolormesh(X.get(), Y.get(), numerical.get().T, cmap='viridis', shading='auto')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(f"{title} (Numerical)")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # 誤差
    ax3 = fig.add_subplot(6, 3, (row-1)*3 + 3)
    diff = (numerical - analytical).get().T
    im3 = ax3.pcolormesh(X.get(), Y.get(), diff, cmap='coolwarm', shading='auto')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title(f"Error (RMSE: {error:.2e})")
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')


def visualize_error_comparison_2d(
    results: Dict[str, Dict[str, List[float]]],
    timings: Dict[str, Dict[str, float]],
    test_func_name: str,
    save_path: str = None,
):
    """
    異なる2Dソルバー間の誤差比較をプロット

    Args:
        results: ソルバー名 -> {関数名 -> [fx誤差, fy誤差, fxx誤差, fxy誤差, fyy誤差]} の辞書
        timings: ソルバー名 -> {関数名 -> 計算時間} の辞書
        test_func_name: テスト関数名
        save_path: 保存先のパス（指定がなければ自動生成）
    """
    solver_names = list(results.keys())

    # バープロット用のデータ準備
    bar_width = 0.15
    indexes = cp.arange(len(solver_names))

    # 誤差データと計算時間を抽出
    errors_fx = [results[name][test_func_name][0] for name in solver_names]
    errors_fy = [results[name][test_func_name][1] for name in solver_names]
    errors_fxx = [results[name][test_func_name][2] for name in solver_names]
    errors_fxy = [results[name][test_func_name][3] for name in solver_names]
    errors_fyy = [results[name][test_func_name][4] for name in solver_names]
    times = [timings[name][test_func_name] for name in solver_names]

    # 色の定義
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 誤差のプロット
    ax1.bar(
        indexes.get() - 2*bar_width,
        errors_fx,
        bar_width,
        label="∂f/∂x",
        color=colors[0],
    )
    ax1.bar(
        indexes.get() - bar_width,
        errors_fy,
        bar_width,
        label="∂f/∂y",
        color=colors[1],
    )
    ax1.bar(
        indexes.get(),
        errors_fxx,
        bar_width,
        label="∂²f/∂x²",
        color=colors[2],
    )
    ax1.bar(
        indexes.get() + bar_width,
        errors_fxy,
        bar_width,
        label="∂²f/∂x∂y",
        color=colors[3],
    )
    ax1.bar(
        indexes.get() + 2*bar_width,
        errors_fyy,
        bar_width,
        label="∂²f/∂y²",
        color=colors[4],
    )

    ax1.set_xlabel("Solver")
    ax1.set_ylabel("Error (L2 norm)")
    ax1.set_title(f"2D Error Comparison for {test_func_name} Function")
    ax1.set_xticks(indexes.get())
    ax1.set_xticklabels(solver_names, rotation=45, ha="right")
    ax1.legend()
    ax1.set_yscale("log")  # 対数スケールで表示

    # 計算時間のプロット
    ax2.bar(indexes.get(), times, color="green")
    ax2.set_xlabel("Solver")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Computation Time")
    ax2.set_xticks(indexes.get())
    ax2.set_xticklabels(solver_names, rotation=45, ha="right")

    plt.tight_layout()

    # 保存先のパス処理
    if save_path is None:
        os.makedirs("results", exist_ok=True)
        save_path = f"results/2d_comparison_{test_func_name.lower()}.png"

    plt.savefig(save_path)
    plt.close()


def visualize_2d_field(
    field: cp.ndarray,
    grid_config: GridConfig2D,
    xy_range: Tuple[Tuple[float, float], Tuple[float, float]],
    title: str,
    save_path: str = None,
    cmap: str = 'viridis',
    with_contour: bool = True,
):
    """
    2次元スカラー場を可視化

    Args:
        field: 可視化する2次元スカラー場 (nx × ny配列)
        grid_config: 2次元グリッド設定
        xy_range: (x軸の範囲, y軸の範囲) のタプル
        title: プロットのタイトル
        save_path: 保存先のパス (Noneの場合は保存しない)
        cmap: カラーマップ
        with_contour: 等高線を表示するかどうか
    """
    nx, ny = grid_config.nx_points, grid_config.ny_points
    x_range, y_range = xy_range
    
    # グリッド点の計算
    x = cp.linspace(x_range[0], x_range[1], nx)
    y = cp.linspace(y_range[0], y_range[1], ny)
    X, Y = cp.meshgrid(x, y)
    
    # NumPy配列に変換
    X_np = X.get()
    Y_np = Y.get()
    field_np = field.get().T  # 転置して表示
    
    plt.figure(figsize=(10, 8))
    
    # カラーマップで表示
    plt.pcolormesh(X_np, Y_np, field_np, cmap=cmap, shading='auto')
    cbar = plt.colorbar()
    cbar.set_label('Value')
    
    # 等高線を追加（オプション）
    if with_contour:
        contour = plt.contour(X_np, Y_np, field_np, colors='black', linewidths=0.5)
        plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"2Dスカラー場を保存: {save_path}")
    
    plt.close()


def visualize_vector_field_2d(
    fx: cp.ndarray,
    fy: cp.ndarray,
    grid_config: GridConfig2D,
    xy_range: Tuple[Tuple[float, float], Tuple[float, float]],
    title: str,
    save_path: str = None,
    subsample: int = 2,  # サブサンプリング係数
):
    """
    2次元ベクトル場（勾配場）を可視化

    Args:
        fx: x方向の勾配成分 (nx × ny配列)
        fy: y方向の勾配成分 (nx × ny配列)
        grid_config: 2次元グリッド設定
        xy_range: (x軸の範囲, y軸の範囲) のタプル
        title: プロットのタイトル
        save_path: 保存先のパス (Noneの場合は保存しない)
        subsample: ベクトル表示を間引く係数（大きいほど表示が少なくなる）
    """
    nx, ny = grid_config.nx_points, grid_config.ny_points
    x_range, y_range = xy_range
    
    # グリッド点の計算
    x = cp.linspace(x_range[0], x_range[1], nx)
    y = cp.linspace(y_range[0], y_range[1], ny)
    X, Y = cp.meshgrid(x, y)
    
    # NumPy配列に変換し、サブサンプリング
    X_sub = X[::subsample, ::subsample].get()
    Y_sub = Y[::subsample, ::subsample].get()
    fx_sub = fx[::subsample, ::subsample].get().T
    fy_sub = fy[::subsample, ::subsample].get().T
    
    # ベクトルの大きさを計算
    magnitude = cp.sqrt(fx**2 + fy**2).get().T
    
    plt.figure(figsize=(10, 8))
    
    # 背景をベクトルの大きさで色付け
    plt.pcolormesh(X.get(), Y.get(), magnitude, cmap='viridis', shading='auto')
    cbar = plt.colorbar()
    cbar.set_label('Gradient Magnitude')
    
    # ベクトル場を表示
    plt.quiver(X_sub, Y_sub, fx_sub, fy_sub, angles='xy', scale_units='xy', scale=1.0, color='white')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"2Dベクトル場を保存: {save_path}")
    
    plt.close()