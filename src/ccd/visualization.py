"""
可視化モジュール - 微分組み合わせ対応版

CCDソルバーの結果を視覚化するためのユーティリティ関数を提供します。
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict, Any

from test_functions import TestFunction
from ccd_core import GridConfig


def visualize_derivative_results(
    test_func: TestFunction,
    f_values: jnp.ndarray,
    numerical_derivatives: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    grid_config: GridConfig,
    x_range: Tuple[float, float],
    solver_name: str,
    save_path: str = None
):
    """
    導関数の計算結果を可視化
    
    Args:
        test_func: テスト関数
        f_values: グリッド点での元の関数値
        numerical_derivatives: 数値計算された導関数のタプル (psi, psi_prime, psi_second, psi_third)
        grid_config: グリッド設定
        x_range: x軸の範囲 (開始位置, 終了位置)
        solver_name: ソルバーの名前 (プロットタイトルに使用)
        save_path: 保存先のパス (Noneの場合は保存しない)
    """
    n = grid_config.n_points
    h = grid_config.h
    x_start = x_range[0]

    # 導関数のアンパック
    psi, psi_prime, psi_second, psi_third = numerical_derivatives

    # グリッド点の計算
    x_points = jnp.array([x_start + i * h for i in range(n)])

    # 高解像度の点での解析解
    x_fine = jnp.linspace(x_range[0], x_range[1], 200)
    analytical_f = jnp.array([test_func.f(x) for x in x_fine])
    analytical_df = jnp.array([test_func.df(x) for x in x_fine])
    analytical_d2f = jnp.array([test_func.d2f(x) for x in x_fine])
    analytical_d3f = jnp.array([test_func.d3f(x) for x in x_fine])

    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Test Results for {test_func.name} Function using {solver_name}")

    # 元関数
    axes[0, 0].plot(x_fine, analytical_f, "b-", label="f(x)")
    axes[0, 0].plot(x_points, f_values, "r-", label="Input f Values")
    axes[0, 0].plot(x_points, psi, "g-", label="Computed ψ")
    axes[0, 0].set_title("Function Values")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 1階導関数
    axes[0, 1].plot(x_fine, analytical_df, "b-", label="Analytical f'")
    # 線で表示するように変更
    axes[0, 1].plot(
        x_points, psi_prime, "r-", label="Computed ψ'"
    )
    axes[0, 1].set_title("First Derivative")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 2階導関数
    axes[1, 0].plot(x_fine, analytical_d2f, "b-", label="Analytical f''")
    # 線で表示するように変更
    axes[1, 0].plot(
        x_points, psi_second, "r-", label="Computed ψ''"
    )
    axes[1, 0].set_title("Second Derivative")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 3階導関数
    axes[1, 1].plot(x_fine, analytical_d3f, "b-", label="Analytical f'''")
    # 線で表示するように変更
    axes[1, 1].plot(
        x_points, psi_third, "r-", label="Computed ψ'''"
    )
    axes[1, 1].set_title("Third Derivative")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        # ディレクトリ部分を取得
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def visualize_error_comparison(
    results: Dict[str, Dict[str, List[float]]],
    timings: Dict[str, Dict[str, float]],
    test_func_name: str,
    save_path: str = None
):
    """
    異なるソルバー間の誤差比較をプロット
    
    Args:
        results: ソルバー名 -> {関数名 -> [1階誤差, 2階誤差, 3階誤差]} の辞書
        timings: ソルバー名 -> {関数名 -> 計算時間} の辞書
        test_func_name: テスト関数名
        save_path: 保存先のパス（指定がなければ自動生成）
    """
    solver_names = list(results.keys())
    
    # バープロット用のデータ準備
    bar_width = 0.25
    indexes = jnp.arange(len(solver_names))
    
    # 1階, 2階, 3階の誤差データを抽出
    errors_1st = [results[name][test_func_name][0] for name in solver_names]
    errors_2nd = [results[name][test_func_name][1] for name in solver_names]
    errors_3rd = [results[name][test_func_name][2] for name in solver_names]
    
    # 計算時間
    times = [timings[name][test_func_name] for name in solver_names]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 誤差のプロット
    ax1.bar(indexes - bar_width, errors_1st, bar_width, label='1st Derivative')
    ax1.bar(indexes, errors_2nd, bar_width, label='2nd Derivative')
    ax1.bar(indexes + bar_width, errors_3rd, bar_width, label='3rd Derivative')
    
    ax1.set_xlabel('Solver / Diff Mode')
    ax1.set_ylabel('Error (L2 norm)')
    ax1.set_title(f'Error Comparison for {test_func_name} Function')
    ax1.set_xticks(indexes)
    ax1.set_xticklabels(solver_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log')  # 対数スケールで表示
    
    # 計算時間のプロット
    ax2.bar(indexes, times, color='green')
    ax2.set_xlabel('Solver / Diff Mode')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time')
    ax2.set_xticks(indexes)
    ax2.set_xticklabels(solver_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存先が指定されていなければデフォルトパスを使用
    if save_path is None:
        # 'results' ディレクトリを作成
        os.makedirs("results", exist_ok=True)
        save_path = f"results/comparison_{test_func_name.lower()}.png"
        
    plt.savefig(save_path)
    plt.close()


def visualize_matrix_properties(L: jnp.ndarray, title: str, save_path: str = None):
    """
    行列の特性を可視化
    
    Args:
        L: 対象の行列
        title: プロットのタイトル
        save_path: 保存先のパス (Noneの場合は保存しない)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # スパイパターンのヒートマップ
    im1 = ax1.imshow(jnp.abs(L), cmap='viridis', norm='log')
    ax1.set_title('Matrix Sparsity Pattern (log scale)')
    plt.colorbar(im1, ax=ax1)
    
    # 特異値分布
    s = jnp.linalg.svd(L, compute_uv=False)
    ax2.semilogy(range(1, len(s) + 1), s, 'r-')
    ax2.set_title('Singular Value Distribution')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Singular Value (log scale)')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        # ディレクトリ部分を取得
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
    plt.close()