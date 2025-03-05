"""
2次元可視化ユーティリティモジュール

2次元CCDソルバーの結果を視覚化するための関数群
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, Optional, List, Union, Any
import os

from grid2d_config import Grid2DConfig
from test2d_functions import Test2DFunction


def visualize_2d_field(
    field: cp.ndarray, 
    grid_config: Grid2DConfig, 
    title: str = "2D Field", 
    colormap: str = "viridis",
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2次元スカラー場を可視化
    
    Args:
        field: 可視化する2次元スカラー場
        grid_config: グリッド設定
        title: プロットのタイトル
        colormap: カラーマップ名
        show_colorbar: カラーバーを表示するかどうか
        figsize: 図のサイズ
        save_path: 保存先のパス（省略可）
        
    Returns:
        生成された図
    """
    # CuPy配列をNumPy配列に変換
    if isinstance(field, cp.ndarray):
        field = cp.asnumpy(field)
    
    # グリッド点を生成
    x = np.linspace(0, (grid_config.nx - 1) * grid_config.hx, grid_config.nx)
    y = np.linspace(0, (grid_config.ny - 1) * grid_config.hy, grid_config.ny)
    X, Y = np.meshgrid(x, y)
    
    # 値をプロット
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(X, Y, field.T, cmap=colormap, shading='auto')
    
    if show_colorbar:
        fig.colorbar(im, ax=ax, label='Value')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_2d_surface(
    field: cp.ndarray, 
    grid_config: Grid2DConfig, 
    title: str = "2D Surface", 
    colormap: str = "viridis",
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 30, 
    azim: float = -60,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2次元スカラー場を3D表面として可視化
    
    Args:
        field: 可視化する2次元スカラー場
        grid_config: グリッド設定
        title: プロットのタイトル
        colormap: カラーマップ名
        show_colorbar: カラーバーを表示するかどうか
        figsize: 図のサイズ
        elev: 仰角
        azim: 方位角
        save_path: 保存先のパス（省略可）
        
    Returns:
        生成された図
    """
    # CuPy配列をNumPy配列に変換
    if isinstance(field, cp.ndarray):
        field = cp.asnumpy(field)
    
    # グリッド点を生成
    x = np.linspace(0, (grid_config.nx - 1) * grid_config.hx, grid_config.nx)
    y = np.linspace(0, (grid_config.ny - 1) * grid_config.hy, grid_config.ny)
    X, Y = np.meshgrid(x, y)
    
    # 3D表面プロット
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 転置しているのは、matplotlibのプロット方向と配列の形状を一致させるため
    surf = ax.plot_surface(X, Y, field.T, cmap=colormap, 
                          linewidth=0, antialiased=False, alpha=0.7)
    
    if show_colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Value')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Value')
    ax.set_title(title)
    
    # 視点を設定
    ax.view_init(elev=elev, azim=azim)
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_error_field(
    numerical: cp.ndarray, 
    analytical: cp.ndarray, 
    grid_config: Grid2DConfig,
    title: str = "Error Field", 
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    数値解と解析解の誤差場を可視化
    
    Args:
        numerical: 数値解
        analytical: 解析解
        grid_config: グリッド設定
        title: プロットのタイトル
        log_scale: 対数スケールを使用するかどうか
        figsize: 図のサイズ
        save_path: 保存先のパス（省略可）
        
    Returns:
        生成された図
    """
    # CuPy配列をNumPy配列に変換
    if isinstance(numerical, cp.ndarray):
        numerical = cp.asnumpy(numerical)
    if isinstance(analytical, cp.ndarray):
        analytical = cp.asnumpy(analytical)
    
    # 誤差を計算
    error = np.abs(numerical - analytical)
    
    # グリッド点を生成
    x = np.linspace(0, (grid_config.nx - 1) * grid_config.hx, grid_config.nx)
    y = np.linspace(0, (grid_config.ny - 1) * grid_config.hy, grid_config.ny)
    X, Y = np.meshgrid(x, y)
    
    # 誤差をプロット
    fig, ax = plt.subplots(figsize=figsize)
    
    if log_scale:
        # 0を避けるために小さな値を加える
        error = np.maximum(error, 1e-15)
        im = ax.pcolormesh(X, Y, error.T, cmap='hot', norm=plt.cm.colors.LogNorm(), shading='auto')
    else:
        im = ax.pcolormesh(X, Y, error.T, cmap='hot', shading='auto')
    
    fig.colorbar(im, ax=ax, label='Absolute Error')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    # 平均誤差と最大誤差を表示
    mean_err = np.mean(error)
    max_err = np.max(error)
    l2_err = np.sqrt(np.mean(error**2))
    
    ax.text(0.05, 0.95, f'Mean Error: {mean_err:.2e}\nMax Error: {max_err:.2e}\nL2 Error: {l2_err:.2e}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_derivative_comparison(
    numerical_derivatives: Dict[str, cp.ndarray],
    analytical_derivatives: Dict[str, cp.ndarray],
    grid_config: Grid2DConfig,
    derivative_keys: List[str] = ["f_x", "f_y", "f_xx", "f_yy"],
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    複数の導関数の数値解と解析解を比較可視化
    
    Args:
        numerical_derivatives: 数値解の辞書 {"f_x": array, ...}
        analytical_derivatives: 解析解の辞書 {"f_x": array, ...}
        grid_config: グリッド設定
        derivative_keys: 可視化する導関数のキーのリスト
        figsize: 図のサイズ
        save_path: 保存先のパス（省略可）
        
    Returns:
        生成された図
    """
    # プロットする導関数の数からサブプロットの配置を計算
    n_plots = len(derivative_keys)
    rows = int(np.ceil(n_plots / 2))
    cols = min(n_plots, 2)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 誤差メトリクス保存用の辞書
    error_metrics = {}
    
    # グリッド点を生成
    x = np.linspace(0, (grid_config.nx - 1) * grid_config.hx, grid_config.nx)
    y = np.linspace(0, (grid_config.ny - 1) * grid_config.hy, grid_config.ny)
    X, Y = np.meshgrid(x, y)
    
    # 各導関数についてプロット
    for i, key in enumerate(derivative_keys):
        if i < len(axes) and key in numerical_derivatives and key in analytical_derivatives:
            # CuPy配列をNumPy配列に変換
            num = cp.asnumpy(numerical_derivatives[key]) if isinstance(numerical_derivatives[key], cp.ndarray) else numerical_derivatives[key]
            ana = cp.asnumpy(analytical_derivatives[key]) if isinstance(analytical_derivatives[key], cp.ndarray) else analytical_derivatives[key]
            
            # 1D断面をプロット（中央の行と列）
            mid_x = grid_config.nx // 2
            mid_y = grid_config.ny // 2
            
            ax = axes[i]
            ax.plot(x, num[mid_x, :], 'r-', label='Numerical (y=const)')
            ax.plot(x, ana[mid_x, :], 'r--', label='Analytical (y=const)')
            ax.plot(y, num[:, mid_y], 'b-', label='Numerical (x=const)')
            ax.plot(y, ana[:, mid_y], 'b--', label='Analytical (x=const)')
            
            ax.set_xlabel('Coordinate')
            ax.set_ylabel('Value')
            ax.set_title(f'{key} Comparison')
            
            # 誤差計算
            error = np.abs(num - ana)
            mean_err = np.mean(error)
            max_err = np.max(error)
            l2_err = np.sqrt(np.mean(error**2))
            
            error_metrics[key] = {"mean": mean_err, "max": max_err, "l2": l2_err}
            
            # 誤差情報を表示
            ax.text(0.05, 0.95, f'Mean Err: {mean_err:.2e}\nMax Err: {max_err:.2e}\nL2 Err: {l2_err:.2e}',
                    transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
            
            if i == 0:  # 最初のプロットにだけ凡例を表示
                ax.legend()
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, error_metrics


def visualize_all_results(
    numerical_results: Dict[str, cp.ndarray],
    analytical_results: Dict[str, cp.ndarray],
    grid_config: Grid2DConfig,
    test_func_name: str = "Function",
    output_dir: str = "results_2d",
    prefix: str = "",
) -> Dict[str, Dict[str, float]]:
    """
    すべての結果を可視化し、誤差メトリクスを返す
    
    Args:
        numerical_results: 数値解の辞書
        analytical_results: 解析解の辞書
        grid_config: グリッド設定
        test_func_name: テスト関数名
        output_dir: 出力ディレクトリ
        prefix: ファイル名の接頭辞
        
    Returns:
        誤差メトリクスの辞書
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 誤差メトリクス保存用の辞書
    error_metrics = {}
    
    # 関数値の可視化
    if "f" in numerical_results and "f" in analytical_results:
        # 数値解の表面プロット
        visualize_2d_surface(
            numerical_results["f"], 
            grid_config, 
            title=f"{test_func_name} - Numerical Solution",
            save_path=os.path.join(output_dir, f"{prefix}{test_func_name}_numerical_surface.png")
        )
        
        # 解析解の表面プロット
        visualize_2d_surface(
            analytical_results["f"], 
            grid_config, 
            title=f"{test_func_name} - Analytical Solution",
            save_path=os.path.join(output_dir, f"{prefix}{test_func_name}_analytical_surface.png")
        )
        
        # 誤差場
        visualize_error_field(
            numerical_results["f"], 
            analytical_results["f"], 
            grid_config,
            title=f"{test_func_name} - Function Value Error",
            save_path=os.path.join(output_dir, f"{prefix}{test_func_name}_error_field.png")
        )
    
    # 導関数の比較可視化
    deriv_keys = [key for key in numerical_results.keys() if key.startswith("f_")]
    deriv_keys = [key for key in deriv_keys if key in analytical_results]
    
    # 1階導関数
    first_order = [key for key in deriv_keys if len(key) == 3]  # f_x, f_y
    if first_order:
        _, metrics1 = visualize_derivative_comparison(
            numerical_results, 
            analytical_results, 
            grid_config,
            first_order,
            save_path=os.path.join(output_dir, f"{prefix}{test_func_name}_first_derivatives.png")
        )
        error_metrics.update(metrics1)
    
    # 2階導関数
    second_order = [key for key in deriv_keys if len(key) == 4]  # f_xx, f_yy, f_xy
    if second_order:
        _, metrics2 = visualize_derivative_comparison(
            numerical_results, 
            analytical_results, 
            grid_config,
            second_order,
            save_path=os.path.join(output_dir, f"{prefix}{test_func_name}_second_derivatives.png")
        )
        error_metrics.update(metrics2)
    
    # 3階導関数
    third_order = [key for key in deriv_keys if len(key) == 5]  # f_xxx, f_yyy, f_xxy, f_xyy
    if third_order:
        _, metrics3 = visualize_derivative_comparison(
            numerical_results, 
            analytical_results, 
            grid_config,
            third_order,
            save_path=os.path.join(output_dir, f"{prefix}{test_func_name}_third_derivatives.png")
        )
        error_metrics.update(metrics3)
    
    return error_metrics
