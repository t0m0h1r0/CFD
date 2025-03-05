"""
2次元ポアソン方程式の解法例

CCD2DSolverを使用して2次元ポアソン方程式を解く例を示します。
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# プロジェクトルートをインポートパスに追加
sys.path.append('path/to/your/project/root')  # 実際のパスに変更してください

from grid_2d_config import Grid2DConfig
from ccd2d_solver import CCD2DSolver

def plot_solution(x, y, u, title="Solution", cmap="viridis"):
    """
    関数uのカラーマップとコンター線を描画
    
    Args:
        x: x座標の1次元配列
        y: y座標の1次元配列
        u: 関数値の2次元配列
        title: プロットのタイトル
        cmap: カラーマップ
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # カラーマップでプロット
    c = ax.pcolormesh(x, y, u, cmap=cmap, shading='auto')
    fig.colorbar(c, ax=ax, label='値')
    
    # コンター線を追加
    contour = ax.contour(x, y, u, colors='k', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    return fig, ax

def compute_error(u_numerical, u_exact):
    """
    数値解と厳密解の誤差を計算
    
    Args:
        u_numerical: 数値解の2次元配列
        u_exact: 厳密解の2次元配列
        
    Returns:
        (L2誤差, 最大誤差)
    """
    diff = u_numerical - u_exact
    l2_error = jnp.sqrt(jnp.mean(diff**2))
    max_error = jnp.max(jnp.abs(diff))
    
    return l2_error, max_error

def example_poisson_dirichlet():
    """
    ディリクレ境界条件を持つポアソン方程式の例
    
    問題: ∇²u = -2π²sin(πx)sin(πy) on [0,1]×[0,1]
          u = 0 on 境界
          
    厳密解: u = sin(πx)sin(πy)
    """
    # グリッドパラメータ
    nx, ny = 64, 64
    hx, hy = 1.0/(nx-1), 1.0/(ny-1)
    
    # グリッド設定
    grid_config = Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        # 全ての境界でu=0
        dirichlet_values={
            'left': 0.0,
            'right': 0.0,
            'bottom': 0.0,
            'top': 0.0
        },
        # ラプラシアン演算子用の係数
        coeffs=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # uxx + uyy
    )
    
    # ソルバーの初期化
    solver = CCD2DSolver(grid_config)
    
    # グリッド点の生成
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # 右辺関数 f(x,y) = -2π²sin(πx)sin(πy)
    pi = jnp.pi
    f = -2 * pi**2 * jnp.sin(pi * X) * jnp.sin(pi * Y)
    
    # ポアソン方程式を解く
    solution = solver.solve(f)
    u_numerical = solution['u']
    
    # 厳密解
    u_exact = jnp.sin(pi * X) * jnp.sin(pi * Y)
    
    # 誤差計算
    l2_error, max_error = compute_error(u_numerical, u_exact)
    print(f"L2誤差: {l2_error:.6e}")
    print(f"最大誤差: {max_error:.6e}")
    
    # 結果をプロット
    fig1, ax1 = plot_solution(x, y, u_numerical, title="数値解 u(x,y)")
    fig2, ax2 = plot_solution(x, y, u_exact, title="厳密解 sin(πx)sin(πy)")
    
    # 誤差のプロット
    fig3, ax3 = plot_solution(x, y, jnp.abs(u_numerical - u_exact), 
                             title=f"絶対誤差 (L2: {l2_error:.2e}, Max: {max_error:.2e})",
                             cmap="hot")
    
    return u_numerical, u_exact, (fig1, fig2, fig3)

def example_poisson_mixed_boundary():
    """
    混合境界条件を持つポアソン方程式の例
    
    問題: ∇²u = 0 on [0,1]×[0,1]
          u = 0 on x=0, x=1, y=0
          u = sin(πx) on y=1
    """
    # グリッドパラメータ
    nx, ny = 64, 64
    hx, hy = 1.0/(nx-1), 1.0/(ny-1)
    
    # x座標の配列
    x = jnp.linspace(0, 1, nx)
    
    # グリッド設定
    grid_config = Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        # 境界条件
        dirichlet_values={
            'left': 0.0,                # x=0: u=0
            'right': 0.0,               # x=1: u=0
            'bottom': 0.0,              # y=0: u=0
            'top': jnp.sin(jnp.pi * x)  # y=1: u=sin(πx)
        },
        # ラプラシアン演算子用の係数
        coeffs=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # uxx + uyy
    )
    
    # ソルバーの初期化
    solver = CCD2DSolver(grid_config)
    
    # グリッド点の生成
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # 右辺関数 f(x,y) = 0 (ラプラス方程式)
    f = jnp.zeros((ny, nx))
    
    # ポアソン方程式を解く
    solution = solver.solve(f)
    u = solution['u']
    
    # 結果をプロット
    fig, ax = plot_solution(x, y, u, title="混合境界条件でのラプラス方程式の解")
    
    return u, fig

def example_heat_diffusion_steady_state():
    """
    定常状態の熱拡散方程式の例
    
    問題: ∇²u = 0 on [0,1]×[0,1]
          u = 0 on x=0, x=1
          u = 0 on y=0
          u = sin(πx) on y=1
          
    物理的には長方形の板の熱伝導問題（上部が不均一に加熱、他の境界が冷却されている）
    """
    # グリッドパラメータ
    nx, ny = 64, 64
    hx, hy = 1.0/(nx-1), 1.0/(ny-1)
    
    # x座標の配列
    x = jnp.linspace(0, 1, nx)
    
    # グリッド設定
    grid_config = Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        # 境界条件
        dirichlet_values={
            'left': 0.0,                # x=0: u=0 (左側が冷却)
            'right': 0.0,               # x=1: u=0 (右側が冷却)
            'bottom': 0.0,              # y=0: u=0 (下側が冷却)
            'top': jnp.sin(jnp.pi * x)  # y=1: u=sin(πx) (上側が不均一に加熱)
        },
        # ラプラシアン演算子用の係数
        coeffs=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # uxx + uyy
    )
    
    # ソルバーの初期化
    solver = CCD2DSolver(grid_config)
    
    # グリッド点の生成
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # 右辺関数 f(x,y) = 0 (ラプラス方程式)
    f = jnp.zeros((ny, nx))
    
    # 熱拡散方程式を解く
    solution = solver.solve(f)
    u = solution['u']
    
    # 熱流束（温度勾配）を計算
    ux = solution['ux']
    uy = solution['uy']
    
    # 結果をプロット
    fig1, ax1 = plot_solution(x, y, u, title="定常状態の温度分布", cmap="hot")
    
    # 熱流束ベクトル場をプロット
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    # 表示用のグリッドを間引く
    skip = 4
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              -ux[::skip, ::skip], -uy[::skip, ::skip],  # 熱は高温から低温へ流れる
              color='k', scale=20)
    c = ax2.pcolormesh(x, y, u, cmap="hot", shading='auto')
    fig2.colorbar(c, ax=ax2, label='温度')
    ax2.set_title("温度分布と熱流束")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    return u, (fig1, fig2)

def main():
    """メイン関数"""
    print("例1: ディリクレ境界条件を持つポアソン方程式")
    u_numerical, u_exact, figs1 = example_poisson_dirichlet()
    
    print("\n例2: 混合境界条件を持つポアソン方程式")
    u_mixed, fig2 = example_poisson_mixed_boundary()
    
    print("\n例3: 定常状態の熱拡散方程式")
    u_heat, figs3 = example_heat_diffusion_steady_state()
    
    # 結果の表示
    plt.show()

if __name__ == "__main__":
    main()