#!/usr/bin/env python3
"""
スパース行列を使用した2次元CCD法コマンドラインインターフェース

メモリ効率の良いスパース行列を用いて、2次元偏微分方程式を解くためのインターフェース。
"""

import argparse
import os
import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# インポートパスを追加（必要に応じて変更）
sys.path.append('.')
sys.path.append('..')

from grid_2d_config import Grid2DConfig
from ccd2d_solver_sparse import SparseCCD2DSolver
from ccd2d_tester import TestFunction2D, TestFunctionFactory2D


class SparseCCD2DMethodTester:
    """スパース行列を使用した2次元CCD法テスター"""
    
    def __init__(self, grid_config, solver_kwargs=None, coeffs=None):
        """初期化"""
        self.grid_config = grid_config
        self.solver_kwargs = solver_kwargs or {}
        
        # グリッドパラメータ
        self.nx, self.ny = grid_config.nx, grid_config.ny
        self.hx, self.hy = grid_config.hx, grid_config.hy
        
        # 係数の設定
        if coeffs is not None:
            self.coeffs = coeffs
            grid_config.coeffs = coeffs
        else:
            self.coeffs = grid_config.coeffs
        
        # テスト関数の取得
        self.test_functions = TestFunctionFactory2D.create_standard_functions()
        
        # x、y座標配列
        self.x = jnp.linspace(0, 1, self.nx)  # 標準区間 [0,1] を使用
        self.y = jnp.linspace(0, 1, self.ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        
        # ソルバーの初期化
        filtered_solver_kwargs = {}
        for key, value in self.solver_kwargs.items():
            if key in ["use_direct_solver"]:  # 受け付けるパラメータのみ
                filtered_solver_kwargs[key] = value
        
        self.solver = SparseCCD2DSolver(grid_config, **filtered_solver_kwargs)
    
    def test_poisson(self, output_dir="results", visualize=True):
        """ポアソン方程式のテストを実行"""
        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 関数 f(x,y) = sin(πx)sin(πy) の解析解はu(x,y) = -sin(πx)sin(πy)/(2π²)
        pi = jnp.pi
        
        # 解析解
        f_values = jnp.sin(pi * self.X) * jnp.sin(pi * self.Y)
        u_exact = -f_values / (2 * pi**2)
        
        # 境界条件を設定（解析解に合わせる）
        self.grid_config.dirichlet_values = {
            'left': u_exact[:, 0],
            'right': u_exact[:, -1],
            'bottom': u_exact[0, :],
            'top': u_exact[-1, :]
        }
        
        # ラプラシアン演算子用の係数に設定
        old_coeffs = self.grid_config.coeffs.copy()
        self.grid_config.coeffs = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # ∇²u = uxx + uyy
        
        # ソルバーを初期化
        solver = SparseCCD2DSolver(self.grid_config, **self.solver_kwargs)
        
        # 右辺関数 f(x,y) = sin(πx)sin(πy)
        f = f_values
        
        print(f"\nポアソン方程式 ∇²u = f を解いています...")
        print(f"グリッドサイズ: {self.nx}x{self.ny}")
        
        # 方程式を解く
        solution = solver.solve(f)
        u_numerical = solution['u']
        
        # 誤差計算
        error = jnp.abs(u_numerical - u_exact)
        l2_error = jnp.sqrt(jnp.mean(error**2))
        max_error = jnp.max(error)
        
        print(f"ポアソン方程式の解析終了:")
        print(f"L2誤差: {l2_error:.6e}")
        print(f"最大誤差: {max_error:.6e}")
        
        # 可視化
        if visualize:
            # プロットの作成
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 数値解
            im0 = axes[0].pcolormesh(self.X, self.Y, u_numerical, cmap='viridis', shading='auto')
            axes[0].set_title('数値解')
            fig.colorbar(im0, ax=axes[0])
            
            # 厳密解
            im1 = axes[1].pcolormesh(self.X, self.Y, u_exact, cmap='viridis', shading='auto')
            axes[1].set_title('厳密解')
            fig.colorbar(im1, ax=axes[1])
            
            # 誤差
            im2 = axes[2].pcolormesh(self.X, self.Y, error, cmap='hot', shading='auto')
            axes[2].set_title(f'誤差 (L2: {l2_error:.2e}, Max: {max_error:.2e})')
            fig.colorbar(im2, ax=axes[2])
            
            # すべてのプロットに座標軸ラベルを設定
            for ax in axes:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            
            # プロットを保存
            plt.tight_layout()
            filename = f"poisson_sparse_{self.nx}x{self.ny}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.show()
        
        # 係数を元に戻す
        self.grid_config.coeffs = old_coeffs
        
        return u_numerical, u_exact, l2_error, max_error


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="スパース行列を使用した2次元CCD法")

    # 親パーサー（共通オプション）
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--nx", type=int, default=32, help="x方向のグリッド点数")
    parent_parser.add_argument("--ny", type=int, default=32, help="y方向のグリッド点数")
    parent_parser.add_argument("--xrange", type=float, nargs=2, default=[0.0, 1.0], help="x軸の範囲 (開始点 終了点)")
    parent_parser.add_argument("--yrange", type=float, nargs=2, default=[0.0, 1.0], help="y軸の範囲 (開始点 終了点)")
    parent_parser.add_argument("--coeffs", type=float, nargs="+", default=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 
                              help="係数 [a, b, c, d, e, f]: a*u + b*ux + c*uxx + d*uy + e*uyy + f*uxy")
    parent_parser.add_argument("--out", type=str, default="results", help="出力ディレクトリ")
    parent_parser.add_argument("--no-viz", action="store_true", help="可視化を無効化")

    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド", required=True)

    # テストコマンド
    test_parser = subparsers.add_parser("test", parents=[parent_parser], help="ポアソン方程式のテスト")
    test_parser.add_argument("--iterative", action="store_true", help="反復法を使用")

    # 比較コマンド
    compare_parser = subparsers.add_parser("compare", parents=[parent_parser], help="異なるグリッドサイズでの比較")
    compare_parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128], help="比較するグリッドサイズ")

    return parser.parse_args()


def create_grid_config(args) -> Grid2DConfig:
    """
    コマンドライン引数からグリッド設定を作成
    
    Args:
        args: コマンドライン引数
        
    Returns:
        Grid2DConfig: 2次元グリッド設定
    """
    nx, ny = args.nx, args.ny
    x_range = args.xrange
    y_range = args.yrange
    
    # グリッド幅を計算
    hx = (x_range[1] - x_range[0]) / (nx - 1)
    hy = (y_range[1] - y_range[0]) / (ny - 1)
    
    # グリッド設定を作成
    return Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        coeffs=args.coeffs
    )


def run_test(args):
    """ポアソン方程式のテストを実行"""
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # ソルバーパラメータ
    solver_kwargs = {
        "use_direct_solver": not args.iterative
    }
    
    # テスターの初期化
    tester = SparseCCD2DMethodTester(
        grid_config=grid_config,
        solver_kwargs=solver_kwargs,
        coeffs=args.coeffs
    )
    
    # テスト実行
    tester.test_poisson(output_dir=args.out, visualize=not args.no_viz)


def run_comparison(args):
    """異なるグリッドサイズでの比較を実行"""
    print(f"異なるグリッドサイズでの比較を実行中...")
    
    # 結果を保存するリスト
    grid_sizes = []
    l2_errors = []
    max_errors = []
    
    # 各グリッドサイズでテスト
    for size in args.sizes:
        print(f"\nグリッドサイズ {size}x{size} でテスト中...")
        
        # グリッド設定を作成
        grid_config = Grid2DConfig(
            nx=size,
            ny=size,
            hx=1.0 / (size - 1),
            hy=1.0 / (size - 1),
            coeffs=args.coeffs
        )
        
        # ソルバーパラメータ
        solver_kwargs = {
            "use_direct_solver": True  # 比較のため直接法を使用
        }
        
        # テスターの初期化
        tester = SparseCCD2DMethodTester(
            grid_config=grid_config,
            solver_kwargs=solver_kwargs,
            coeffs=args.coeffs
        )
        
        # テスト実行（可視化は最後のサイズのみ）
        visualize = not args.no_viz and size == args.sizes[-1]
        _, _, l2_error, max_error = tester.test_poisson(output_dir=args.out, visualize=visualize)
        
        # 結果を保存
        grid_sizes.append(size)
        l2_errors.append(float(l2_error))
        max_errors.append(float(max_error))
    
    # 結果をプロット
    if not args.no_viz:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # グリッドサイズと誤差のプロット（対数スケール）
        ax1.loglog(grid_sizes, l2_errors, 'o-', label='L2誤差')
        ax1.loglog(grid_sizes, max_errors, 's-', label='最大誤差')
        ax1.set_xlabel('グリッドサイズ N (NxN)')
        ax1.set_ylabel('誤差')
        ax1.set_title('グリッドサイズと誤差の関係')
        ax1.grid(True)
        ax1.legend()
        
        # 収束率の計算と表示
        convergence_rates_l2 = []
        convergence_rates_max = []
        
        for i in range(1, len(grid_sizes)):
            # 格子幅の比率（h2/h1 = N1/N2）
            ratio = grid_sizes[i] / grid_sizes[i-1]
            
            # 誤差の比率
            error_ratio_l2 = l2_errors[i-1] / l2_errors[i]
            error_ratio_max = max_errors[i-1] / max_errors[i]
            
            # 収束率: p in O(h^p) = log(error_ratio) / log(h_ratio)
            rate_l2 = jnp.log(error_ratio_l2) / jnp.log(ratio)
            rate_max = jnp.log(error_ratio_max) / jnp.log(ratio)
            
            convergence_rates_l2.append(float(rate_l2))
            convergence_rates_max.append(float(rate_max))
        
        # 収束率のバープロット
        bar_x = range(len(convergence_rates_l2))
        bar_labels = [f"{grid_sizes[i]}->{grid_sizes[i+1]}" for i in range(len(convergence_rates_l2))]
        
        width = 0.35
        ax2.bar(bar_x, convergence_rates_l2, width, label='L2誤差の収束率')
        ax2.bar([x + width for x in bar_x], convergence_rates_max, width, label='最大誤差の収束率')
        
        ax2.set_xticks([x + width/2 for x in bar_x])
        ax2.set_xticklabels(bar_labels)
        ax2.set_ylabel('収束率 p (誤差 ~ h^p)')
        ax2.set_title('グリッドサイズの変化に対する収束率')
        ax2.grid(True, axis='y')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"convergence_analysis.png"), dpi=150)
        plt.show()
    
    # 結果の表示
    print("\n=== 収束分析結果 ===")
    print(f"{'グリッドサイズ':<15} {'L2誤差':<15} {'最大誤差':<15}")
    print("-" * 45)
    
    for i, size in enumerate(grid_sizes):
        print(f"{size:<15} {l2_errors[i]:<15.6e} {max_errors[i]:<15.6e}")
    
    print("\n収束率:")
    for i in range(len(convergence_rates_l2)):
        print(f"{grid_sizes[i]}->{grid_sizes[i+1]}: L2率={convergence_rates_l2[i]:.2f}, 最大率={convergence_rates_max[i]:.2f}")


def run_cli():
    """コマンドラインインターフェースの実行"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.out, exist_ok=True)
    
    # コマンドに応じて処理を分岐
    if args.command == "test":
        run_test(args)
    elif args.command == "compare":
        run_comparison(args)
    else:
        print(f"未知のコマンド: {args.command}")


if __name__ == "__main__":
    run_cli()