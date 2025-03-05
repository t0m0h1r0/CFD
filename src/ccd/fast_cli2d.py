#!/usr/bin/env python3
"""
高速スパース2次元CCDソルバーのコマンドラインインターフェース

事前計算された逆行列を用いて、線形時間で複数の問題を解くためのインターフェース。
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, Optional, List

# インポートパスを追加（必要に応じて変更）
sys.path.append('.')
sys.path.append('..')

from grid_2d_config import Grid2DConfig
from fast_ccd2d_solver import FastSparseCCD2DSolver


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="高速スパース2次元CCD法ソルバー")

    # 親パーサー（共通オプション）
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--nx", type=int, default=64, help="x方向のグリッド点数")
    parent_parser.add_argument("--ny", type=int, default=64, help="y方向のグリッド点数")
    parent_parser.add_argument("--xrange", type=float, nargs=2, default=[0.0, 1.0], help="x軸の範囲 (開始点 終了点)")
    parent_parser.add_argument("--yrange", type=float, nargs=2, default=[0.0, 1.0], help="y軸の範囲 (開始点 終了点)")
    parent_parser.add_argument("--coeffs", type=float, nargs="+", default=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 
                              help="係数 [a, b, c, d, e, f]: a*u + b*ux + c*uxx + d*uy + e*uyy + f*uxy")
    parent_parser.add_argument("--out", type=str, default="results", help="出力ディレクトリ")
    parent_parser.add_argument("--no-viz", action="store_true", help="可視化を無効化")

    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド", required=True)

    # 単一問題のテスト
    test_parser = subparsers.add_parser("test", parents=[parent_parser], help="単一のポアソン方程式を解く")
    
    # パラメータスイープと収束分析
    sweep_parser = subparsers.add_parser("sweep", parents=[parent_parser], help="パラメータスイープを実行")
    sweep_parser.add_argument("--freq", type=float, default=1.0, help="基本周波数（πの倍数）")
    sweep_parser.add_argument("--freqs", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0], help="周波数のリスト（πの倍数）")
    sweep_parser.add_argument("--count", type=int, default=5, help="各周波数での反復回数")
    
    # タイミング分析
    time_parser = subparsers.add_parser("time", parents=[parent_parser], help="速度分析を実行")
    time_parser.add_argument("--n-problems", type=int, default=10, help="解く問題の数")
    time_parser.add_argument("--no-precompute", action="store_true", help="事前計算なしで解く")

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
    """単一のポアソン方程式を解くテスト"""
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # グリッド点の生成
    x = np.linspace(args.xrange[0], args.xrange[1], args.nx)
    y = np.linspace(args.yrange[0], args.yrange[1], args.ny)
    X, Y = np.meshgrid(x, y)
    
    # 右辺関数 f(x,y) = sin(πx)sin(πy)
    pi = np.pi
    f_values = np.sin(pi * X) * np.sin(pi * Y)
    
    # 解析解 u(x,y) = -sin(πx)sin(πy)/(2π²)
    u_exact = -f_values / (2 * pi**2)
    
    # 境界条件を設定（解析解に合わせる）
    grid_config.dirichlet_values = {
        'left': u_exact[:, 0],
        'right': u_exact[:, -1],
        'bottom': u_exact[0, :],
        'top': u_exact[-1, :]
    }
    
    # 係数をラプラシアン用に設定
    grid_config.coeffs = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # ∇²u = uxx + uyy
    
    # ソルバーを初期化
    print("\n高速スパースCCDソルバーを初期化中...")
    solver = FastSparseCCD2DSolver(grid_config, precompute=True)
    
    # 方程式を解く
    print(f"\nポアソン方程式 ∇²u = f を解いています...")
    solution = solver.solve(f_values)
    u_numerical = solution['u']
    
    # 誤差計算
    error = np.abs(u_numerical - u_exact)
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)
    
    print(f"ポアソン方程式の解析終了:")
    print(f"L2誤差: {l2_error:.6e}")
    print(f"最大誤差: {max_error:.6e}")
    
    # 可視化
    if not args.no_viz:
        # プロットの作成
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 数値解
        im0 = axes[0].pcolormesh(X, Y, u_numerical, cmap='viridis', shading='auto')
        axes[0].set_title('数値解')
        fig.colorbar(im0, ax=axes[0])
        
        # 厳密解
        im1 = axes[1].pcolormesh(X, Y, u_exact, cmap='viridis', shading='auto')
        axes[1].set_title('厳密解')
        fig.colorbar(im1, ax=axes[1])
        
        # 誤差
        im2 = axes[2].pcolormesh(X, Y, error, cmap='hot', shading='auto')
        axes[2].set_title(f'誤差 (L2: {l2_error:.2e}, Max: {max_error:.2e})')
        fig.colorbar(im2, ax=axes[2])
        
        # すべてのプロットに座標軸ラベルを設定
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        # プロットを保存
        os.makedirs(args.out, exist_ok=True)
        plt.tight_layout()
        filename = f"fast_poisson_{args.nx}x{args.ny}.png"
        plt.savefig(os.path.join(args.out, filename), dpi=150)
        plt.show()
    
    return u_numerical, u_exact, l2_error, max_error


def run_parameter_sweep(args):
    """周波数パラメータのスイープを実行"""
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # グリッド点の生成
    x = np.linspace(args.xrange[0], args.xrange[1], args.nx)
    y = np.linspace(args.yrange[0], args.yrange[1], args.ny)
    X, Y = np.meshgrid(x, y)
    
    # 基本周波数
    pi = np.pi
    
    # 結果を格納する辞書
    results = {
        'freqs': args.freqs,
        'l2_errors': [],
        'max_errors': []
    }
    
    # 各周波数でのテスト問題を準備
    test_problems = []
    for freq in args.freqs:
        for _ in range(args.count):
            # 異なる位相のsin波を生成
            phase_x = np.random.uniform(0, 2*pi)
            phase_y = np.random.uniform(0, 2*pi)
            
            # 右辺関数 f(x,y) = sin(k*π*x + φx)sin(k*π*y + φy)
            f_values = np.sin(freq*pi*X + phase_x) * np.sin(freq*pi*Y + phase_y)
            
            # 解析解 u(x,y) = -f(x,y)/(2k²π²)
            u_exact = -f_values / (2 * (freq*pi)**2)
            
            test_problems.append({
                'f': f_values,
                'u_exact': u_exact,
                'freq': freq
            })
    
    # ソルバーを初期化
    print("\n高速スパースCCDソルバーを初期化中...")
    
    # 最初の問題の境界条件を設定
    first_problem = test_problems[0]
    grid_config.dirichlet_values = {
        'left': first_problem['u_exact'][:, 0],
        'right': first_problem['u_exact'][:, -1],
        'bottom': first_problem['u_exact'][0, :],
        'top': first_problem['u_exact'][-1, :]
    }
    
    # 係数をラプラシアン用に設定
    grid_config.coeffs = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # ∇²u = uxx + uyy
    
    # ソルバーの初期化と事前計算
    solver = FastSparseCCD2DSolver(grid_config, precompute=True)
    
    # 各周波数ごとにまとめて解く
    all_solutions = []
    freq_index = 0
    current_freq = args.freqs[0]
    problems_for_freq = []
    
    for problem in test_problems:
        if problem['freq'] != current_freq:
            # 周波数が変わったら、前の周波数の問題をまとめて解く
            print(f"\n周波数 {current_freq}π の問題を解いています...")
            
            # 前の問題をまとめて解く
            freq_solutions = []
            
            # 各問題の右辺関数リストを作成
            f_list = [p['f'] for p in problems_for_freq]
            
            # 境界条件を更新
            for i, p in enumerate(problems_for_freq):
                if i == 0:  # 最初の問題の境界条件を設定
                    grid_config.dirichlet_values = {
                        'left': p['u_exact'][:, 0],
                        'right': p['u_exact'][:, -1],
                        'bottom': p['u_exact'][0, :],
                        'top': p['u_exact'][-1, :]
                    }
                    
                    # システム行列を再構築
                    solver.setup_system_matrix()
            
            # 一括して解く
            solutions = solver.solve_multiple(f_list)
            
            # 誤差計算
            for sol, prob in zip(solutions, problems_for_freq):
                u_numerical = sol['u']
                u_exact = prob['u_exact']
                
                error = np.abs(u_numerical - u_exact)
                l2_error = np.sqrt(np.mean(error**2))
                max_error = np.max(error)
                
                freq_solutions.append({
                    'u_numerical': u_numerical,
                    'u_exact': u_exact,
                    'l2_error': l2_error,
                    'max_error': max_error
                })
            
            all_solutions.append({
                'freq': current_freq,
                'solutions': freq_solutions
            })
            
            # 次の周波数の準備
            current_freq = problem['freq']
            problems_for_freq = [problem]
            freq_index += 1
        else:
            # 同じ周波数なら問題をリストに追加
            problems_for_freq.append(problem)
    
    # 最後の周波数の問題を処理
    if problems_for_freq:
        print(f"\n周波数 {current_freq}π の問題を解いています...")
        
        # 境界条件を更新
        for i, p in enumerate(problems_for_freq):
            if i == 0:  # 最初の問題の境界条件を設定
                grid_config.dirichlet_values = {
                    'left': p['u_exact'][:, 0],
                    'right': p['u_exact'][:, -1],
                    'bottom': p['u_exact'][0, :],
                    'top': p['u_exact'][-1, :]
                }
                
                # システム行列を再構築
                solver.setup_system_matrix()
        
        # 各問題の右辺関数リストを作成
        f_list = [p['f'] for p in problems_for_freq]
        
        # 一括して解く
        solutions = solver.solve_multiple(f_list)
        
        # 誤差計算
        freq_solutions = []
        for sol, prob in zip(solutions, problems_for_freq):
            u_numerical = sol['u']
            u_exact = prob['u_exact']
            
            error = np.abs(u_numerical - u_exact)
            l2_error = np.sqrt(np.mean(error**2))
            max_error = np.max(error)
            
            freq_solutions.append({
                'u_numerical': u_numerical,
                'u_exact': u_exact,
                'l2_error': l2_error,
                'max_error': max_error
            })
        
        all_solutions.append({
            'freq': current_freq,
            'solutions': freq_solutions
        })
    
    # 各周波数での平均誤差を計算
    print("\n=== 周波数別の誤差分析 ===")
    print(f"{'周波数':<10} {'L2誤差':<15} {'最大誤差':<15}")
    print("-" * 40)
    
    for freq_data in all_solutions:
        freq = freq_data['freq']
        solutions = freq_data['solutions']
        
        # 平均誤差を計算
        avg_l2_error = np.mean([sol['l2_error'] for sol in solutions])
        avg_max_error = np.mean([sol['max_error'] for sol in solutions])
        
        print(f"{freq:π<10} {avg_l2_error:<15.6e} {avg_max_error:<15.6e}")
        
        # 結果を保存
        results['l2_errors'].append(avg_l2_error)
        results['max_errors'].append(avg_max_error)
    
    # 可視化
    if not args.no_viz:
        # 保存先ディレクトリを作成
        os.makedirs(args.out, exist_ok=True)
        
        # 周波数と誤差の関係を可視化
        plt.figure(figsize=(10, 6))
        plt.loglog(results['freqs'], results['l2_errors'], 'o-', label='L2誤差')
        plt.loglog(results['freqs'], results['max_errors'], 's-', label='最大誤差')
        plt.xlabel('周波数 (πの倍数)')
        plt.ylabel('誤差')
        plt.title(f'周波数と誤差の関係 ({args.nx}×{args.ny}グリッド)')
        plt.grid(True)
        plt.legend()
        
        # プロットを保存
        filename = f"freq_error_analysis_{args.nx}x{args.ny}.png"
        plt.savefig(os.path.join(args.out, filename), dpi=150)
        plt.show()
    
    return results


def run_timing_analysis(args):
    """速度分析を実行"""
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # グリッド点の生成
    x = np.linspace(args.xrange[0], args.xrange[1], args.nx)
    y = np.linspace(args.yrange[0], args.yrange[1], args.ny)
    X, Y = np.meshgrid(x, y)
    
    # 基本周波数
    pi = np.pi
    
    # テスト問題を生成
    test_problems = []
    for i in range(args.n_problems):
        # 異なる周波数と位相のsin波を生成
        freq = 1.0 + i * 0.5  # 1.0, 1.5, 2.0, ...
        phase_x = np.random.uniform(0, 2*pi)
        phase_y = np.random.uniform(0, 2*pi)
        
        # 右辺関数 f(x,y) = sin(k*π*x + φx)sin(k*π*y + φy)
        f_values = np.sin(freq*pi*X + phase_x) * np.sin(freq*pi*Y + phase_y)
        
        # 解析解 u(x,y) = -f(x,y)/(2k²π²)
        u_exact = -f_values / (2 * (freq*pi)**2)
        
        test_problems.append({
            'f': f_values,
            'u_exact': u_exact
        })
    
    # 境界条件を設定（最初の問題に合わせる）
    grid_config.dirichlet_values = {
        'left': test_problems[0]['u_exact'][:, 0],
        'right': test_problems[0]['u_exact'][:, -1],
        'bottom': test_problems[0]['u_exact'][0, :],
        'top': test_problems[0]['u_exact'][-1, :]
    }
    
    # 係数をラプラシアン用に設定
    grid_config.coeffs = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # ∇²u = uxx + uyy
    
    # 時間計測結果
    timings = {
        'precompute': None,
        'no_precompute': None,
        'per_problem_precompute': None,
        'per_problem_no_precompute': None
    }
    
    # 事前計算あり
    if not args.no_precompute:
        print("\n事前計算ありのタイミング分析:")
        
        # 初期化とシステム行列の事前計算
        start_time_init = time.time()
        solver_precompute = FastSparseCCD2DSolver(grid_config, precompute=True)
        precompute_time = time.time() - start_time_init
        print(f"初期化と事前計算: {precompute_time:.2f}秒")
        
        # 問題を解く
        start_time_solve = time.time()
        for i, problem in enumerate(test_problems):
            solver_precompute.solve(problem['f'])
            if (i + 1) % 5 == 0 or i == len(test_problems) - 1:
                print(f"  {i + 1}/{len(test_problems)} 問題を解きました")
        
        solve_time = time.time() - start_time_solve
        total_time = precompute_time + solve_time
        per_problem = solve_time / len(test_problems)
        
        print(f"解く時間: {solve_time:.2f}秒 (平均: {per_problem:.2f}秒/問題)")
        print(f"合計時間: {total_time:.2f}秒")
        
        timings['precompute'] = total_time
        timings['per_problem_precompute'] = per_problem
    
    # 事前計算なし
    print("\n事前計算なしのタイミング分析:")
    
    # 初期化（事前計算なし）
    start_time_init = time.time()
    solver_no_precompute = FastSparseCCD2DSolver(grid_config, precompute=False)
    init_time = time.time() - start_time_init
    print(f"初期化: {init_time:.2f}秒")
    
    # 問題を解く
    start_time_solve = time.time()
    for i, problem in enumerate(test_problems):
        solver_no_precompute.solve(problem['f'])
        if (i + 1) % 5 == 0 or i == len(test_problems) - 1:
            print(f"  {i + 1}/{len(test_problems)} 問題を解きました")
    
    solve_time = time.time() - start_time_solve
    total_time = init_time + solve_time
    per_problem = solve_time / len(test_problems)
    
    print(f"解く時間: {solve_time:.2f}秒 (平均: {per_problem:.2f}秒/問題)")
    print(f"合計時間: {total_time:.2f}秒")
    
    timings['no_precompute'] = total_time
    timings['per_problem_no_precompute'] = per_problem
    
    # 結果を可視化
    if not args.no_viz and not args.no_precompute:
        # 保存先ディレクトリを作成
        os.makedirs(args.out, exist_ok=True)
        
        # タイミング比較のバープロット
        plt.figure(figsize=(10, 6))
        
        # 合計時間の比較
        plt.subplot(1, 2, 1)
        bars = plt.bar(['事前計算あり', '事前計算なし'], 
                     [timings['precompute'], timings['no_precompute']])
        
        # バーに値を表示
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}s', ha='center', va='bottom')
        
        plt.ylabel('合計実行時間 (秒)')
        plt.title('合計実行時間の比較')
        
        # 1問題あたりの時間比較
        plt.subplot(1, 2, 2)
        bars = plt.bar(['事前計算あり', '事前計算なし'], 
                     [timings['per_problem_precompute'], timings['per_problem_no_precompute']])
        
        # バーに値を表示
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.ylabel('1問題あたりの時間 (秒)')
        plt.title('1問題あたりの実行時間')
        
        plt.tight_layout()
        
        # プロットを保存
        filename = f"timing_analysis_{args.nx}x{args.ny}_{args.n_problems}problems.png"
        plt.savefig(os.path.join(args.out, filename), dpi=150)
        plt.show()
    
    return timings


def run_cli():
    """コマンドラインインターフェースの実行"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.out, exist_ok=True)
    
    # コマンドに応じて処理を分岐
    if args.command == "test":
        run_test(args)
    elif args.command == "sweep":
        run_parameter_sweep(args)
    elif args.command == "time":
        run_timing_analysis(args)
    else:
        print(f"未知のコマンド: {args.command}")


if __name__ == "__main__":
    run_cli()