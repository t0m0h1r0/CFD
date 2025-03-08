# cli.py
import argparse
import os
import cupy as cp
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from grid import Grid
from tester import CCDTester
from test_functions import TestFunctionFactory, TestFunction
from visualization import CCDVisualizer

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="1D CCD Method Implementation")
    
    # 引数の追加
    parser.add_argument('--n-points', type=int, default=21,
                        help='Number of grid points')
    parser.add_argument('--x-range', type=float, nargs=2, default=[-1.0, 1.0],
                        help='Range of x coordinates (min max)')
    parser.add_argument('--test-func', type=str, default='Sin',
                        help='Test function name')
    parser.add_argument('--boundary', type=str, choices=['dirichlet', 'neumann', 'mixed'],
                        default='dirichlet', help='Boundary condition type')
    parser.add_argument('--no-visualization', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--convergence-test', action='store_true',
                        help='Run grid convergence test')
    parser.add_argument('--test-all-functions', action='store_true',
                        help='Run tests for all available test functions')
    parser.add_argument('--rehu-scaling', type=float, default=None, 
                        help='Apply Rehu scaling with the specified Rehu number')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for output filenames')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for output images')
    parser.add_argument('--show', action='store_true',
                        help='Show plots (in addition to saving)')
    
    return parser.parse_args()

def run_convergence_test(
    func_name: str,
    x_range: Tuple[float, float],
    use_dirichlet: bool,
    use_neumann: bool,
    rehu_number: Optional[float],
    prefix: str,
    dpi: int,
    show: bool
):
    """グリッド収束性テストを実行"""
    # テスト関数を選択
    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next((f for f in test_funcs if f.name == func_name), test_funcs[0])
    
    # グリッドサイズ
    grid_sizes = [11, 21, 41, 81, 161]
    
    # 基準グリッドでテスターを作成
    base_grid = Grid(grid_sizes[0], x_range)
    tester = CCDTester(base_grid)
    
    # 収束性テストを実行
    print(f"Running grid convergence test for {selected_func.name} function...")
    print(f"Using {'Dirichlet' if use_dirichlet else ''} "
          f"{'and ' if use_dirichlet and use_neumann else ''}"
          f"{'Neumann' if use_neumann else ''} boundary conditions")
    
    if rehu_number is not None:
        print(f"Applying Rehu scaling with number: {rehu_number}")
    
    results = tester.run_grid_convergence_test(
        selected_func, grid_sizes, x_range, use_dirichlet, use_neumann, rehu_number=rehu_number
    )
    
    # 結果を表示
    print("\nGrid Convergence Results:")
    print(f"{'Grid Size':<10} {'h':<10} {'ψ error':<15} {'ψ\' error':<15} {'ψ\" error':<15} {'ψ\'\" error':<15}")
    print("-" * 80)
    
    for n in grid_sizes:
        h = (x_range[1] - x_range[0]) / (n - 1)
        print(f"{n:<10} {h:<10.6f} {results[n][0]:<15.6e} {results[n][1]:<15.6e} {results[n][2]:<15.6e} {results[n][3]:<15.6e}")
    
    # 可視化
    visualizer = CCDVisualizer()
    visualizer.visualize_grid_convergence(
        selected_func.name, grid_sizes, results,
        prefix=prefix, save=True, show=show, dpi=dpi
    )

def test_all_functions(
    n_points: int,
    x_range: Tuple[float, float],
    use_dirichlet: bool,
    use_neumann: bool,
    rehu_number: Optional[float],
    visualize: bool,
    prefix: str,
    dpi: int,
    show: bool
):
    """全てのテスト関数に対してテストを実行"""
    # テスト関数の取得
    test_funcs = TestFunctionFactory.create_standard_functions()
    
    # 結果を保存する辞書
    results_summary = {}
    
    # グリッドの作成
    grid = Grid(n_points, x_range)
    visualizer = CCDVisualizer() if visualize else None
    
    print(f"\n==== Testing All Functions ({n_points} points) ====")
    print(f"Using {'Dirichlet' if use_dirichlet else ''} "
          f"{'and ' if use_dirichlet and use_neumann else ''}"
          f"{'Neumann' if use_neumann else ''} boundary conditions")
    
    if rehu_number is not None:
        print(f"Applying Rehu scaling with number: {rehu_number}")
    
    print("\n" + "-" * 80)
    print(f"{'Function':<15} {'ψ error':<15} {'ψ\' error':<15} {'ψ\" error':<15} {'ψ\'\" error':<15}")
    print("-" * 80)
    
    # 各関数に対してテストを実行
    for func in test_funcs:
        # テスターの作成
        tester = CCDTester(grid)
        
        # テストの実行
        results = tester.run_test_with_options(
            func, 
            use_dirichlet=use_dirichlet, 
            use_neumann=use_neumann,
            rehu_number=rehu_number
        )
        
        # 結果の表示
        errors = results["errors"]
        print(f"{func.name:<15} {errors[0]:<15.6e} {errors[1]:<15.6e} {errors[2]:<15.6e} {errors[3]:<15.6e}")
        
        # 結果を保存
        results_summary[func.name] = errors
        
        # 可視化
        if visualize:
            visualizer.visualize_derivatives(
                grid, 
                results["function"],
                results["numerical"],
                results["exact"],
                results["errors"],
                prefix=f"{prefix}_{func.name.lower()}" if prefix else func.name.lower(),
                save=True,
                show=show,
                dpi=dpi
            )
    
    print("-" * 80)
    
    # すべての関数の誤差を比較するグラフを生成
    if visualize:
        compare_all_functions_errors(
            results_summary, 
            prefix=prefix, 
            dpi=dpi, 
            show=show
        )
    
    return results_summary

def compare_all_functions_errors(
    results_summary: Dict[str, List[float]],
    prefix: str = "",
    dpi: int = 150,
    show: bool = False
):
    """全テスト関数の誤差比較グラフを生成"""
    # 出力ディレクトリの確認
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 関数名とエラーデータを抽出
    func_names = list(results_summary.keys())
    
    # 比較グラフの作成
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    error_types = ["ψ", "ψ'", "ψ''", "ψ'''"]
    
    for i, (ax, error_type) in enumerate(zip(axes.flat, error_types)):
        errors = [results_summary[name][i] for name in func_names]
        
        # 誤差の対数棒グラフ
        bars = ax.bar(func_names, errors)
        
        # グラフの装飾
        ax.set_yscale('log')
        ax.set_title(f"{error_type} Error Comparison")
        ax.set_xlabel("Test Function")
        ax.set_ylabel("Error (log scale)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # ラベルが重ならないように回転
        ax.set_xticklabels(func_names, rotation=45, ha='right')
        
        # 値をバーの上に表示
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # オフセット
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    filename = f"{output_dir}/{prefix}_all_functions_comparison.png" if prefix else f"{output_dir}/all_functions_comparison.png"
    plt.savefig(filename, dpi=dpi)
    print(f"All functions comparison plot saved to {filename}")
    
    # 表示
    if show:
        plt.show()
    else:
        plt.close(fig)

def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    
    # 境界条件の設定
    use_dirichlet = args.boundary in ['dirichlet', 'mixed']
    use_neumann = args.boundary in ['neumann', 'mixed']
    
    # 全関数テスト
    if args.test_all_functions:
        test_all_functions(
            args.n_points, 
            tuple(args.x_range),
            use_dirichlet, 
            use_neumann,
            args.rehu_scaling,
            not args.no_visualization,
            args.prefix,
            args.dpi,
            args.show
        )
        return
    
    # 収束性テスト
    if args.convergence_test:
        run_convergence_test(
            args.test_func, 
            tuple(args.x_range),
            use_dirichlet, 
            use_neumann,
            args.rehu_scaling,
            args.prefix, 
            args.dpi, 
            args.show
        )
        return
    
    # 通常のテスト（単一関数）
    # グリッドの作成
    grid = Grid(args.n_points, tuple(args.x_range))
    
    # テスターの作成
    tester = CCDTester(grid)
    
    # テスト関数の選択
    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next((f for f in test_funcs if f.name == args.test_func), test_funcs[0])
    
    # テストの実行
    print(f"\nRunning test with {selected_func.name} function...")
    print(f"Using {'Dirichlet' if use_dirichlet else ''} "
          f"{'and ' if use_dirichlet and use_neumann else ''}"
          f"{'Neumann' if use_neumann else ''} boundary conditions")
    
    if args.rehu_scaling is not None:
        print(f"Applying Rehu scaling with number: {args.rehu_scaling}")
    
    results = tester.run_test_with_options(
        selected_func, 
        use_dirichlet=use_dirichlet, 
        use_neumann=use_neumann,
        rehu_number=args.rehu_scaling
    )
    
    # 結果の表示
    print("\nError Analysis:")
    print(f"  ψ error: {results['errors'][0]:.6e}")
    print(f"  ψ' error: {results['errors'][1]:.6e}")
    print(f"  ψ'' error: {results['errors'][2]:.6e}")
    print(f"  ψ''' error: {results['errors'][3]:.6e}")
    
    # 可視化
    if not args.no_visualization:
        visualizer = CCDVisualizer()
        visualizer.visualize_derivatives(
            grid, 
            results["function"],
            results["numerical"],
            results["exact"],
            results["errors"],
            prefix=args.prefix,
            save=True,
            show=args.show,
            dpi=args.dpi
        )

if __name__ == "__main__":
    run_cli()