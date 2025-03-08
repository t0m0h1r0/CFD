# cli.py
import argparse
import os
import cupy as cp
from typing import List, Tuple
from grid import Grid
from tester import CCDTester
from test_functions import TestFunctionFactory
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
    results = tester.run_grid_convergence_test(
        selected_func, grid_sizes, x_range, use_dirichlet, use_neumann
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

def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    
    # 収束性テスト
    if args.convergence_test:
        # 境界条件の設定
        use_dirichlet = args.boundary in ['dirichlet', 'mixed']
        use_neumann = args.boundary in ['neumann', 'mixed']
        
        run_convergence_test(
            args.test_func, tuple(args.x_range),
            use_dirichlet, use_neumann,
            args.prefix, args.dpi, args.show
        )
        return
    
    # グリッドの作成
    grid = Grid(args.n_points, tuple(args.x_range))
    
    # テスターの作成
    tester = CCDTester(grid)
    
    # テスト関数の選択
    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next((f for f in test_funcs if f.name == args.test_func), test_funcs[0])
    
    # 境界条件の設定
    use_dirichlet = args.boundary in ['dirichlet', 'mixed']
    use_neumann = args.boundary in ['neumann', 'mixed']
    
    # テストの実行
    print(f"Running test with {selected_func.name} function...")
    print(f"Using {'Dirichlet' if use_dirichlet else ''} "
          f"{'and ' if use_dirichlet and use_neumann else ''}"
          f"{'Neumann' if use_neumann else ''} boundary conditions")
    
    results = tester.run_test_with_options(
        selected_func, use_dirichlet=use_dirichlet, use_neumann=use_neumann
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