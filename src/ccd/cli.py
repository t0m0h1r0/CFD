import argparse
import os
import time
from grid import Grid
from tester import CCDTester1D, CCDTester2D
from test_functions import TestFunctionFactory
from visualization1d import CCDVisualizer
from visualization2d import CCD2DVisualizer
from scaling import plugin_manager

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="CCD法ソルバー")
    
    # 基本設定
    parser.add_argument("--dim", type=int, choices=[1, 2], default=1, help="次元 (1 or 2)")
    parser.add_argument("--func", type=str, default="Sine", help="テスト関数名")
    parser.add_argument("-e","--equation", type=str, default="poisson", help="方程式セット名")
    parser.add_argument("-o", "--out", type=str, default="results", help="出力ディレクトリ")
    parser.add_argument("--prefix", type=str, default="", help="出力ファイル名の接頭辞")
    
    # グリッド設定
    parser.add_argument("--nx", type=int, default=21, help="x方向の格子点数")
    parser.add_argument("--ny", type=int, default=21, help="y方向の格子点数 (2Dのみ)")
    parser.add_argument("--xrange", type=float, nargs=2, default=[-1.0, 1.0], help="x方向の範囲")
    parser.add_argument("--yrange", type=float, nargs=2, default=[-1.0, 1.0], help="y方向の範囲 (2Dのみ)")
    
    # ソルバー設定
    parser.add_argument("--solver", type=str, choices=['direct', 'gmres', 'cg', 'cgs'], 
                      default='direct', help="ソルバー手法")
    parser.add_argument("--scaling", type=str, default=None, help="スケーリング手法")
    parser.add_argument("--analyze", action="store_true", help="行列を分析する")
    parser.add_argument("--monitor", action="store_true", help="収束過程をモニタリングする")
    
    # テストモード
    parser.add_argument("--list", action="store_true", help="利用可能な関数一覧を表示")
    parser.add_argument("--list-scaling", action="store_true", help="スケーリング手法一覧を表示")
    parser.add_argument("--converge", action="store_true", help="格子収束テスト実行")
    parser.add_argument("--all", action="store_true", help="全関数でテスト実行")
    parser.add_argument("--compare-scaling", action="store_true", help="スケーリング手法比較")
    
    return parser.parse_args()

def create_tester(args):
    """引数に基づいてテスターを作成"""
    # グリッドとテスター作成
    if args.dim == 1:
        grid = Grid(args.nx, x_range=tuple(args.xrange))
        tester = CCDTester1D(grid)
    else:
        grid = Grid(args.nx, args.ny, x_range=tuple(args.xrange), y_range=tuple(args.yrange))
        tester = CCDTester2D(grid)
    
    # テスター設定
    solver_options = {
        "tol": 1e-10, 
        "maxiter": 1000,
        "monitor_convergence": args.monitor,
        "output_dir": args.out,
        "prefix": args.prefix
    }
    tester.set_solver_options(args.solver, solver_options, args.analyze)
    tester.scaling_method = args.scaling
    tester.set_equation_set(args.equation)
    
    return tester

def list_functions(dim):
    """利用可能な関数一覧表示"""
    functions = (TestFunctionFactory.create_standard_1d_functions() if dim == 1 
              else TestFunctionFactory.create_standard_2d_functions())
    print(f"\n利用可能な{dim}D関数:")
    for func in functions:
        print(f"- {func.name}")

def list_scaling_methods():
    """利用可能なスケーリング手法一覧表示"""
    plugins = plugin_manager.get_available_plugins()
    print("\n利用可能なスケーリング手法:")
    for name in plugins:
        print(f"- {name}")

def compare_scaling(args):
    """スケーリング手法の比較"""
    tester = create_tester(args)
    test_func = tester.get_test_function(args.func)
    scaling_methods = plugin_manager.get_available_plugins()
    
    print(f"\n{test_func.name}関数でスケーリング手法を比較中...")
    print(f"{'手法':<15} {'時間(秒)':<10} {'反復回数':<10} {'最大誤差':<15}")
    print("-" * 50)
    
    for method in scaling_methods:
        tester.scaling_method = method
        start_time = time.time()
        result = tester.run_test_with_options(test_func)
        elapsed = time.time() - start_time
        
        iterations = getattr(tester.solver, 'last_iterations', "N/A")
        max_error = max(result['errors'])
        
        print(f"{method:<15} {elapsed:<10.4f} {iterations!s:<10} {max_error:<15.6e}")

def run_convergence_test(args):
    """格子収束性テスト実行"""
    tester = create_tester(args)
    test_func = tester.get_test_function(args.func)
    
    # 収束テスト用グリッドサイズ
    grid_sizes = [11, 21, 41, 81] if args.dim == 1 else [11, 21, 31, 41]
    
    print(f"\n{test_func.name}関数で格子収束性テスト実行中...")
    results = tester.run_grid_convergence_test(
        test_func, grid_sizes, args.xrange, 
        args.yrange if args.dim == 2 else None
    )
    
    # 可視化
    visualizer = CCDVisualizer(output_dir=args.out) if args.dim == 1 else CCD2DVisualizer(output_dir=args.out)
    visualizer.visualize_grid_convergence(test_func.name, grid_sizes, results, prefix=args.prefix)

def test_all_functions(args):
    """全関数テスト実行"""
    tester = create_tester(args)
    functions = (TestFunctionFactory.create_standard_1d_functions() if args.dim == 1 
              else TestFunctionFactory.create_standard_2d_functions())
    visualizer = CCDVisualizer(output_dir=args.out) if args.dim == 1 else CCD2DVisualizer(output_dir=args.out)
    
    print(f"\n全{args.dim}D関数テスト実行中...")
    results = {}
    
    for func in functions:
        print(f"- {func.name}テスト中...")
        result = tester.run_test_with_options(func)
        results[func.name] = result["errors"]
        
        if args.dim == 1:
            visualizer.visualize_derivatives(
                tester.grid, func.name, result["numerical"],
                result["exact"], result["errors"], prefix=args.prefix
            )
        else:
            visualizer.visualize_solution(
                tester.grid, func.name, result["numerical"],
                result["exact"], result["errors"], prefix=args.prefix
            )
    
    # 全関数比較
    visualizer.compare_all_functions_errors(results, grid_size=args.nx, prefix=args.prefix)

def run_single_test(args):
    """単一関数テスト実行"""
    tester = create_tester(args)
    test_func = tester.get_test_function(args.func)
    
    print(f"\n{test_func.name}関数でテスト実行中...")
    start_time = time.time()
    result = tester.run_test_with_options(test_func)
    elapsed = time.time() - start_time
    
    # 結果表示
    print(f"\n実行時間: {elapsed:.4f}秒")
    
    # 可視化
    if args.dim == 1:
        visualizer = CCDVisualizer(output_dir=args.out)
        visualizer.visualize_derivatives(
            tester.grid, test_func.name, result["numerical"],
            result["exact"], result["errors"], prefix=args.prefix
        )
    else:
        visualizer = CCD2DVisualizer(output_dir=args.out)
        visualizer.visualize_solution(
            tester.grid, test_func.name, result["numerical"],
            result["exact"], result["errors"], prefix=args.prefix
        )

def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    # 一覧表示機能
    if args.list:
        list_functions(args.dim)
        return
    
    if args.list_scaling:
        list_scaling_methods()
        return
    
    # テストモード選択実行
    if args.compare_scaling:
        compare_scaling(args)
    elif args.converge:
        run_convergence_test(args)
    elif args.all:
        test_all_functions(args)
    else:
        run_single_test(args)

if __name__ == "__main__":
    run_cli()