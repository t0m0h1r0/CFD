import argparse
import os
import time
from grid import Grid
from tester import CCDTester1D, CCDTester2D
from test_functions import TestFunctionFactory
from visualization1d import CCDVisualizer
from visualization2d import CCD2DVisualizer

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="CCD法の統合CLI (1D/2D)")
    
    # 共通基本オプション
    common = parser.add_argument_group('共通オプション')
    common.add_argument("--dim", type=int, choices=[1, 2], default=1, help="問題の次元")
    common.add_argument("--test-func", type=str, default="Sine", help="テスト関数名")
    common.add_argument("--no-visualization", action="store_true", help="可視化を無効化")
    common.add_argument("--convergence-test", action="store_true", help="格子収束性テスト実行")
    common.add_argument("--test-all-functions", action="store_true", help="全関数でテスト実行")
    common.add_argument("--prefix", type=str, default="", help="出力ファイル名の接頭辞")
    common.add_argument("--output-dir", "-o", type=str, default="results", help="出力ディレクトリ")
    common.add_argument("--list-functions", action="store_true", help="関数一覧表示")
    common.add_argument("--x-range", type=float, nargs=2, default=[-1.0, 1.0], help="x座標範囲")
    
    # グリッドオプション
    grid = parser.add_argument_group('グリッドオプション')
    grid.add_argument("--nx-points", type=int, default=21, help="x方向格子点数")
    grid.add_argument("--ny-points", type=int, default=21, help="y方向格子点数 (2D専用)")
    grid.add_argument("--y-range", type=float, nargs=2, default=[-1.0, 1.0], help="y座標範囲 (2D専用)")
    
    # 方程式セット
    eqs = parser.add_argument_group('方程式セットオプション')
    eqs.add_argument("--equation-set", type=str, default="poisson", help="方程式セット名")
    
    # ソルバーオプション
    solver = parser.add_argument_group('ソルバーオプション')
    solver.add_argument("--solver", type=str, 
                     choices=['direct', 'gmres', 'cg', 'cgs', 'lsqr', 'lsmr', 'minres'],
                     default='direct', help="ソルバー方式")
    solver.add_argument("--solver-tol", type=float, default=1e-10, help="収束許容誤差")
    solver.add_argument("--solver-maxiter", type=int, default=1000, help="最大反復回数")
    solver.add_argument("--solver-restart", type=int, default=100, help="GMRESリスタート値")
    solver.add_argument("--no-preconditioner", action="store_true", help="前処理を無効化")
    solver.add_argument("--analyze-matrix", action="store_true", help="行列の疎性を分析")
    solver.add_argument("--monitor-convergence", action="store_true", help="収束過程をモニタリング")
    solver.add_argument("--display-interval", type=int, default=10, help="収束表示間隔")
    
    # スケーリング
    scaling = parser.add_argument_group('スケーリングオプション')
    scaling.add_argument("--scaling", type=str, default=None, help="スケーリング手法名")
    scaling.add_argument("--list-scaling", action="store_true", help="スケーリング手法一覧表示")
    scaling.add_argument("--compare-scaling", action="store_true", help="スケーリング手法比較")
    
    return parser.parse_args()

def get_solver_options(args):
    """ソルバーオプションの取得"""
    return {
        "tol": args.solver_tol,
        "maxiter": args.solver_maxiter,
        "restart": args.solver_restart,
        "use_preconditioner": not args.no_preconditioner,
        "monitor_convergence": args.monitor_convergence,
        "display_interval": args.display_interval,
        "output_dir": args.output_dir,
        "prefix": args.prefix
    }

def list_scaling_methods():
    """利用可能なスケーリング手法一覧表示"""
    from scaling import plugin_manager
    plugins = plugin_manager.get_available_plugins()
    
    print("\n利用可能なスケーリング手法:")
    for i, name in enumerate(plugins, 1):
        scaler = plugin_manager.get_plugin(name)
        print(f"{i:2d}. {name}: {scaler.description}")
    print()

def list_available_functions(dim=1):
    """利用可能なテスト関数一覧表示"""
    if dim == 1:
        functions = TestFunctionFactory.create_standard_1d_functions()
        print("\n利用可能な1Dテスト関数:")
    else:
        functions = TestFunctionFactory.create_standard_2d_functions()
        print("\n利用可能な2Dテスト関数:")
    
    for i, func in enumerate(functions, 1):
        print(f"{i:2d}. {func.name}")
    
    if dim == 2:
        print("\n注: 1次元関数名を指定すると、自動的にテンソル積拡張された2次元関数が使用されます。")

def create_tester(args):
    """引数からテスターを作成"""
    x_range = tuple(args.x_range)
    
    if args.dim == 1:
        grid = Grid(args.nx_points, x_range=x_range)
        return CCDTester1D(grid)
    else:
        y_range = tuple(args.y_range)
        grid = Grid(args.nx_points, args.ny_points, x_range=x_range, y_range=y_range)
        return CCDTester2D(grid)

def setup_tester(args, tester):
    """テスターを設定"""
    solver_options = get_solver_options(args)
    tester.set_solver_options(args.solver, solver_options, args.analyze_matrix)
    tester.scaling_method = args.scaling
    tester.set_equation_set(args.equation_set)
    return tester

def compare_scaling_methods(args):
    """異なるスケーリング手法の性能を比較"""
    from scaling import plugin_manager
    scaling_methods = plugin_manager.get_available_plugins()
    
    tester = create_tester(args)
    test_func = tester.get_test_function(args.test_func)
    solver_options = get_solver_options(args)
    
    print(f"\n{test_func.name}関数でスケーリング手法を比較...")
    if args.dim == 1:
        print(f"グリッドサイズ: {args.nx_points} 点")
    else:
        print(f"グリッドサイズ: {args.nx_points}x{args.ny_points} 点")
    print(f"ソルバー: {args.solver}")
    
    print("\n" + "-" * 80)
    print(f"{'スケーリング手法':<20} {'実行時間 (s)':<12} {'反復回数':<10} {'誤差':<12}")
    print("-" * 80)
    
    for method in scaling_methods:
        tester.set_solver_options(args.solver, solver_options, False)
        tester.scaling_method = method
        tester.set_equation_set(args.equation_set)
        
        start_time = time.time()
        result = tester.run_test_with_options(test_func)
        end_time = time.time()
        
        iter_count = getattr(tester.solver, 'last_iterations', None)
        iter_str = str(iter_count) if iter_count is not None else "N/A"
        max_error = max(result['errors'])
        
        print(f"{method:<20} {end_time - start_time:<12.4f} {iter_str:<10} {max_error:<12.6e}")
    
    print("-" * 80)

def run_convergence_test(args):
    """格子収束性テストを実行"""
    tester = create_tester(args)
    tester = setup_tester(args, tester)
    
    # グリッドサイズ
    if args.dim == 1:
        grid_sizes = [11, 21, 41, 81, 161]
        y_range = None
    else:
        grid_sizes = [11, 21, 31, 41]
        y_range = tuple(args.y_range)
    
    # テスト関数の取得
    test_func = tester.get_test_function(args.test_func)
    
    print(f"{test_func.name}関数の格子収束性テスト実行中...")
    print(f"ソルバー: {args.solver}, 方程式セット: {args.equation_set}")
    print(f"スケーリング: {args.scaling if args.scaling else 'なし'}")
    
    results = tester.run_grid_convergence_test(test_func, grid_sizes, tuple(args.x_range), y_range)
    
    # 結果表示
    if args.dim == 1:
        print("\n格子収束性テスト結果:")
        print(f"{'サイズ':<8} {'h':<8} {'ψ誤差':<12} {'ψ\'誤差':<12} {'ψ\"誤差':<12} {'ψ\'\"誤差':<12}")
        print("-" * 70)
        
        for n in grid_sizes:
            h = (args.x_range[1] - args.x_range[0]) / (n - 1)
            print(f"{n:<8} {h:<8.6f} {results[n][0]:<12.6e} {results[n][1]:<12.6e} "
                  f"{results[n][2]:<12.6e} {results[n][3]:<12.6e}")
        
        # 可視化
        visualizer = CCDVisualizer(output_dir=args.output_dir)
        visualizer.visualize_grid_convergence(test_func.name, grid_sizes, results, 
                                             prefix=args.prefix, save=True)
    else:
        print("\n格子収束性テスト結果:")
        headers = ["サイズ", "h", "ψ誤差", "ψx誤差", "ψy誤差", "ψxx誤差", "ψyy誤差", "ψxxx誤差", "ψyyy誤差"]
        print("  ".join(f"{h:<10}" for h in headers))
        print("-" * 100)
        
        for n in grid_sizes:
            h = (args.x_range[1] - args.x_range[0]) / (n - 1)
            errors = [f"{err:.6e}" for err in results[n]]
            print(f"{n:<10}  {h:<10.6f}  " + "  ".join(f"{err:<10}" for err in errors))
        
        # 可視化
        visualizer = CCD2DVisualizer(output_dir=args.output_dir)
        visualizer.visualize_grid_convergence(test_func.name, grid_sizes, results, 
                                             prefix=args.prefix, save=True)

def test_all_functions(args):
    """全テスト関数でのテスト実行"""
    tester = create_tester(args)
    tester = setup_tester(args, tester)
    
    # テスト関数一覧取得
    if args.dim == 1:
        functions = TestFunctionFactory.create_standard_1d_functions()
        visualizer = CCDVisualizer(output_dir=args.output_dir) if not args.no_visualization else None
    else:
        functions = TestFunctionFactory.create_standard_2d_functions()
        visualizer = CCD2DVisualizer(output_dir=args.output_dir) if not args.no_visualization else None
    
    print("\n==== 全関数のテスト ====")
    if args.dim == 1:
        print(f"1D モード ({args.nx_points} 点)")
        print("-" * 70)
        print(f"{'関数':<12} {'ψ誤差':<12} {'ψ\'誤差':<12} {'ψ\"誤差':<12} {'ψ\'\"誤差':<12}")
        print("-" * 70)
    else:
        print(f"2D モード ({args.nx_points}x{args.ny_points} 点)")
        print("-" * 100)
        headers = ["関数", "ψ誤差", "ψx誤差", "ψy誤差", "ψxx誤差", "ψyy誤差", "ψxxx誤差", "ψyyy誤差"]
        print("  ".join(f"{h:<10}" for h in headers))
        print("-" * 100)
    
    print(f"ソルバー: {args.solver}, 方程式セット: {args.equation_set}")
    print(f"スケーリング: {args.scaling if args.scaling else 'なし'}")
    
    results_summary = {}
    
    # 各関数でテスト実行
    for func in functions:
        results = tester.run_test_with_options(func)
        errors = results["errors"]
        
        if args.dim == 1:
            print(f"{func.name:<12} {errors[0]:<12.6e} {errors[1]:<12.6e} {errors[2]:<12.6e} {errors[3]:<12.6e}")
        else:
            error_strs = [f"{err:.6e}" for err in errors]
            print(f"{func.name:<10}  " + "  ".join(f"{err:<10}" for err in error_strs))
        
        results_summary[func.name] = errors
        
        # 可視化
        if not args.no_visualization:
            prefix = f"{args.prefix}_{func.name.lower()}" if args.prefix else func.name.lower()
            if args.dim == 1:
                visualizer.visualize_derivatives(grid=tester.grid, function=results["function"],
                                               numerical=results["numerical"], exact=results["exact"],
                                               errors=results["errors"], prefix=prefix, save=True)
            else:
                visualizer.visualize_solution(grid=tester.grid, function_name=results["function"],
                                            numerical=results["numerical"], exact=results["exact"],
                                            errors=results["errors"], prefix=prefix, save=True)
    
    # 誤差比較グラフ
    if not args.no_visualization:
        if args.dim == 1:
            visualizer.compare_all_functions_errors(results_summary, prefix=args.prefix)
        else:
            visualizer.compare_all_functions_errors(results_summary, grid_size=args.nx_points, prefix=args.prefix)
    
    return results_summary

def run_single_test(args):
    """単一関数のテスト実行"""
    tester = create_tester(args)
    tester = setup_tester(args, tester)
    
    # テスト関数取得
    test_func = tester.get_test_function(args.test_func)
    
    print(f"\n{test_func.name}関数でテスト実行中...")
    print(f"次元: {args.dim}D")
    print(f"ソルバー: {args.solver}, 方程式セット: {args.equation_set}")
    print(f"スケーリング: {args.scaling if args.scaling else 'なし'}")
    
    # テスト実行
    start_time = time.time()
    results = tester.run_test_with_options(test_func)
    end_time = time.time()
    
    # 反復回数
    iter_count = getattr(tester.solver, 'last_iterations', None)
    iter_str = str(iter_count) if iter_count is not None else "N/A"
    
    # 結果出力
    print("\n誤差分析:")
    if args.dim == 1:
        print(f"  ψ誤差:    {results['errors'][0]:.6e}")
        print(f"  ψ'誤差:   {results['errors'][1]:.6e}")
        print(f"  ψ''誤差:  {results['errors'][2]:.6e}")
        print(f"  ψ'''誤差: {results['errors'][3]:.6e}")
    else:
        print(f"  ψ誤差:    {results['errors'][0]:.6e}")
        print(f"  ψx誤差:   {results['errors'][1]:.6e}")
        print(f"  ψy誤差:   {results['errors'][2]:.6e}")
        print(f"  ψxx誤差:  {results['errors'][3]:.6e}")
        print(f"  ψyy誤差:  {results['errors'][4]:.6e}")
        print(f"  ψxxx誤差: {results['errors'][5]:.6e}")
        print(f"  ψyyy誤差: {results['errors'][6]:.6e}")
    
    print(f"  実行時間: {end_time - start_time:.4f} 秒")
    if iter_count is not None:
        print(f"  反復回数: {iter_str}")
    
    # 可視化
    if not args.no_visualization:
        if args.dim == 1:
            visualizer = CCDVisualizer(output_dir=args.output_dir)
            visualizer.visualize_derivatives(tester.grid, results["function"],
                                          results["numerical"], results["exact"],
                                          results["errors"], prefix=args.prefix, save=True)
        else:
            visualizer = CCD2DVisualizer(output_dir=args.output_dir)
            visualizer.visualize_solution(tester.grid, function_name=results["function"],
                                        numerical=results["numerical"], exact=results["exact"],
                                        errors=results["errors"], prefix=args.prefix, save=True)
    
    return results

def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 各種リスト表示
    if args.list_functions:
        list_available_functions(args.dim)
        return
    
    if args.list_scaling:
        list_scaling_methods()
        return
    
    # スケーリング手法比較
    if args.compare_scaling:
        compare_scaling_methods(args)
        return
    
    # 全関数テスト
    if args.test_all_functions:
        test_all_functions(args)
        return
    
    # 収束性テスト
    if args.convergence_test:
        run_convergence_test(args)
        return
    
    # 通常のテスト（単一関数）
    run_single_test(args)

if __name__ == "__main__":
    run_cli()