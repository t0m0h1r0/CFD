import argparse
import os
import time
from grid import Grid
from tester import CCDTester1D, CCDTester2D  # 直接次元別クラスをインポート
from test_functions1d import TestFunctionFactory
from test_functions2d import TestFunction2DGenerator
from visualization1d import CCDVisualizer
from visualization2d import CCD2DVisualizer

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="CCD法の統合CLI (1D/2D)")
    
    # 共通基本オプション
    common_group = parser.add_argument_group('共通オプション')
    common_group.add_argument("--dim", type=int, choices=[1, 2], default=1, help="問題の次元 (1 or 2)")
    common_group.add_argument("--test-func", type=str, default="Sine", help="テスト関数名")
    common_group.add_argument("--no-visualization", action="store_true", help="可視化を無効化")
    common_group.add_argument("--convergence-test", action="store_true", help="格子収束性テストを実行")
    common_group.add_argument("--test-all-functions", action="store_true", help="全てのテスト関数でテストを実行")
    common_group.add_argument("--prefix", type=str, default="", help="出力ファイル名の接頭辞")
    common_group.add_argument("--output-dir", "-o", type=str, default="results", help="画像出力先ディレクトリ（デフォルト: results）")
    common_group.add_argument("--list-functions", action="store_true", help="利用可能なテスト関数の一覧を表示")
    common_group.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        help="x座標の範囲 (最小値 最大値)"
    )
    
    # 共通グリッドオプション
    grid_group = parser.add_argument_group('グリッドオプション')
    grid_group.add_argument("--nx-points", type=int, default=21, help="x方向の格子点の数 (1D/2D)")
    
    # 2D固有のオプション
    dim2_group = parser.add_argument_group('2D固有オプション')
    dim2_group.add_argument("--ny-points", type=int, default=21, help="y方向の格子点の数 (2D専用)")
    dim2_group.add_argument("--y-range", type=float, nargs=2, default=[-1.0, 1.0], help="y座標の範囲 (最小値 最大値)")
    
    # 方程式セットオプション
    equation_set_group = parser.add_argument_group('方程式セットオプション')
    equation_set_group.add_argument(
        "--equation-set",
        type=str,
        default="poisson",
        help="使用する方程式セット (デフォルト: poisson)"
    )
    
    # ソルバーオプション
    solver_group = parser.add_argument_group('ソルバーオプション')
    solver_group.add_argument(
        "--solver", 
        type=str, 
        choices=['direct', 'gmres', 'cg', 'cgs'], 
        default='direct',
        help="使用するソルバー (デフォルト: direct)"
    )
    solver_group.add_argument(
        "--solver-tol", 
        type=float, 
        default=1e-10,
        help="反復ソルバーの収束許容誤差 (デフォルト: 1e-10)"
    )
    solver_group.add_argument(
        "--solver-maxiter", 
        type=int, 
        default=1000,
        help="反復ソルバーの最大反復回数 (デフォルト: 1000)"
    )
    solver_group.add_argument(
        "--solver-restart", 
        type=int, 
        default=100,
        help="GMRESのリスタート値 (デフォルト: 100)"
    )
    solver_group.add_argument(
        "--no-preconditioner", 
        action="store_true",
        help="前処理を使用しない"
    )
    solver_group.add_argument(
        "--analyze-matrix", 
        action="store_true",
        help="行列の疎性を分析して表示"
    )
    
    # スケーリングオプション
    scaling_group = parser.add_argument_group('スケーリングオプション')
    scaling_group.add_argument(
        "--scaling", 
        type=str,
        default=None,
        help="使用するスケーリング手法 (デフォルト: なし)"
    )
    scaling_group.add_argument(
        "--list-scaling", 
        action="store_true",
        help="利用可能なスケーリング手法の一覧を表示"
    )
    scaling_group.add_argument(
        "--compare-scaling", 
        action="store_true",
        help="異なるスケーリング手法の性能を比較"
    )
    
    return parser.parse_args()

def get_solver_options(args):
    """コマンドライン引数からソルバーオプションを取得"""
    return {
        "tol": args.solver_tol,
        "maxiter": args.solver_maxiter,
        "restart": args.solver_restart,
        "use_preconditioner": not args.no_preconditioner,
    }

def list_scaling_methods():
    """利用可能なスケーリング手法の一覧を表示"""
    from scaling import plugin_manager
    
    plugins = plugin_manager.get_available_plugins()
    
    print("\n利用可能なスケーリング手法:")
    for i, name in enumerate(plugins, 1):
        scaler = plugin_manager.get_plugin(name)
        print(f"{i:2d}. {name}: {scaler.description}")
    print()

def list_available_functions(dim=1):
    """利用可能なテスト関数の一覧を表示"""
    if dim == 1:
        functions = TestFunctionFactory.create_standard_functions()
        print("\n利用可能な1Dテスト関数:")
    else:
        functions = TestFunction2DGenerator.create_standard_functions()
        print("\n利用可能な2Dテスト関数:")
    
    for i, func in enumerate(functions, 1):
        print(f"{i:2d}. {func.name}")
    
    if dim == 2:
        print("\n注: 1次元の関数名を指定すると、自動的にテンソル積拡張された2次元関数が使用されます。")

def compare_scaling_methods(args):
    """異なるスケーリング手法の性能を比較"""
    from scaling import plugin_manager
    
    # 利用可能なすべてのスケーリング手法を取得
    scaling_methods = plugin_manager.get_available_plugins()
    
    # グリッドを作成
    x_range = tuple(args.x_range)
    if args.dim == 1:
        grid = Grid(args.nx_points, x_range=x_range)
        # 1Dテスターを使用
        tester = CCDTester1D(grid)
    else:
        y_range = tuple(args.y_range)
        grid = Grid(args.nx_points, args.ny_points, x_range=x_range, y_range=y_range)
        # 2Dテスターを使用
        tester = CCDTester2D(grid)
    
    # テスト関数取得
    test_func = tester.get_test_function(args.test_func)
    
    results = {}
    
    print(f"\n{test_func.name}関数でスケーリング手法を比較しています...")
    if args.dim == 1:
        print(f"グリッドサイズ: {args.nx_points} 点")
    else:
        print(f"グリッドサイズ: {args.nx_points}x{args.ny_points} 点")
    print(f"ソルバー: {args.solver}")
    
    print("\n" + "-" * 90)
    print(f"{'スケーリング手法':<25} {'実行時間 (s)':<15} {'反復回数':<15} {'誤差':<15}")
    print("-" * 90)
    
    solver_options = get_solver_options(args)
    
    for method in scaling_methods:
        # このスケーリング手法でテスターを設定
        tester.set_solver_options(args.solver, solver_options, False)
        tester.scaling_method = method
        
        # 解の時間を計測
        start_time = time.time()
        result = tester.run_test_with_options(test_func)
        end_time = time.time()
        
        # 反復回数を取得（反復ソルバーの場合）
        iter_count = "N/A"
        if hasattr(tester.solver, 'last_iterations') and tester.solver.last_iterations is not None:
            iter_count = str(tester.solver.last_iterations)
        
        # 最大誤差を計算
        max_error = max(result['errors'])
        
        # 結果を表示
        print(f"{method:<25} {end_time - start_time:<15.4f} {iter_count:<15} {max_error:<15.6e}")
        
        # 後の分析のために保存
        results[method] = {
            'time': end_time - start_time,
            'iterations': tester.solver.last_iterations if hasattr(tester.solver, 'last_iterations') else None,
            'errors': result['errors']
        }
    
    print("-" * 90)
    return results

def run_convergence_test(args):
    """格子収束性テストを実行"""
    x_range = tuple(args.x_range)
    
    # グリッドサイズ
    if args.dim == 1:
        grid_sizes = [11, 21, 41, 81, 161]
        # グリッドを作成（1D）
        base_grid = Grid(grid_sizes[0], x_range=x_range)
        y_range = None
        # 1Dテスターを使用
        tester = CCDTester1D(base_grid)
    else:
        grid_sizes = [11, 21, 31, 41]
        y_range = tuple(args.y_range)
        # グリッドを作成（2D）
        base_grid = Grid(grid_sizes[0], grid_sizes[0], x_range=x_range, y_range=y_range)
        # 2Dテスターを使用
        tester = CCDTester2D(base_grid)
    
    # ソルバー設定
    solver_options = get_solver_options(args)
    tester.set_solver_options(args.solver, solver_options, args.analyze_matrix)
    tester.scaling_method = args.scaling
    
    # 方程式セット設定
    tester.set_equation_set(args.equation_set)
    
    # テスト関数取得
    test_func = tester.get_test_function(args.test_func)
    
    # 収束性テストを実行
    print(f"{test_func.name}関数での格子収束性テストを実行しています...")
    print(f"ソルバー: {args.solver}")
    print(f"方程式セット: {args.equation_set}")
    print(f"スケーリング: {args.scaling if args.scaling else 'なし'}")
    
    results = tester.run_grid_convergence_test(
        test_func, grid_sizes, x_range, y_range
    )
    
    # 結果を表示
    if args.dim == 1:
        print("\n格子収束性テストの結果:")
        print(
            f"{'格子サイズ':<10} {'h':<10} {'ψ誤差':<15} {"ψ'誤差":<15} {'ψ"誤差':<15} {'ψ\'"誤差':<15}"
        )
        print("-" * 80)
        
        for n in grid_sizes:
            h = (x_range[1] - x_range[0]) / (n - 1)
            print(
                f"{n:<10} {h:<10.6f} {results[n][0]:<15.6e} {results[n][1]:<15.6e} {results[n][2]:<15.6e} {results[n][3]:<15.6e}"
            )
        
        # 可視化
        output_dir = args.output_dir
        visualizer = CCDVisualizer(output_dir=output_dir)
        visualizer.visualize_grid_convergence(
            test_func.name,
            grid_sizes,
            results,
            prefix=args.prefix,
            save=True,
        )
    else:
        print("\n格子収束性テストの結果:")
        print(f"{'格子サイズ':<10} {'h':<10} {'ψ誤差':<15} {'ψx誤差':<15} {'ψy誤差':<15} {'ψxx誤差':<15} {'ψyy誤差':<15} {'ψxxx誤差':<15} {'ψyyy誤差':<15}")
        print("-" * 125)
        
        for n in grid_sizes:
            h = (x_range[1] - x_range[0]) / (n - 1)
            print(
                f"{n:<10} {h:<10.6f} {results[n][0]:<15.6e} {results[n][1]:<15.6e} {results[n][2]:<15.6e} "
                f"{results[n][3]:<15.6e} {results[n][4]:<15.6e} {results[n][5]:<15.6e} {results[n][6]:<15.6e}"
            )
        
        # 可視化
        output_dir = args.output_dir
        visualizer = CCD2DVisualizer(output_dir=output_dir)
        visualizer.visualize_grid_convergence(
            test_func.name,
            grid_sizes,
            results,
            prefix=args.prefix,
            save=True,
        )

def test_all_functions(args):
    """全てのテスト関数に対してテストを実行"""
    x_range = tuple(args.x_range)
    
    # 結果を保存する辞書
    results_summary = {}
    
    # グリッドの作成
    if args.dim == 1:
        grid = Grid(args.nx_points, x_range=x_range)
        # 出力ディレクトリを指定
        output_dir = args.output_dir
        visualizer = CCDVisualizer(output_dir=output_dir) if not args.no_visualization else None
        functions = TestFunctionFactory.create_standard_functions()
        # 1Dテスターを使用
        tester = CCDTester1D(grid)
    else:
        y_range = tuple(args.y_range)
        grid = Grid(args.nx_points, args.ny_points, x_range=x_range, y_range=y_range)
        output_dir = args.output_dir
        visualizer = CCD2DVisualizer(output_dir=output_dir) if not args.no_visualization else None
        functions = TestFunction2DGenerator.create_standard_functions()
        # 2Dテスターを使用
        tester = CCDTester2D(grid)
    
    # ソルバー設定
    solver_options = get_solver_options(args)
    tester.set_solver_options(args.solver, solver_options, args.analyze_matrix)
    tester.scaling_method = args.scaling
    
    # 方程式セット設定
    tester.set_equation_set(args.equation_set)
    
    print("\n==== 全関数のテスト ====")
    if args.dim == 1:
        print(f"1D モード ({args.nx_points} 点)")
        print("-" * 80)
        print(f"{'関数名':<15} {'ψ誤差':<15} {"ψ'誤差":<15} {'ψ"誤差':<15} {'ψ\'"誤差':<15}")
        print("-" * 80)
    else:
        print(f"2D モード ({args.nx_points}x{args.ny_points} 点)")
        print("-" * 125)
        print(f"{'関数名':<15} {'ψ誤差':<15} {'ψx誤差':<15} {'ψy誤差':<15} {'ψxx誤差':<15} {'ψyy誤差':<15} {'ψxxx誤差':<15} {'ψyyy誤差':<15}")
        print("-" * 125)
    
    print(f"ソルバー: {args.solver}")
    print(f"方程式セット: {args.equation_set}")
    print(f"スケーリング: {args.scaling if args.scaling else 'なし'}")
    
    # 各関数に対してテストを実行
    for func in functions:
        # テストの実行
        results = tester.run_test_with_options(func)
        
        # 結果の表示
        errors = results["errors"]
        
        if args.dim == 1:
            print(
                f"{func.name:<15} {errors[0]:<15.6e} {errors[1]:<15.6e} {errors[2]:<15.6e} {errors[3]:<15.6e}"
            )
        else:
            print(
                f"{func.name:<15} {errors[0]:<15.6e} {errors[1]:<15.6e} {errors[2]:<15.6e} "
                f"{errors[3]:<15.6e} {errors[4]:<15.6e} {errors[5]:<15.6e} {errors[6]:<15.6e}"
            )
        
        # 結果を保存
        results_summary[func.name] = errors
        
        # 可視化
        if not args.no_visualization:
            if args.dim == 1:
                visualizer.visualize_derivatives(
                    grid,
                    results["function"],
                    results["numerical"],
                    results["exact"],
                    results["errors"],
                    prefix=f"{args.prefix}_{func.name.lower()}" if args.prefix else func.name.lower(),
                    save=True,
                )
            else:
                visualizer.visualize_solution(
                    grid,
                    results["function"],
                    results["numerical"],
                    results["exact"],
                    results["errors"],
                    prefix=f"{args.prefix}_{func.name.lower()}" if args.prefix else func.name.lower(),
                    save=True,
                )
    
    # すべての関数の誤差を比較するグラフを生成
    if not args.no_visualization:
        if args.dim == 1:
            visualizer.compare_all_functions_errors(
                results_summary,
                prefix=args.prefix,
            )
        else:
            visualizer.compare_all_functions_errors(
                results_summary,
                grid_size=args.nx_points,
                prefix=args.prefix,
            )
    
    return results_summary

def run_single_test(args):
    """単一関数のテストを実行"""
    x_range = tuple(args.x_range)
    
    # グリッドの作成
    if args.dim == 1:
        grid = Grid(args.nx_points, x_range=x_range)
        # 1Dテスターを使用
        tester = CCDTester1D(grid)
    else:
        y_range = tuple(args.y_range)
        grid = Grid(args.nx_points, args.ny_points, x_range=x_range, y_range=y_range)
        # 2Dテスターを使用
        tester = CCDTester2D(grid)
    
    # ソルバー設定
    solver_options = get_solver_options(args)
    tester.set_solver_options(args.solver, solver_options, args.analyze_matrix)
    tester.scaling_method = args.scaling
    
    # 方程式セット設定
    tester.set_equation_set(args.equation_set)
    
    # テスト関数取得
    test_func = tester.get_test_function(args.test_func)
    
    # テストの実行
    print(f"\n{test_func.name}関数でテストを実行しています...")
    print(f"次元: {args.dim}D")
    print(f"ソルバー: {args.solver}")
    print(f"方程式セット: {args.equation_set}")
    print(f"スケーリング: {args.scaling if args.scaling else 'なし'}")
    
    # 解の時間を計測
    start_time = time.time()
    results = tester.run_test_with_options(test_func)
    end_time = time.time()
    
    # 反復回数を取得（反復ソルバーの場合）
    iter_count = "N/A"
    if hasattr(tester.solver, 'last_iterations') and tester.solver.last_iterations is not None:
        iter_count = str(tester.solver.last_iterations)
    
    # 結果の表示
    print("\n誤差分析:")
    
    if args.dim == 1:
        print(f"  ψ誤差:   {results['errors'][0]:.6e}")
        print(f"  ψ'誤差:  {results['errors'][1]:.6e}")
        print(f"  ψ''誤差: {results['errors'][2]:.6e}")
        print(f"  ψ'''誤差:{results['errors'][3]:.6e}")
    else:
        print(f"  ψ誤差:   {results['errors'][0]:.6e}")
        print(f"  ψx誤差:  {results['errors'][1]:.6e}")
        print(f"  ψy誤差:  {results['errors'][2]:.6e}")
        print(f"  ψxx誤差: {results['errors'][3]:.6e}")
        print(f"  ψyy誤差: {results['errors'][4]:.6e}")
        print(f"  ψxxx誤差:{results['errors'][5]:.6e}")
        print(f"  ψyyy誤差:{results['errors'][6]:.6e}")
    
    print(f"  実行時間: {end_time - start_time:.4f} 秒")
    
    if iter_count != "N/A":
        print(f"  反復回数: {iter_count}")
    
    # 可視化
    if not args.no_visualization:
        if args.dim == 1:
            output_dir = args.output_dir
            visualizer = CCDVisualizer(output_dir=output_dir)
            visualizer.visualize_derivatives(
                grid,
                results["function"],
                results["numerical"],
                results["exact"],
                results["errors"],
                prefix=args.prefix,
                save=True,
            )
        else:
            output_dir = args.output_dir
            visualizer = CCD2DVisualizer(output_dir=output_dir)
            visualizer.visualize_solution(
                grid,
                results["function"],
                results["numerical"],
                results["exact"],
                results["errors"],
                prefix=args.prefix,
                save=True,
            )
    
    return results

def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 関数一覧の表示
    if args.list_functions:
        list_available_functions(args.dim)
        return
    
    # スケーリング手法一覧の表示
    if args.list_scaling:
        list_scaling_methods()
        return
    
    # スケーリング手法の比較
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