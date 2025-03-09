# cli.py
import argparse
import os
import time
from typing import Tuple, Optional, Dict, Any
from grid1d import Grid
from tester1d import CCDTester
from test_functions1d import TestFunctionFactory
from visualization1d import CCDVisualizer
from equation_sets1d import EquationSet


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="1D CCD法の実装")

    # 基本的な引数
    parser.add_argument("--n-points", type=int, default=21, help="格子点の数")
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        help="x座標の範囲 (最小値 最大値)",
    )
    parser.add_argument("--test-func", type=str, default="Sin", help="テスト関数名")
    parser.add_argument(
        "--no-visualization", action="store_true", help="可視化を無効化"
    )
    parser.add_argument(
        "--convergence-test", action="store_true", help="格子収束性テストを実行"
    )
    parser.add_argument(
        "--test-all-functions",
        action="store_true",
        help="全てのテスト関数でテストを実行",
    )
    parser.add_argument("--prefix", type=str, default="", help="出力ファイル名の接頭辞")

    # 方程式セットオプション
    equation_set_group = parser.add_argument_group('方程式セットオプション')
    equation_set_group.add_argument(
        "--equation-set",
        type=str,
        choices=list(EquationSet.get_available_sets().keys()),
        default="poisson",
        help="使用する方程式セット (デフォルト: poisson)"
    )

    # ソルバー関連のオプションを追加
    solver_group = parser.add_argument_group('ソルバーオプション')
    
    # ソルバー種類
    solver_group.add_argument(
        "--solver", 
        type=str, 
        choices=['direct', 'gmres', 'cg', 'cgs'], 
        default='direct',
        help="使用するソルバー (デフォルト: direct)"
    )
    
    # 反復法関連のオプション
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
    
    # 行列分析
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


def get_solver_options(args: argparse.Namespace) -> Dict[str, Any]:
    """
    コマンドライン引数からソルバーオプションを取得
    
    Args:
        args: コマンドライン引数の名前空間
        
    Returns:
        Dict[str, Any]: ソルバーオプション辞書
    """
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

def compare_scaling_methods(func_name, n_points, x_range, solver_method, solver_options):
    """異なるスケーリング手法の性能を比較"""
    from scaling import plugin_manager
    import time
    
    # 利用可能なすべてのスケーリング手法を取得
    scaling_methods = plugin_manager.get_available_plugins()
    
    # グリッドとテスト関数を作成
    grid = Grid(n_points, x_range)
    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next((f for f in test_funcs if f.name == func_name), test_funcs[0])
    
    results = {}
    
    print(f"\n{selected_func.name}関数でスケーリング手法を比較しています...")
    print(f"グリッドサイズ: {n_points} 点")
    print(f"ソルバー: {solver_method}")
    print("\n" + "-" * 80)
    print(f"{'スケーリング手法':<25} {'実行時間 (s)':<15} {'反復回数':<15} {'誤差':<15}")
    print("-" * 80)
    
    for method in scaling_methods:
        # このスケーリング手法でテスターを作成
        tester = CCDTester(grid)
        tester.set_solver_options(solver_method, solver_options, False)
        tester.scaling_method = method  # テスターにスケーリング手法を設定
        
        # 解の時間を計測
        start_time = time.time()
        result = tester.run_test_with_options(selected_func)
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
    
    print("-" * 80)
    return results

def run_convergence_test(
    func_name: str,
    x_range: Tuple[float, float],
    prefix: str,
    solver_method: str = "direct",
    solver_options: Optional[Dict[str, Any]] = None,
    analyze_matrix: bool = False,
    equation_set_name: str = "poisson",
    scaling_method: str = None,
):
    """格子収束性テストを実行"""
    # テスト関数を選択
    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next((f for f in test_funcs if f.name == func_name), test_funcs[0])

    # グリッドサイズ
    grid_sizes = [11, 21, 41, 81, 161]

    # 基準グリッドでテスターを作成
    base_grid = Grid(grid_sizes[0], x_range)
    tester = CCDTester(base_grid)

    # ソルバー設定
    if solver_options:
        tester.set_solver_options(solver_method, solver_options, analyze_matrix)
        tester.solver.scaling_method = scaling_method
        
    # 方程式セット設定
    tester.set_equation_set(equation_set_name)

    # 収束性テストを実行
    print(f"{selected_func.name}関数での格子収束性テストを実行しています...")
    print("ディリクレ境界条件とノイマン境界条件を使用")
    print(f"ソルバー: {solver_method}")
    print(f"方程式セット: {equation_set_name}")
    print(f"スケーリング: {scaling_method if scaling_method else 'なし'}")

    results = tester.run_grid_convergence_test(
        selected_func, grid_sizes, x_range
    )

    # 結果を表示
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
    visualizer = CCDVisualizer()
    visualizer.visualize_grid_convergence(
        selected_func.name,
        grid_sizes,
        results,
        prefix=prefix,
        save=True,
    )

def test_all_functions(
    n_points: int,
    x_range: Tuple[float, float],
    visualize: bool,
    prefix: str,
    solver_method: str = "direct",
    solver_options: Optional[Dict[str, Any]] = None,
    analyze_matrix: bool = False,
    equation_set_name: str = "poisson",
    scaling_method: str = None,
):
    """全てのテスト関数に対してテストを実行"""
    # テスト関数の取得
    test_funcs = TestFunctionFactory.create_standard_functions()

    # 結果を保存する辞書
    results_summary = {}

    # グリッドの作成
    grid = Grid(n_points, x_range)
    visualizer = CCDVisualizer() if visualize else None
    
    # テスターの作成（一度だけ）
    tester = CCDTester(grid)
    
    # ソルバー設定
    tester.set_solver_options(solver_method, solver_options, analyze_matrix)
    tester.scaling_method = scaling_method  # テスターにスケーリング方法を設定
        
    # 方程式セット設定
    tester.set_equation_set(equation_set_name)

    print(f"\n==== 全関数のテスト ({n_points} 点) ====")
    print("ディリクレ境界条件とノイマン境界条件を使用")
    print(f"ソルバー: {solver_method}")
    print(f"方程式セット: {equation_set_name}")
    print(f"スケーリング: {scaling_method if scaling_method else 'なし'}")

    print("\n" + "-" * 80)
    print(
        f"{'関数名':<15} {'ψ誤差':<15} {"ψ'誤差":<15} {'ψ"誤差':<15} {'ψ\'"誤差':<15}"
    )
    print("-" * 80)

    # 各関数に対してテストを実行
    for func in test_funcs:
        # テストの実行（同じテスターインスタンスを再利用）
        results = tester.run_test_with_options(func)

        # 結果の表示
        errors = results["errors"]
        print(
            f"{func.name:<15} {errors[0]:<15.6e} {errors[1]:<15.6e} {errors[2]:<15.6e} {errors[3]:<15.6e}"
        )

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
            )

    print("-" * 80)

    # すべての関数の誤差を比較するグラフを生成
    if visualize:
        visualizer.compare_all_functions_errors(
            results_summary, prefix=prefix,
        )

    return results_summary


def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()

    # 出力ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    
    # スケーリング手法一覧の表示
    if args.list_scaling:
        list_scaling_methods()
        return
    
    # スケーリング手法の比較
    if args.compare_scaling:
        compare_scaling_methods(
            args.test_func, 
            args.n_points, 
            tuple(args.x_range),
            args.solver,
            get_solver_options(args)
        )
        return
    
    # ソルバーオプションの取得
    solver_options = get_solver_options(args)

    # 全関数テスト
    if args.test_all_functions:
        test_all_functions(
            args.n_points,
            tuple(args.x_range),
            not args.no_visualization,
            args.prefix,
            args.solver,
            solver_options,
            args.analyze_matrix,
            args.equation_set,
            args.scaling,
        )
        return

    # 収束性テスト
    if args.convergence_test:
        run_convergence_test(
            args.test_func,
            tuple(args.x_range),
            args.prefix,
            args.solver,
            solver_options,
            args.analyze_matrix,
            args.equation_set,
            args.scaling,
        )
        return

    # 通常のテスト（単一関数）
    # グリッドの作成
    grid = Grid(args.n_points, tuple(args.x_range))

    # テスターの作成
    tester = CCDTester(grid)
    
    # ソルバー設定
    tester.set_solver_options(args.solver, solver_options, args.analyze_matrix)
    tester.solver.scaling_method = args.scaling
    
    # 方程式セット設定
    tester.set_equation_set(args.equation_set)

    # テスト関数の選択
    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next(
        (f for f in test_funcs if f.name == args.test_func), test_funcs[0]
    )

    # テストの実行
    print(f"\n{selected_func.name}関数でテストを実行しています...")
    print("ディリクレ境界条件とノイマン境界条件を使用")
    print(f"ソルバー: {args.solver}")
    print(f"方程式セット: {args.equation_set}")
    print(f"スケーリング: {args.scaling if args.scaling else 'なし'}")

    # 解の時間を計測
    start_time = time.time()
    results = tester.run_test_with_options(selected_func)
    end_time = time.time()

    # 反復回数を取得（反復ソルバーの場合）
    iter_count = "N/A"
    if hasattr(tester.solver, 'last_iterations') and tester.solver.last_iterations is not None:
        iter_count = str(tester.solver.last_iterations)

    # 結果の表示
    print("\n誤差分析:")
    print(f"  ψ誤差:   {results['errors'][0]:.6e}")
    print(f"  ψ'誤差:  {results['errors'][1]:.6e}")
    print(f"  ψ''誤差: {results['errors'][2]:.6e}")
    print(f"  ψ'''誤差:{results['errors'][3]:.6e}")
    print(f"  実行時間: {end_time - start_time:.4f} 秒")
    if iter_count != "N/A":
        print(f"  反復回数: {iter_count}")

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
        )


if __name__ == "__main__":
    run_cli()