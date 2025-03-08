import argparse
import os
from grid2d import Grid2D
from tester2d import CCD2DTester
from test_functions2d import TestFunction2DGenerator
from visualization2d import CCD2DVisualizer

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="2D CCD法の実装")
    
    # 基本的な引数
    parser.add_argument("--nx-points", type=int, default=21, help="x方向の格子点の数")
    parser.add_argument("--ny-points", type=int, default=21, help="y方向の格子点の数")
    parser.add_argument("--x-range", type=float, nargs=2, default=[-1.0, 1.0], help="x座標の範囲 (最小値 最大値)")
    parser.add_argument("--y-range", type=float, nargs=2, default=[-1.0, 1.0], help="y座標の範囲 (最小値 最大値)")
    parser.add_argument("--test-func", type=str, default="Sine2D", help="テスト関数名")
    parser.add_argument("--no-visualization", action="store_true", help="可視化を無効化")
    parser.add_argument("--convergence-test", action="store_true", help="格子収束性テストを実行")
    parser.add_argument("--test-all-functions", action="store_true", help="全てのテスト関数でテストを実行")
    parser.add_argument("--prefix", type=str, default="", help="出力ファイル名の接頭辞")
    parser.add_argument("--list-functions", action="store_true", help="利用可能なテスト関数の一覧を表示")
    
    # 方程式セットオプション
    equation_set_group = parser.add_argument_group('方程式セットオプション')
    equation_set_group.add_argument(
        "--equation-set",
        type=str,
        choices=['poisson', 'derivative'],
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
    
    return parser.parse_args()

def get_solver_options(args):
    """コマンドライン引数からソルバーオプションを取得"""
    return {
        "tol": args.solver_tol,
        "maxiter": args.solver_maxiter,
        "restart": args.solver_restart,
        "use_preconditioner": not args.no_preconditioner,
    }

def list_available_functions():
    """利用可能なテスト関数の一覧を表示"""
    functions = TestFunction2DGenerator.create_standard_functions()
    
    print("\n利用可能な2Dテスト関数:")
    for i, func in enumerate(functions, 1):
        print(f"{i:2d}. {func.name}")
    print("\n注: 1次元の関数名を指定すると、自動的にテンソル積拡張された2次元関数が使用されます。")

def run_convergence_test(
    func_name,
    x_range,
    y_range,
    prefix="",
    solver_method="direct",
    solver_options=None,
    analyze_matrix=False,
    equation_set_name="poisson"
):
    """格子収束性テストを実行"""
    # グリッドサイズ
    grid_sizes = [11, 21, 31, 41]
    
    # 基準グリッドでテスターを作成
    base_grid = Grid2D(grid_sizes[0], grid_sizes[0], x_range, y_range)
    tester = CCD2DTester(base_grid)
    
    # ソルバー設定
    tester.set_solver_options(solver_method, solver_options, analyze_matrix)
    
    # 方程式セット設定
    tester.set_equation_set(equation_set_name)
    
    # テスト関数を取得（文字列名で）
    
    # 収束性テストを実行
    print(f"{func_name}関数での格子収束性テストを実行しています...")
    print(f"ソルバー: {solver_method}")
    print(f"方程式セット: {equation_set_name}")
    
    results = tester.run_grid_convergence_test(
        func_name, grid_sizes, x_range, y_range
    )
    
    # 結果を表示
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
    visualizer = CCD2DVisualizer()
    visualizer.visualize_grid_convergence(
        func_name,
        grid_sizes,
        results,
        prefix=prefix,
        save=True,
    )

def test_all_functions(
    nx_points,
    ny_points,
    x_range,
    y_range,
    visualize=True,
    prefix="",
    solver_method="direct",
    solver_options=None,
    analyze_matrix=False,
    equation_set_name="poisson"
):
    """全てのテスト関数に対してテストを実行"""
    # テスト関数の取得
    test_funcs = TestFunction2DGenerator.create_standard_functions()
    
    # 結果を保存する辞書
    results_summary = {}
    
    # グリッドの作成
    grid = Grid2D(nx_points, ny_points, x_range, y_range)
    visualizer = CCD2DVisualizer() if visualize else None
    
    # テスターの作成
    tester = CCD2DTester(grid)
    
    # ソルバー設定
    tester.set_solver_options(solver_method, solver_options, analyze_matrix)
    
    # 方程式セット設定
    tester.set_equation_set(equation_set_name)
    
    print(f"\n==== 全関数のテスト ({nx_points}x{ny_points} 点) ====")
    print(f"ソルバー: {solver_method}")
    print(f"方程式セット: {equation_set_name}")
    
    print("\n" + "-" * 125)
    print(f"{'関数名':<15} {'ψ誤差':<15} {'ψx誤差':<15} {'ψy誤差':<15} {'ψxx誤差':<15} {'ψyy誤差':<15} {'ψxxx誤差':<15} {'ψyyy誤差':<15}")
    print("-" * 125)
    
    # 各関数に対してテストを実行
    for func in test_funcs:
        # テストの実行
        results = tester.run_test_with_options(func)
        
        # 結果の表示
        errors = results["errors"]
        print(
            f"{func.name:<15} {errors[0]:<15.6e} {errors[1]:<15.6e} {errors[2]:<15.6e} "
            f"{errors[3]:<15.6e} {errors[4]:<15.6e} {errors[5]:<15.6e} {errors[6]:<15.6e}"
        )
        
        # 結果を保存
        results_summary[func.name] = errors
        
        # 可視化
        if visualize:
            visualizer.visualize_solution(
                grid,
                results["function"],
                results["numerical"],
                results["exact"],
                results["errors"],
                prefix=f"{prefix}_{func.name.lower()}" if prefix else func.name.lower(),
                save=True,
            )
    
    print("-" * 125)
    
    # すべての関数の誤差を比較するグラフを生成
    if visualize:
        visualizer.compare_all_functions_errors(
            results_summary,
            grid_size=nx_points,
            prefix=prefix,
        )
    
    return results_summary

def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs("results_2d", exist_ok=True)
    
    # 関数一覧の表示
    if args.list_functions:
        list_available_functions()
        return
    
    # ソルバーオプションの取得
    solver_options = get_solver_options(args)
    
    # 全関数テスト
    if args.test_all_functions:
        test_all_functions(
            args.nx_points,
            args.ny_points,
            tuple(args.x_range),
            tuple(args.y_range),
            not args.no_visualization,
            args.prefix,
            args.solver,
            solver_options,
            args.analyze_matrix,
            args.equation_set,
        )
        return
    
    # 収束性テスト
    if args.convergence_test:
        run_convergence_test(
            args.test_func,
            tuple(args.x_range),
            tuple(args.y_range),
            args.prefix,
            args.solver,
            solver_options,
            args.analyze_matrix,
            args.equation_set,
        )
        return
    
    # 通常のテスト（単一関数）
    # グリッドの作成
    grid = Grid2D(args.nx_points, args.ny_points, tuple(args.x_range), tuple(args.y_range))
    
    # テスターの作成
    tester = CCD2DTester(grid)
    
    # ソルバー設定
    tester.set_solver_options(args.solver, solver_options, args.analyze_matrix)
    
    # 方程式セット設定
    tester.set_equation_set(args.equation_set)
    
    # テストの実行
    print(f"\n{args.test_func}関数でテストを実行しています...")
    print(f"ソルバー: {args.solver}")
    print(f"方程式セット: {args.equation_set}")
    
    results = tester.run_test_with_options(args.test_func)
    
    # 結果の表示
    print("\n誤差分析:")
    print(f"  ψ誤差:   {results['errors'][0]:.6e}")
    print(f"  ψx誤差:  {results['errors'][1]:.6e}")
    print(f"  ψy誤差:  {results['errors'][2]:.6e}")
    print(f"  ψxx誤差: {results['errors'][3]:.6e}")
    print(f"  ψyy誤差: {results['errors'][4]:.6e}")
    print(f"  ψxxx誤差:{results['errors'][5]:.6e}")
    print(f"  ψyyy誤差:{results['errors'][6]:.6e}")
    
    # 可視化
    if not args.no_visualization:
        visualizer = CCD2DVisualizer()
        visualizer.visualize_solution(
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