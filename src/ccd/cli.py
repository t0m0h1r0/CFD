import argparse
import os
from grid import Grid
from tester import CCDTester
from test_functions import TestFunctionFactory
from visualization import CCDVisualizer
from equation_sets import EquationSet

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="1D CCD法の実装")

    # 基本的な引数
    parser.add_argument("--n-points", type=int, default=21, help="格子点の数")
    parser.add_argument("--x-range", type=float, nargs=2, default=[-1.0, 1.0], help="x座標の範囲")
    parser.add_argument("--test-func", type=str, default="Sin", help="テスト関数名")
    parser.add_argument("--no-visualization", action="store_true", help="可視化を無効化")
    parser.add_argument("--convergence-test", action="store_true", help="格子収束性テストを実行")
    parser.add_argument("--test-all-functions", action="store_true", help="全てのテスト関数でテストを実行")
    parser.add_argument("--prefix", type=str, default="", help="出力ファイル名の接頭辞")
    parser.add_argument("--equation-set", type=str, default="poisson", help="使用する方程式セット")
    parser.add_argument("--solver", type=str, default="direct", help="使用するソルバー")
    parser.add_argument("--solver-tol", type=float, default=1e-10, help="反復ソルバーの収束許容誤差")
    parser.add_argument("--solver-maxiter", type=int, default=1000, help="反復ソルバーの最大反復回数")
    parser.add_argument("--no-preconditioner", action="store_true", help="前処理を使用しない")

    return parser.parse_args()

def get_solver_options(args):
    return {
        "tol": args.solver_tol,
        "maxiter": args.solver_maxiter,
        "restart": 100,
        "use_preconditioner": not args.no_preconditioner,
    }

def test_all_functions(n_points, x_range, visualize, prefix, solver_method="direct", 
                     solver_options=None, analyze_matrix=False, equation_set_name="poisson"):
    """全てのテスト関数に対してテストを実行"""
    test_funcs = TestFunctionFactory.create_standard_functions()
    results_summary = {}
    grid = Grid(n_points, x_range)
    visualizer = CCDVisualizer() if visualize else None
    tester = CCDTester(grid)
    
    if solver_options:
        tester.set_solver_options(solver_method, solver_options, analyze_matrix)
    tester.set_equation_set(equation_set_name)

    print(f"\n==== 全関数のテスト ({n_points} 点) ====")
    print("ソルバー: " + solver_method)
    print("方程式セット: " + equation_set_name)
    print("\n" + "-" * 80)
    print(f"{'関数名':<15} {'ψ誤差':<15} {"ψ'誤差":<15} {'ψ"誤差':<15} {'ψ\'"誤差':<15}")
    print("-" * 80)

    for func in test_funcs:
        results = tester.run_test_with_options(func)
        errors = results["errors"]
        print(f"{func.name:<15} {errors[0]:<15.6e} {errors[1]:<15.6e} {errors[2]:<15.6e} {errors[3]:<15.6e}")
        results_summary[func.name] = errors

        if visualize:
            visualizer.visualize_derivatives(grid, results["function"], results["numerical"],
                                           results["exact"], results["errors"], prefix=prefix)

    print("-" * 80)
    return results_summary

def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    solver_options = get_solver_options(args)

    if args.test_all_functions:
        test_all_functions(args.n_points, tuple(args.x_range), not args.no_visualization,
                         args.prefix, args.solver, solver_options, False, args.equation_set)
        return

    # 個別関数のテスト処理
    grid = Grid(args.n_points, tuple(args.x_range))
    tester = CCDTester(grid)
    tester.set_solver_options(args.solver, solver_options, False)
    tester.set_equation_set(args.equation_set)

    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next((f for f in test_funcs if f.name == args.test_func), test_funcs[0])

    print(f"\n{selected_func.name}関数でテストを実行しています...")
    results = tester.run_test_with_options(selected_func)

    print("\n誤差分析:")
    print(f"  ψ誤差: {results['errors'][0]:.6e}")
    print(f"  ψ'誤差: {results['errors'][1]:.6e}")
    print(f"  ψ''誤差: {results['errors'][2]:.6e}")
    print(f"  ψ'''誤差: {results['errors'][3]:.6e}")

    if not args.no_visualization:
        visualizer = CCDVisualizer()
        visualizer.visualize_derivatives(grid, results["function"], results["numerical"],
                                       results["exact"], results["errors"], prefix=args.prefix)

if __name__ == "__main__":
    run_cli()