# cli.py
import argparse
import os
from typing import Tuple, Optional
from grid import Grid
from tester import CCDTester
from test_functions import TestFunctionFactory
from visualization import CCDVisualizer


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="1D CCD法の実装")

    # 引数の追加
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
    parser.add_argument(
        "--rehu-scaling",
        type=float,
        default=None,
        help="指定したRehu数でスケーリングを適用",
    )
    parser.add_argument("--prefix", type=str, default="", help="出力ファイル名の接頭辞")
    parser.add_argument("--dpi", type=int, default=150, help="出力画像のDPI")
    parser.add_argument(
        "--show", action="store_true", help="プロットを表示（ファイル保存に加えて）"
    )

    return parser.parse_args()


def run_convergence_test(
    func_name: str,
    x_range: Tuple[float, float],
    rehu_number: Optional[float],
    prefix: str,
    dpi: int,
    show: bool,
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

    # 収束性テストを実行
    print(f"{selected_func.name}関数での格子収束性テストを実行しています...")
    print("ディリクレ境界条件とノイマン境界条件を使用")

    if rehu_number is not None:
        print(f"Rehuスケーリングを適用（Rehu数: {rehu_number}）")

    results = tester.run_grid_convergence_test(
        selected_func, grid_sizes, x_range, rehu_number=rehu_number
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
        show=show,
        dpi=dpi,
    )


def test_all_functions(
    n_points: int,
    x_range: Tuple[float, float],
    rehu_number: Optional[float],
    visualize: bool,
    prefix: str,
    dpi: int,
    show: bool,
):
    """全てのテスト関数に対してテストを実行"""
    # テスト関数の取得
    test_funcs = TestFunctionFactory.create_standard_functions()

    # 結果を保存する辞書
    results_summary = {}

    # グリッドの作成
    grid = Grid(n_points, x_range)
    visualizer = CCDVisualizer() if visualize else None

    print(f"\n==== 全関数のテスト ({n_points} 点) ====")
    print("ディリクレ境界条件とノイマン境界条件を使用")

    if rehu_number is not None:
        print(f"Rehuスケーリングを適用（Rehu数: {rehu_number}）")

    print("\n" + "-" * 80)
    print(
        f"{'関数名':<15} {'ψ誤差':<15} {"ψ'誤差":<15} {'ψ"誤差':<15} {'ψ\'"誤差':<15}"
    )
    print("-" * 80)

    # 各関数に対してテストを実行
    for func in test_funcs:
        # テスターの作成
        tester = CCDTester(grid)

        # テストの実行
        results = tester.run_test_with_options(func, rehu_number=rehu_number)

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
                show=show,
                dpi=dpi,
            )

    print("-" * 80)

    # すべての関数の誤差を比較するグラフを生成
    if visualize:
        visualizer.compare_all_functions_errors(
            results_summary, prefix=prefix, dpi=dpi, show=show
        )

    return results_summary


def run_cli():
    """CLIエントリーポイント"""
    args = parse_args()

    # 出力ディレクトリの作成
    os.makedirs("results", exist_ok=True)

    # 全関数テスト
    if args.test_all_functions:
        test_all_functions(
            args.n_points,
            tuple(args.x_range),
            args.rehu_scaling,
            not args.no_visualization,
            args.prefix,
            args.dpi,
            args.show,
        )
        return

    # 収束性テスト
    if args.convergence_test:
        run_convergence_test(
            args.test_func,
            tuple(args.x_range),
            args.rehu_scaling,
            args.prefix,
            args.dpi,
            args.show,
        )
        return

    # 通常のテスト（単一関数）
    # グリッドの作成
    grid = Grid(args.n_points, tuple(args.x_range))

    # テスターの作成
    tester = CCDTester(grid)

    # テスト関数の選択
    test_funcs = TestFunctionFactory.create_standard_functions()
    selected_func = next(
        (f for f in test_funcs if f.name == args.test_func), test_funcs[0]
    )

    # テストの実行
    print(f"\n{selected_func.name}関数でテストを実行しています...")
    print("ディリクレ境界条件とノイマン境界条件を使用")

    if args.rehu_scaling is not None:
        print(f"Rehuスケーリングを適用（Rehu数: {args.rehu_scaling}）")

    results = tester.run_test_with_options(selected_func, rehu_number=args.rehu_scaling)

    # 結果の表示
    print("\n誤差分析:")
    print(f"  ψ誤差: {results['errors'][0]:.6e}")
    print(f"  ψ'誤差: {results['errors'][1]:.6e}")
    print(f"  ψ''誤差: {results['errors'][2]:.6e}")
    print(f"  ψ'''誤差: {results['errors'][3]:.6e}")

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
            dpi=args.dpi,
        )


if __name__ == "__main__":
    run_cli()
