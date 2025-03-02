#!/usr/bin/env python3
"""
簡素化されたCCD法コマンドラインインターフェース（ノイマン・ディリクレ境界条件対応）

最も重要な機能に焦点を当てたシンプルなコマンドラインツールを提供します。
"""

import argparse
import os

from ccd_core import GridConfig
from composite_solver import CCDCompositeSolver
from ccd_tester import CCDMethodTester
from ccd_diagnostics import CCDSolverDiagnostics
from solver_comparator import SolverComparator


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="CCD法の計算・テスト")

    # 親パーサーを作成 - 共通オプション
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--n", type=int, default=256, help="グリッド点の数")
    parent_parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        help="x軸の範囲 (開始点 終了点)",
    )
    parent_parser.add_argument(
        "--coeffs",
        type=float,
        nargs="+",
        default=[1.0, 0.0, 0.0, 0.0],
        help="[a, b, c, d] 係数リスト (f = a*psi + b*psi' + c*psi'' + d*psi''')",
    )

    # サブコマンド
    subparsers = parser.add_subparsers(
        dest="command", help="実行するコマンド", required=True
    )

    # テストコマンド - 親パーサーから引数を継承
    test_parser = subparsers.add_parser(
        "test", parents=[parent_parser], help="テストを実行"
    )
    test_parser.add_argument(
        "--scaling", type=str, default="none", help="スケーリング手法"
    )
    test_parser.add_argument("--reg", type=str, default="none", help="正則化手法")
    test_parser.add_argument("--no-viz", action="store_true", help="可視化を無効化")

    # 診断コマンド - 親パーサーから引数を継承
    diag_parser = subparsers.add_parser(
        "diagnostics", parents=[parent_parser], help="診断を実行"
    )
    diag_parser.add_argument(
        "--scaling", type=str, default="none", help="スケーリング手法"
    )
    diag_parser.add_argument("--reg", type=str, default="none", help="正則化手法")
    diag_parser.add_argument("--viz", action="store_true", help="可視化を有効化")

    # 比較コマンド - 親パーサーから引数を継承
    compare_parser = subparsers.add_parser(
        "compare", parents=[parent_parser], help="ソルバー間の比較を実行"
    )
    compare_parser.add_argument(
        "--mode", choices=["scaling", "reg"], default="reg", help="比較モード"
    )
    compare_parser.add_argument("--no-viz", action="store_true", help="可視化を無効化")

    # 一覧表示コマンド - こちらも親パーサーから引数を継承（必要に応じて）
    list_parser = subparsers.add_parser(
        "list", parents=[parent_parser], help="使用可能な設定を一覧表示"
    )

    return parser.parse_args()


def run_cli():
    """コマンドラインインターフェースの実行"""
    # プラグインを読み込み
    CCDCompositeSolver.load_plugins(silent=True)

    args = parse_args()

    # グリッド設定
    n = args.n
    x_range = tuple(args.xrange)
    L = x_range[1] - x_range[0]

    # コマンドの実行
    if args.command == "test":
        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)

        # 基本的なグリッド設定を作成（境界条件は後でccd_testerで設定される）
        grid_config = GridConfig(n_points=n, h=L / (n - 1))

        # ソルバー設定を準備（境界条件はccd_testerで設定されるため、ここでは指定しない）
        solver_kwargs = {
            "scaling": args.scaling,
            "regularization": args.reg,
            "coeffs": args.coeffs,
        }

        # テストを実行
        tester = CCDMethodTester(
            CCDCompositeSolver,
            grid_config,
            x_range,
            solver_kwargs=solver_kwargs,
            coeffs=args.coeffs,
        )

        name = f"{args.scaling}_{args.reg}"
        print(f"テスト実行中: {name} (coeffs={args.coeffs})")
        tester.run_tests(prefix=f"{name.lower()}_", visualize=not args.no_viz)

    elif args.command == "diagnostics":
        # 基本的なグリッド設定を作成
        grid_config = GridConfig(n_points=n, h=L / (n - 1))

        # 診断用ソルバーの作成
        solver_kwargs = {
            "scaling": args.scaling,
            "regularization": args.reg,
            "coeffs": args.coeffs,
        }

        # 診断を実行
        diagnostics = CCDSolverDiagnostics(
            CCDCompositeSolver, grid_config, solver_kwargs
        )

        name = f"{args.scaling}_{args.reg}"
        print(f"診断実行中: {name} (coeffs={args.coeffs})")
        diagnostics.perform_full_diagnosis(visualize=args.viz)

    elif args.command == "compare":
        # 比較モードに応じて構成を設定
        if args.mode == "scaling":
            # スケーリング戦略比較
            scaling_methods = CCDCompositeSolver.available_scaling_methods()
            configs = [(s.capitalize(), s, "none", {}) for s in scaling_methods]
        else:  # 'reg'
            # 正則化戦略比較
            reg_methods = CCDCompositeSolver.available_regularization_methods()
            configs = [(r.capitalize(), "none", r, {}) for r in reg_methods]

        # 基本的なグリッド設定を作成
        grid_config = GridConfig(n_points=n, h=L / (n - 1))

        # ソルバーリストの作成
        solvers_list = []
        for name, scaling, regularization, params in configs:
            params_copy = params.copy()
            params_copy["coeffs"] = args.coeffs

            # ソルバー用のグリッド設定を作成
            solver_grid_config = GridConfig(
                n_points=grid_config.n_points, h=grid_config.h
            )

            tester = CCDMethodTester(
                CCDCompositeSolver,
                solver_grid_config,
                x_range,
                solver_kwargs={
                    "scaling": scaling,
                    "regularization": regularization,
                    **params_copy,
                },
                coeffs=args.coeffs,
            )
            solvers_list.append((name, tester))

        # 比較実行
        print(f"比較実行中: {len(configs)}個の{args.mode}設定を比較します")
        comparator = SolverComparator(solvers_list, grid_config, x_range)
        comparator.run_comparison(
            save_results=True, visualize=not args.no_viz, prefix=f"{args.mode}_"
        )

    elif args.command == "list":
        # 利用可能なスケーリング・正則化手法を表示
        print("=== 利用可能なスケーリング戦略 ===")
        for method in CCDCompositeSolver.available_scaling_methods():
            print(f"- {method}")

        print("\n=== 利用可能な正則化戦略 ===")
        for method in CCDCompositeSolver.available_regularization_methods():
            print(f"- {method}")


if __name__ == "__main__":
    run_cli()
