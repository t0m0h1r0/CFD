#!/usr/bin/env python3
"""
疎行列対応CCD法コマンドラインインターフェース

メモリ使用量を削減した疎行列版のコマンドラインツールを提供します。
"""

import argparse
import os

from grid_config import GridConfig
from sparse_ccd_solver import SparseCompositeSolver
from ccd_tester import CCDMethodTester
from sparse_ccd_tester import SparseCCDMethodTester
from ccd_diagnostics import CCDSolverDiagnostics
from solver_comparator import SolverComparator


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="CCD法の計算・テスト（疎行列対応版）")

    # 親パーサーを作成 - 共通オプション
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--n", type=int, default=32, help="グリッド点の数")
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
    parent_parser.add_argument(
        "--sparse", action="store_true", help="疎行列ソルバーを使用（メモリ使用量削減）"
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
    diag_parser.add_argument(
        "--func", type=str, default="Sine", help="個別テストに使用するテスト関数の名前"
    )

    # 比較コマンド - 親パーサーから引数を継承
    compare_parser = subparsers.add_parser(
        "compare", parents=[parent_parser], help="ソルバー間の比較を実行"
    )
    compare_parser.add_argument(
        "--mode", choices=["scaling", "reg"], default="reg", help="比較モード"
    )
    compare_parser.add_argument("--no-viz", action="store_true", help="可視化を無効化")
    compare_parser.add_argument(
        "--dense-vs-sparse", action="store_true", 
        help="密行列と疎行列ソルバーを比較（他のオプションより優先）"
    )

    # 一覧表示コマンド - こちらも親パーサーから引数を継承（必要に応じて）
    list_parser = subparsers.add_parser(
        "list", parents=[parent_parser], help="使用可能な設定を一覧表示"
    )

    return parser.parse_args()


def run_cli():
    """コマンドラインインターフェースの実行"""
    # プラグインを読み込み
    SparseCompositeSolver.load_plugins(silent=True)

    args = parse_args()

    # グリッド設定
    n = args.n
    x_range = tuple(args.xrange)
    L = x_range[1] - x_range[0]
    
    # coeffsを設定したグリッド設定を作成
    grid_config = GridConfig(n_points=n, h=L / (n - 1), coeffs=args.coeffs)

    # 適切なソルバークラスを選択
    if args.sparse:
        print("疎行列ソルバーを使用")
        solver_class = SparseCompositeSolver
        tester_class = SparseCCDMethodTester
    else:
        # 従来の密行列ソルバーを使用
        from composite_solver import CCDCompositeSolver
        solver_class = CCDCompositeSolver
        tester_class = CCDMethodTester

    # コマンドの実行
    if args.command == "test":
        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)

        # ソルバー設定を準備
        solver_kwargs = {
            "scaling": args.scaling,
            "regularization": args.reg,
        }

        # テストを実行
        tester = tester_class(
            solver_class,
            grid_config,
            x_range,
            solver_kwargs=solver_kwargs,
            coeffs=args.coeffs,
        )

        name = f"{args.scaling}_{args.reg}"
        prefix = "sparse_" if args.sparse else ""
        print(f"テスト実行中: {prefix}{name} (coeffs={args.coeffs})")
        tester.run_tests(prefix=f"{prefix}{name.lower()}_", visualize=not args.no_viz)

    elif args.command == "diagnostics":
        # 診断用ソルバーの作成
        solver_kwargs = {
            "scaling": args.scaling,
            "regularization": args.reg,
        }

        # 診断を実行
        diagnostics = CCDSolverDiagnostics(
            solver_class, grid_config, solver_kwargs=solver_kwargs
        )

        name = f"{args.scaling}_{args.reg}"
        prefix = "Sparse" if args.sparse else ""
        print(f"診断実行中: {prefix}{name} (coeffs={args.coeffs})")
        diagnostics.perform_diagnosis(visualize=args.viz, test_func_name=args.func)

    elif args.command == "compare":
        # 密行列対疎行列比較モード
        if args.dense_vs_sparse:
            # 密行列と疎行列ソルバーを比較
            from composite_solver import CCDCompositeSolver

            # グリッド設定
            dense_grid_config = GridConfig(
                n_points=grid_config.n_points, 
                h=grid_config.h,
                coeffs=args.coeffs
            )
            sparse_grid_config = GridConfig(
                n_points=grid_config.n_points, 
                h=grid_config.h,
                coeffs=args.coeffs
            )

            # テスターを作成
            dense_tester = CCDMethodTester(
                CCDCompositeSolver,
                dense_grid_config,
                x_range,
                solver_kwargs={
                    "scaling": "none",
                    "regularization": "none",
                },
                coeffs=args.coeffs,
            )
            
            sparse_tester = SparseCCDMethodTester(
                SparseCompositeSolver,
                sparse_grid_config,
                x_range,
                solver_kwargs={
                    "scaling": "none",
                    "regularization": "none",
                },
                coeffs=args.coeffs,
            )

            # ソルバーリスト
            solvers_list = [
                ("Dense", dense_tester),
                ("Sparse", sparse_tester),
            ]

            # 比較実行
            print("密行列と疎行列の比較実行中")
            comparator = SolverComparator(solvers_list, grid_config, x_range)
            comparator.run_comparison(
                save_results=True, visualize=not args.no_viz, prefix="dense_vs_sparse_"
            )
            
        else:
            # 通常の比較モード（スケーリングまたは正則化）
            # 比較モードに応じて構成を設定
            if args.mode == "scaling":
                # スケーリング戦略比較
                scaling_methods = solver_class.available_scaling_methods()
                configs = [(s.capitalize(), s, "none", {}) for s in scaling_methods]
            else:  # 'reg'
                # 正則化戦略比較
                reg_methods = solver_class.available_regularization_methods()
                configs = [(r.capitalize(), "none", r, {}) for r in reg_methods]

            # ソルバーリストの作成
            solvers_list = []
            for name, scaling, regularization, params in configs:
                params_copy = params.copy()
                
                # 各ソルバー用のグリッド設定を作成（coeffsを含む）
                solver_grid_config = GridConfig(
                    n_points=grid_config.n_points, 
                    h=grid_config.h,
                    coeffs=args.coeffs
                )

                tester = tester_class(
                    solver_class,
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
            prefix = "sparse_" if args.sparse else ""
            comparator = SolverComparator(solvers_list, grid_config, x_range)
            comparator.run_comparison(
                save_results=True, visualize=not args.no_viz, prefix=f"{prefix}{args.mode}_"
            )

    elif args.command == "list":
        # 利用可能なスケーリング・正則化手法を表示
        print("=== 利用可能なスケーリング戦略 ===")
        for method in solver_class.available_scaling_methods():
            print(f"- {method}")

        print("\n=== 利用可能な正則化戦略 ===")
        for method in solver_class.available_regularization_methods():
            print(f"- {method}")


if __name__ == "__main__":
    run_cli()
