#!/usr/bin/env python3
"""
拡張されたCCD法コマンドラインインターフェース

1次元と2次元のCCD計算・テストを切り替え可能なツールを提供します。
"""

import argparse
import os

from grid_config import GridConfig
from composite_solver import CCDCompositeSolver
from ccd_tester import CCDMethodTester
from ccd_diagnostics import CCDSolverDiagnostics
from solver_comparator import SolverComparator

# 2次元CCD用のクラスをインポート（存在する場合）
try:
    from grid_config_2d import GridConfig2D
    from composite_solver_2d import CCDCompositeSolver2D
    from ccd_tester_2d import CCDMethodTester2D
    from ccd_diagnostics_2d import CCDSolverDiagnostics2D
    from solver_comparator_2d import SolverComparator2D
    has_2d_support = True
except ImportError:
    has_2d_support = False


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="CCD法の計算・テスト")

    # 親パーサーを作成 - 共通オプション
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--n", type=int, default=32, help="グリッド点の数 (1Dの場合) または x方向のグリッド点の数 (2Dの場合)")
    parent_parser.add_argument("--m", type=int, default=32, help="y方向のグリッド点の数 (2Dの場合のみ使用)")
    parent_parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        help="x軸の範囲 (開始点 終了点)",
    )
    parent_parser.add_argument(
        "--yrange",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        help="y軸の範囲 (2Dの場合のみ使用)(開始点 終了点)",
    )
    parent_parser.add_argument(
        "--coeffs",
        type=float,
        nargs="+",
        default=[1.0, 0.0, 0.0, 0.0],
        help="[a, b, c, d] 係数リスト (f = a*psi + b*psi' + c*psi'' + d*psi''')",
    )
    
    # 次元の指定（1Dまたは2D）
    parent_parser.add_argument(
        "--dim",
        type=int,
        choices=[1, 2],
        default=1,
        help="計算の次元 (1: 1次元, 2: 2次元) デフォルトは1",
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

    # 一覧表示コマンド - こちらも親パーサーから引数を継承
    list_parser = subparsers.add_parser(
        "list", parents=[parent_parser], help="使用可能な設定を一覧表示"
    )

    return parser.parse_args()


def create_1d_config(args):
    """1次元グリッド設定を作成"""
    n = args.n
    x_range = tuple(args.xrange)
    L = x_range[1] - x_range[0]

    # coeffsを設定したグリッド設定を作成
    return GridConfig(n_points=n, h=L / (n - 1), coeffs=args.coeffs)


def create_2d_config(args):
    """2次元グリッド設定を作成"""
    if not has_2d_support:
        raise ImportError("2次元サポートモジュールが見つかりません。モジュールをインストールしてください。")

    nx = args.n
    ny = args.m
    x_range = tuple(args.xrange)
    y_range = tuple(args.yrange)
    Lx = x_range[1] - x_range[0]
    Ly = y_range[1] - y_range[0]

    # 境界値（デフォルトはゼロ）
    boundary_values = {
        "left": [0.0] * ny,
        "right": [0.0] * ny,
        "bottom": [0.0] * nx,
        "top": [0.0] * nx,
    }

    # 2次元グリッド設定を作成
    return GridConfig2D(
        nx_points=nx,
        ny_points=ny,
        hx=Lx / (nx - 1),
        hy=Ly / (ny - 1),
        boundary_values=boundary_values,
        coeffs=args.coeffs
    )


def run_cli():
    """コマンドラインインターフェースの実行"""
    # プラグインを読み込み
    CCDCompositeSolver.load_plugins(silent=True)

    args = parse_args()

    # 次元を確認
    is_2d = args.dim == 2

    # 2次元モードが選択されたが、サポートされていない場合
    if is_2d and not has_2d_support:
        print("警告: 2次元サポートモジュールが見つかりません。1次元モードで実行します。")
        is_2d = False

    try:
        # 次元に応じたグリッド設定を作成
        grid_config = create_2d_config(args) if is_2d else create_1d_config(args)
    except ImportError as e:
        print(f"エラー: {e}")
        return
    
    # 範囲を取得
    if is_2d:
        x_range = tuple(args.xrange)
        y_range = tuple(args.yrange)
    else:
        x_range = tuple(args.xrange)

    # ソルバー設定を準備
    solver_kwargs = {
        "scaling": args.scaling,
        "regularization": args.reg,
    }

    # コマンドの実行
    if args.command == "test":
        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)

        # 次元に応じてテスターを作成
        if is_2d:
            tester = CCDMethodTester2D(
                CCDCompositeSolver2D,
                grid_config,
                (x_range, y_range),
                solver_kwargs=solver_kwargs,
                coeffs=args.coeffs,
            )
        else:
            tester = CCDMethodTester(
                CCDCompositeSolver,
                grid_config,
                x_range,
                solver_kwargs=solver_kwargs,
                coeffs=args.coeffs,
            )

        name = f"{args.scaling}_{args.reg}"
        dim_str = "2D" if is_2d else "1D"
        print(f"{dim_str}テスト実行中: {name} (coeffs={args.coeffs})")
        tester.run_tests(prefix=f"{dim_str.lower()}_{name.lower()}_", visualize=not args.no_viz)

    elif args.command == "diagnostics":
        # 次元に応じて診断クラスを選択
        if is_2d:
            diagnostics = CCDSolverDiagnostics2D(
                CCDCompositeSolver2D, grid_config, solver_kwargs=solver_kwargs
            )
        else:
            diagnostics = CCDSolverDiagnostics(
                CCDCompositeSolver, grid_config, solver_kwargs=solver_kwargs
            )

        name = f"{args.scaling}_{args.reg}"
        dim_str = "2D" if is_2d else "1D"
        print(f"{dim_str}診断実行中: {name} (coeffs={args.coeffs})")
        diagnostics.perform_diagnosis(visualize=args.viz, test_func_name=args.func)

    elif args.command == "compare":
        # 比較モードに応じて構成を設定
        if args.mode == "scaling":
            # スケーリング戦略比較
            if is_2d:
                scaling_methods = CCDCompositeSolver2D.available_scaling_methods()
            else:
                scaling_methods = CCDCompositeSolver.available_scaling_methods()
            configs = [(s.capitalize(), s, "none", {}) for s in scaling_methods]
        else:  # 'reg'
            # 正則化戦略比較
            if is_2d:
                reg_methods = CCDCompositeSolver2D.available_regularization_methods()
            else:
                reg_methods = CCDCompositeSolver.available_regularization_methods()
            configs = [(r.capitalize(), "none", r, {}) for r in reg_methods]

        # ソルバーリストの作成
        solvers_list = []
        for name, scaling, regularization, params in configs:
            params_copy = params.copy()

            if is_2d:
                # 2次元比較
                tester = CCDMethodTester2D(
                    CCDCompositeSolver2D,
                    grid_config,
                    (x_range, y_range),
                    solver_kwargs={
                        "scaling": scaling,
                        "regularization": regularization,
                        **params_copy,
                    },
                    coeffs=args.coeffs,
                )
            else:
                # 1次元比較
                tester = CCDMethodTester(
                    CCDCompositeSolver,
                    grid_config,
                    x_range,
                    solver_kwargs={
                        "scaling": scaling,
                        "regularization": regularization,
                        **params_copy,
                    },
                    coeffs=args.coeffs,
                )
            solvers_list.append((name, tester))

        # 比較クラスを選択
        if is_2d:
            comparator = SolverComparator2D(solvers_list, grid_config, (x_range, y_range))
        else:
            comparator = SolverComparator(solvers_list, grid_config, x_range)

        # 比較実行
        dim_str = "2D" if is_2d else "1D"
        print(f"{dim_str}比較実行中: {len(configs)}個の{args.mode}設定を比較します")
        comparator.run_comparison(
            save_results=True, visualize=not args.no_viz, prefix=f"{dim_str.lower()}_{args.mode}_"
        )

    elif args.command == "list":
        # 利用可能なスケーリング・正則化手法を表示
        if is_2d:
            solver_class = CCDCompositeSolver2D
            dim_str = "2D"
        else:
            solver_class = CCDCompositeSolver
            dim_str = "1D"

        print(f"=== {dim_str}で利用可能なスケーリング戦略 ===")
        for method in solver_class.available_scaling_methods():
            print(f"- {method}")

        print(f"\n=== {dim_str}で利用可能な正則化戦略 ===")
        for method in solver_class.available_regularization_methods():
            print(f"- {method}")


if __name__ == "__main__":
    run_cli()