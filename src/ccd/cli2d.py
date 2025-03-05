#!/usr/bin/env python3
"""
2次元CCD法コマンドラインインターフェース

コマンドラインから2次元偏微分方程式を解くためのインターフェース。
cli.pyに準じたインターフェースとコマンドオプションを提供します。
"""

import argparse
import os
import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# インポートパスを追加（必要に応じて変更）
sys.path.append('.')
sys.path.append('..')

from grid_2d_config import Grid2DConfig
from ccd2d_solver import CCD2DSolver
from ccd2d_tester import CCD2DMethodTester, TestFunction2D, TestFunctionFactory2D


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="2次元CCD法による偏微分方程式の解法")

    # 親パーサー（共通オプション）
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--nx", type=int, default=32, help="x方向のグリッド点数")
    parent_parser.add_argument("--ny", type=int, default=32, help="y方向のグリッド点数")
    parent_parser.add_argument("--xrange", type=float, nargs=2, default=[0.0, 1.0], help="x軸の範囲 (開始点 終了点)")
    parent_parser.add_argument("--yrange", type=float, nargs=2, default=[0.0, 1.0], help="y軸の範囲 (開始点 終了点)")
    parent_parser.add_argument("--coeffs", type=float, nargs="+", default=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 
                              help="係数 [a, b, c, d, e, f]: a*u + b*ux + c*uxx + d*uy + e*uyy + f*uxy")

    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド", required=True)

    # テストを実行するコマンド
    test_parser = subparsers.add_parser("test", parents=[parent_parser], help="2次元CCD法のテストを実行")
    test_parser.add_argument("--scaling", type=str, default="none", help="スケーリング手法")
    test_parser.add_argument("--reg", type=str, default="none", help="正則化手法")
    test_parser.add_argument("--no-viz", action="store_true", help="可視化を無効化")

    # 診断コマンド
    diag_parser = subparsers.add_parser("diagnostics", parents=[parent_parser], help="診断を実行")
    diag_parser.add_argument("--scaling", type=str, default="none", help="スケーリング手法")
    diag_parser.add_argument("--reg", type=str, default="none", help="正則化手法")
    diag_parser.add_argument("--viz", action="store_true", help="可視化を有効化")
    diag_parser.add_argument("--func", type=str, default="SinProduct", help="個別テストに使用するテスト関数の名前")

    # 比較コマンド
    compare_parser = subparsers.add_parser("compare", parents=[parent_parser], help="ソルバー間の比較を実行")
    compare_parser.add_argument("--mode", choices=["scaling", "reg"], default="reg", help="比較モード")
    compare_parser.add_argument("--no-viz", action="store_true", help="可視化を無効化")

    # 一覧表示コマンド
    list_parser = subparsers.add_parser("list", parents=[parent_parser], help="使用可能な設定を一覧表示")

    return parser.parse_args()


def create_grid_config(args) -> Grid2DConfig:
    """
    コマンドライン引数からグリッド設定を作成
    
    Args:
        args: コマンドライン引数
        
    Returns:
        Grid2DConfig: 2次元グリッド設定
    """
    nx, ny = args.nx, args.ny
    x_range = args.xrange
    y_range = args.yrange
    
    # グリッド幅を計算
    hx = (x_range[1] - x_range[0]) / (nx - 1)
    hy = (y_range[1] - y_range[0]) / (ny - 1)
    
    # グリッド設定を作成
    return Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        coeffs=args.coeffs
    )


def run_test(args):
    """
    2次元CCD法のテストを実行
    
    Args:
        args: コマンドライン引数
    """
    # 出力ディレクトリの作成
    os.makedirs("results", exist_ok=True)

    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # ソルバーのパラメータ（スケーリングと正則化の設定）
    # 注意: CCD2DSolverは直接scaling/regularizationを受け取れないため、
    # ここでは単にuse_direct_solverのみを設定する
    solver_kwargs = {
        "use_direct_solver": True
    }
    
    # 将来的にスケーリングと正則化が実装された場合に備えて情報を表示
    if args.scaling != "none" or args.reg != "none":
        print(f"警告: スケーリング '{args.scaling}' と正則化 '{args.reg}' は現在の2D実装ではサポートされていません")
        print("現在はデフォルト設定を使用します")
    
    # テスターの初期化
    tester = CCD2DMethodTester(
        grid_config=grid_config,
        solver_kwargs=solver_kwargs,
        coeffs=args.coeffs
    )
    
    # テスト実行
    name = f"{args.scaling}_{args.reg}"
    print(f"テスト実行中: {name} (coeffs={args.coeffs})")
    tester.run_tests(prefix=f"{name.lower()}_", visualize=not args.no_viz)


def run_diagnostics(args):
    """
    2次元CCD法の診断を実行
    
    Args:
        args: コマンドライン引数
    """
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # ソルバーのパラメータ
    # 注意: CCD2DSolverは直接scaling/regularizationを受け取れないため、
    # ここでは単にuse_direct_solverのみを設定する
    solver_kwargs = {
        "use_direct_solver": True
    }
    
    # 将来的にスケーリングと正則化が実装された場合に備えて情報を表示
    if args.scaling != "none" or args.reg != "none":
        print(f"警告: スケーリング '{args.scaling}' と正則化 '{args.reg}' は現在の2D実装ではサポートされていません")
        print("現在はデフォルト設定を使用します")
    
    # テスト関数の取得
    test_functions = TestFunctionFactory2D.create_standard_functions()
    test_func = next((f for f in test_functions if f.name == args.func), test_functions[0])
    
    # テスターの初期化
    tester = CCD2DMethodTester(
        grid_config=grid_config,
        solver_kwargs=solver_kwargs,
        coeffs=args.coeffs
    )
    
    name = f"{args.scaling}_{args.reg}"
    print(f"診断実行中: {name} (coeffs={args.coeffs})")
    
    # 単一の関数に対する詳細分析を実行
    result = tester.analyze_single_function(
        test_func=test_func,
        prefix=f"{name.lower()}_diag_"
    )
    
    # 結果の表示
    print(f"\n=== {args.func} の診断結果 ===")
    print(f"L2誤差 (u): {result['errors']['u']:.6e}")
    for k, v in result['errors'].items():
        if k != 'u':
            print(f"L2誤差 ({k}): {v:.6e}")
    print(f"計算時間: {result['elapsed_time']:.4f}秒")


def run_comparison(args):
    """
    異なる設定でのCCD法の比較を実行
    
    Args:
        args: コマンドライン引数
    """
    print("比較モード: 現在は実装されていません")
    print("将来的にはcli.pyに準じて異なるスケーリングや正則化の方法を比較する機能を実装予定です")


def show_available_settings():
    """
    利用可能な設定を表示
    """
    print("=== 2次元CCD法の設定 ===")
    
    # 利用可能なテスト関数
    test_functions = TestFunctionFactory2D.create_standard_functions()
    
    print("\n利用可能なテスト関数:")
    for i, func in enumerate(test_functions):
        print(f"{i+1}. {func.name}")
    
    # 係数の説明
    print("\n係数の説明 [a, b, c, d, e, f]:")
    print("対応する方程式: a*u + b*ux + c*uxx + d*uy + e*uyy + f*uxy = g(x,y)")
    print("例:")
    print("- ポアソン方程式 (∇²u = g): [0, 0, 1, 0, 1, 0]")
    print("- 熱拡散方程式 (∂u/∂t = α∇²u): [1, 0, -α, 0, -α, 0]")
    print("- 移流拡散方程式 (a∇²u + b∇u = g): [0, b, a, b, a, 0]")
    
    # スケーリングと正則化の設定（将来的に拡張予定）
    print("\nスケーリング手法:")
    print("- none: スケーリングなし")
    
    print("\n正則化手法:")
    print("- none: 正則化なし")


def run_cli():
    """コマンドラインインターフェースの実行"""
    args = parse_args()
    
    # コマンドに応じて処理を分岐
    if args.command == "test":
        run_test(args)
    elif args.command == "diagnostics":
        run_diagnostics(args)
    elif args.command == "compare":
        run_comparison(args)
    elif args.command == "list":
        show_available_settings()
    else:
        print(f"未知のコマンド: {args.command}")


if __name__ == "__main__":
    run_cli()