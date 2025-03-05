#!/usr/bin/env python3
"""
2次元CCD法コマンドラインインターフェース

2次元結合コンパクト差分法のコマンドラインツールを提供します。
"""

import argparse
import os
import json
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from grid2d_config import Grid2DConfig
from ccd2d_solver import CCD2DSolver
from test2d_functions import Test2DFunction, Test2DFunctionFactory
from ccd2d_tester import CCD2DMethodTester
from visualization2d_utils import (
    visualize_2d_field, 
    visualize_2d_surface, 
    visualize_error_field,
    visualize_derivative_comparison,
    visualize_all_results
)


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="2次元CCD法の計算・テスト")

    # 親パーサーを作成 - 共通オプション
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--nx", type=int, default=41, help="x方向のグリッド点の数")
    parent_parser.add_argument("--ny", type=int, default=41, help="y方向のグリッド点の数")
    parent_parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="x軸の範囲 (開始点 終了点)",
    )
    parent_parser.add_argument(
        "--yrange",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="y軸の範囲 (開始点 終了点)",
    )
    parent_parser.add_argument(
        "--coeff",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        help="係数を設定 例: --coeff a 1.0 --coeff bx 0.5",
    )
    parent_parser.add_argument(
        "--x-order", type=int, default=3, help="x方向の最大微分階数"
    )
    parent_parser.add_argument(
        "--y-order", type=int, default=3, help="y方向の最大微分階数"
    )
    parent_parser.add_argument(
        "--output-dir", type=str, default="ccd2d_results", help="出力ディレクトリ"
    )
    parent_parser.add_argument(
        "--solver",
        type=str,
        choices=["direct", "gmres", "cg", "bicgstab", "lsqr"],
        default="gmres",
        help="ソルバーの種類",
    )
    parent_parser.add_argument(
        "--no-viz", action="store_true", help="可視化を無効化"
    )

    # サブコマンド
    subparsers = parser.add_subparsers(
        dest="command", help="実行するコマンド", required=True
    )

    # 計算コマンド - 親パーサーから引数を継承
    compute_parser = subparsers.add_parser(
        "compute", parents=[parent_parser], help="指定したテスト関数で計算を実行"
    )
    compute_parser.add_argument(
        "--function", type=str, default="Sine2D", help="テスト関数の名前"
    )
    compute_parser.add_argument(
        "--save-data", action="store_true", help="計算結果をNumPy配列として保存"
    )

    # テストコマンド - 親パーサーから引数を継承
    test_parser = subparsers.add_parser(
        "test", parents=[parent_parser], help="テストを実行"
    )
    test_parser.add_argument(
        "--function", type=str, default=None, help="特定のテスト関数のみを使用（省略時は全関数）"
    )
    test_parser.add_argument(
        "--prefix", type=str, default="", help="出力ファイル名の接頭辞"
    )

    # 収束性調査コマンド - 親パーサーから引数を継承
    convergence_parser = subparsers.add_parser(
        "convergence", parents=[parent_parser], help="収束性の調査を実行"
    )
    convergence_parser.add_argument(
        "--function", type=str, default="Sine2D", help="テスト関数の名前"
    )
    convergence_parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[11, 21, 41, 81],
        help="テストするグリッドサイズのリスト",
    )
    convergence_parser.add_argument(
        "--rect-grid", action="store_true", help="長方形グリッドを使用（デフォルトは正方形）"
    )

    # CSVデータ読み込みコマンド - 親パーサーから引数を継承
    csv_parser = subparsers.add_parser(
        "csv", parents=[parent_parser], help="CSVファイルから計算を実行"
    )
    csv_parser.add_argument(
        "--input-file", type=str, required=True, help="入力CSVファイルのパス"
    )
    csv_parser.add_argument(
        "--has-header", action="store_true", help="CSVファイルにヘッダがある場合に指定"
    )
    csv_parser.add_argument(
        "--delimiter", type=str, default=",", help="CSVの区切り文字"
    )

    # 設定プリセットコマンド
    preset_parser = subparsers.add_parser(
        "preset", help="プリセット設定を管理"
    )
    preset_subparsers = preset_parser.add_subparsers(
        dest="preset_command", help="プリセットコマンド", required=True
    )
    
    # プリセット保存コマンド
    preset_save_parser = preset_subparsers.add_parser(
        "save", parents=[parent_parser], help="現在の設定をプリセットとして保存"
    )
    preset_save_parser.add_argument(
        "--name", type=str, required=True, help="プリセット名"
    )
    
    # プリセット読み込みコマンド
    preset_load_parser = preset_subparsers.add_parser(
        "load", help="プリセット設定を読み込み"
    )
    preset_load_parser.add_argument(
        "--name", type=str, required=True, help="プリセット名"
    )
    preset_load_parser.add_argument(
        "--command", type=str, required=True, choices=["compute", "test", "convergence", "csv"],
        help="読み込み後に実行するコマンド"
    )
    
    # プリセット一覧表示コマンド
    preset_subparsers.add_parser(
        "list", help="利用可能なプリセットを一覧表示"
    )

    return parser.parse_args()


def find_test_function(name: str) -> Test2DFunction:
    """指定した名前のテスト関数を検索"""
    test_functions = Test2DFunctionFactory.create_standard_functions()
    for func in test_functions:
        if func.name.lower() == name.lower():
            return func
    
    # 見つからなかった場合、デフォルトを返す
    print(f"警告: '{name}'という名前のテスト関数が見つかりませんでした。代わりにSine2Dを使用します。")
    for func in test_functions:
        if func.name == "Sine2D":
            return func
    
    # それでも見つからなければ、最初の関数を返す
    return test_functions[0]


def create_grid_config(args) -> Grid2DConfig:
    """コマンドライン引数からグリッド設定を作成"""
    nx, ny = args.nx, args.ny
    x_range = tuple(args.xrange)
    y_range = tuple(args.yrange)
    
    # グリッド幅を計算
    hx = (x_range[1] - x_range[0]) / (nx - 1)
    hy = (y_range[1] - y_range[0]) / (ny - 1)
    
    # 係数辞書を構築
    coeffs = {}
    if args.coeff:
        for name, value in args.coeff:
            coeffs[name] = float(value)
    
    # グリッド設定オブジェクトを作成
    return Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        x_deriv_order=args.x_order,
        y_deriv_order=args.y_order,
        coeffs=coeffs
    )


def get_solver_kwargs(args) -> Dict[str, Any]:
    """コマンドライン引数からソルバーパラメータを作成"""
    if args.solver == "direct":
        return {"use_iterative": False}
    else:
        return {
            "use_iterative": True,
            "solver_type": args.solver,
            "solver_kwargs": {"tol": 1e-10, "maxiter": 1000}
        }


def run_compute_command(args):
    """computeコマンドを実行"""
    print(f"\n=== 2次元CCD計算 ({args.function}) ===")
    
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # テスト関数を検索
    test_func = find_test_function(args.function)
    print(f"テスト関数: {test_func.name}")
    
    # グリッド点でのx, y座標を生成
    x_points = cp.linspace(args.xrange[0], args.xrange[1], args.nx)
    y_points = cp.linspace(args.yrange[0], args.yrange[1], args.ny)
    X, Y = cp.meshgrid(x_points, y_points, indexing='ij')
    
    # テスト関数をグリッド上で評価
    f_values = test_func.f(X, Y)
    
    # 解析解も計算
    analytical_results = Test2DFunctionFactory.evaluate_function_on_grid(
        test_func, X, Y
    )
    
    # ソルバーパラメータを取得
    solver_kwargs = get_solver_kwargs(args)
    
    # 2次元CCDソルバーの作成
    solver = CCD2DSolver(grid_config, **solver_kwargs)
    
    # システム情報の表示
    system_info = solver.get_system_info()
    print("\nシステム情報:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 数値解の計算
    numerical_results = solver.solve(f_values)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 結果の保存（オプション）
    if args.save_data:
        # NumPy配列として保存
        data_dir = os.path.join(args.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 数値解を保存
        for key, array in numerical_results.items():
            np.save(os.path.join(data_dir, f"{test_func.name}_{key}_numerical.npy"), cp.asnumpy(array))
        
        # 解析解も保存
        for key, array in analytical_results.items():
            np.save(os.path.join(data_dir, f"{test_func.name}_{key}_analytical.npy"), cp.asnumpy(array))
        
        print(f"データを {data_dir} に保存しました")
    
    # 可視化（オプション）
    if not args.no_viz:
        # すべての結果を可視化
        error_metrics = visualize_all_results(
            numerical_results,
            analytical_results,
            grid_config,
            test_func_name=test_func.name,
            output_dir=args.output_dir,
            prefix=""
        )
        
        # 誤差メトリクスを表示
        print("\n誤差メトリクス:")
        for deriv_key, metrics in error_metrics.items():
            print(f"  {deriv_key}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.2e}")
        
        print(f"可視化結果を {args.output_dir} に保存しました")
    
    print("\n計算が完了しました")
    return 0


def run_test_command(args):
    """testコマンドを実行"""
    print("\n=== 2次元CCDテスト ===")
    
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # ソルバーパラメータを取得
    solver_kwargs = get_solver_kwargs(args)
    
    # テスト関数を選択（指定があれば特定の関数のみ）
    if args.function:
        test_func = find_test_function(args.function)
        test_functions = [test_func]
        print(f"テスト関数: {test_func.name}")
    else:
        test_functions = Test2DFunctionFactory.create_standard_functions()
        print(f"テスト関数: すべて ({len(test_functions)}個)")
    
    # テスターの作成
    tester = CCD2DMethodTester(
        CCD2DSolver,
        grid_config,
        x_range=tuple(args.xrange),
        y_range=tuple(args.yrange),
        solver_kwargs=solver_kwargs,
        test_functions=test_functions
    )
    
    # テストを実行
    results = tester.run_tests(
        prefix=args.prefix,
        visualize=not args.no_viz,
        output_dir=args.output_dir
    )
    
    print(f"\nテストが完了しました。結果は {args.output_dir} に保存されています。")
    return 0


def run_convergence_command(args):
    """convergenceコマンドを実行"""
    print("\n=== 2次元CCD収束性調査 ===")
    
    # 最小グリッドサイズでグリッド設定を作成
    min_grid_size = min(args.grid_sizes)
    args.nx = args.ny = min_grid_size
    grid_config = create_grid_config(args)
    
    # ソルバーパラメータを取得
    solver_kwargs = get_solver_kwargs(args)
    
    # テスト関数を選択
    test_func = find_test_function(args.function)
    print(f"テスト関数: {test_func.name}")
    print(f"グリッドサイズ: {args.grid_sizes}")
    
    # テスターの作成
    tester = CCD2DMethodTester(
        CCD2DSolver,
        grid_config,
        x_range=tuple(args.xrange),
        y_range=tuple(args.yrange),
        solver_kwargs=solver_kwargs,
        test_functions=[test_func]
    )
    
    # 収束性の調査を実行
    convergence_results = tester.convergence_study(
        test_func,
        args.grid_sizes,
        is_square_grid=not args.rect_grid,
        output_dir=args.output_dir
    )
    
    print(f"\n収束性調査が完了しました。結果は {args.output_dir} に保存されています。")
    return 0


def run_csv_command(args):
    """csvコマンドを実行"""
    print("\n=== CSVファイルからの2次元CCD計算 ===")
    
    # CSVファイルの存在確認
    if not os.path.exists(args.input_file):
        print(f"エラー: 入力ファイル '{args.input_file}' が見つかりません。")
        return 1
    
    # CSVファイルの読み込み
    try:
        import pandas as pd
        df = pd.read_csv(args.input_file, header=0 if args.has_header else None, delimiter=args.delimiter)
        print(f"CSVファイル '{args.input_file}' を読み込みました（{df.shape[0]}行 x {df.shape[1]}列）")
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return 1
    
    # データの確認と2次元配列への変換
    if df.shape[0] < 2 or df.shape[1] < 2:
        print(f"エラー: CSVデータが小さすぎます。少なくとも2x2のデータが必要です。")
        return 1
    
    # グリッドサイズの更新
    args.nx = df.shape[0]
    args.ny = df.shape[1]
    
    # グリッド設定を作成
    grid_config = create_grid_config(args)
    
    # ソルバーパラメータを取得
    solver_kwargs = get_solver_kwargs(args)
    
    # CSVデータをCuPy配列に変換
    f_values = cp.array(df.values)
    
    # 2次元CCDソルバーの作成
    solver = CCD2DSolver(grid_config, **solver_kwargs)
    
    # システム情報の表示
    system_info = solver.get_system_info()
    print("\nシステム情報:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 数値解の計算
    numerical_results = solver.solve(f_values)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 結果の保存
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 数値解を保存
    for key, array in numerical_results.items():
        np.save(os.path.join(data_dir, f"csv_{key}_numerical.npy"), cp.asnumpy(array))
    
    # 入力データも保存
    np.save(os.path.join(data_dir, "csv_input.npy"), cp.asnumpy(f_values))
    
    print(f"計算結果を {data_dir} に保存しました")
    
    # 可視化（オプション）
    if not args.no_viz:
        # 入力データの可視化
        visualize_2d_field(
            f_values, 
            grid_config, 
            title="Input Data from CSV",
            save_path=os.path.join(args.output_dir, "csv_input_field.png")
        )
        
        # 数値解の主要な導関数を可視化
        for key in ["f_x", "f_y", "f_xx", "f_yy"]:
            if key in numerical_results:
                visualize_2d_field(
                    numerical_results[key], 
                    grid_config, 
                    title=f"Derivative: {key}",
                    save_path=os.path.join(args.output_dir, f"csv_{key}_field.png")
                )
        
        print(f"可視化結果を {args.output_dir} に保存しました")
    
    print("\n計算が完了しました")
    return 0


def run_preset_command(args):
    """プリセット関連コマンドを実行"""
    # プリセットディレクトリの作成
    preset_dir = "presets"
    os.makedirs(preset_dir, exist_ok=True)
    
    if args.preset_command == "save":
        # 現在の設定をプリセットとして保存
        preset_data = vars(args).copy()
        
        # 不要なキーを削除
        for key in ["preset_command", "command", "name"]:
            if key in preset_data:
                del preset_data[key]
        
        # プリセットファイルに保存
        preset_file = os.path.join(preset_dir, f"{args.name}.json")
        with open(preset_file, "w") as f:
            json.dump(preset_data, f, indent=2)
        
        print(f"プリセット '{args.name}' を保存しました")
        return 0
    
    elif args.preset_command == "load":
        # プリセットファイルの読み込み
        preset_file = os.path.join(preset_dir, f"{args.name}.json")
        if not os.path.exists(preset_file):
            print(f"エラー: プリセット '{args.name}' が見つかりません")
            return 1
        
        # プリセットデータの読み込み
        with open(preset_file, "r") as f:
            preset_data = json.load(f)
        
        print(f"プリセット '{args.name}' を読み込みました")
        
        # 引数オブジェクトにプリセットデータをマージ
        args_dict = vars(args)
        for key, value in preset_data.items():
            args_dict[key] = value
        
        # 指定されたコマンドを実行
        args.command = args.command  # このまま維持
        return run_command(args)
    
    elif args.preset_command == "list":
        # 利用可能なプリセットを一覧表示
        preset_files = [f for f in os.listdir(preset_dir) if f.endswith(".json")]
        if not preset_files:
            print("利用可能なプリセットはありません")
            return 0
        
        print("利用可能なプリセット:")
        for preset_file in preset_files:
            preset_name = os.path.splitext(preset_file)[0]
            
            # プリセットの概要情報を読み込む
            try:
                with open(os.path.join(preset_dir, preset_file), "r") as f:
                    preset_data = json.load(f)
                
                nx = preset_data.get("nx", "?")
                ny = preset_data.get("ny", "?")
                solver = preset_data.get("solver", "?")
                
                print(f"  {preset_name}: {nx}x{ny} グリッド, {solver} ソルバー")
            except:
                print(f"  {preset_name}: (読み込みエラー)")
        
        return 0
    
    return 1  # 未知のプリセットコマンド


def run_command(args):
    """コマンドに応じて適切な処理を実行"""
    if args.command == "compute":
        return run_compute_command(args)
    elif args.command == "test":
        return run_test_command(args)
    elif args.command == "convergence":
        return run_convergence_command(args)
    elif args.command == "csv":
        return run_csv_command(args)
    elif args.command == "preset":
        return run_preset_command(args)
    else:
        print(f"エラー: 未知のコマンド '{args.command}'")
        return 1


def main():
    """メインエントリーポイント"""
    args = parse_args()
    return run_command(args)


if __name__ == "__main__":
    exit(main())
