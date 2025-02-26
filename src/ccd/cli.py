"""
コマンドラインインターフェースモジュール - プラグイン対応版

CCD法のテスト・診断・比較のためのシンプルなコマンドラインツールを提供します。
"""

import argparse
import os
from typing import Dict, Any, List, Tuple

from ccd_core import GridConfig
from unified_solver import CCDCompositeSolver
from presets import get_combined_presets, get_preset_by_name
from ccd_tester import CCDMethodTester
from ccd_diagnostics import CCDSolverDiagnostics
from solver_comparator import SolverComparator


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description='CCD法のテストと診断（プラグイン対応版）')
    
    # 共通オプション
    parser.add_argument('--n', type=int, default=256, help='グリッド点の数')
    parser.add_argument('--xrange', type=float, nargs=2, default=[-1.0, 1.0], 
                      help='x軸の範囲 (開始点 終了点)')
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド', required=True)
    
    # テストコマンド
    test_parser = subparsers.add_parser('test', help='テストを実行')
    test_parser.add_argument('--preset', type=str, help='プリセットソルバー設定')
    test_parser.add_argument('--scaling', type=str, default='none', help='スケーリング手法')
    test_parser.add_argument('--reg', type=str, default='none', help='正則化手法')
    test_parser.add_argument('--params', type=str, help='パラメータ (key1=value1,key2=value2 形式)')
    test_parser.add_argument('--no-viz', action='store_true', help='可視化を無効化')
    
    # 診断コマンド
    diag_parser = subparsers.add_parser('diagnostics', help='診断を実行')
    diag_parser.add_argument('--preset', type=str, help='プリセットソルバー設定')
    diag_parser.add_argument('--scaling', type=str, default='none', help='スケーリング手法')
    diag_parser.add_argument('--reg', type=str, default='none', help='正則化手法')
    diag_parser.add_argument('--params', type=str, help='パラメータ (key1=value1,key2=value2 形式)')
    diag_parser.add_argument('--viz', action='store_true', help='可視化を有効化')
    
    # 比較コマンド
    compare_parser = subparsers.add_parser('compare', help='ソルバー間の比較を実行')
    compare_parser.add_argument('--mode', choices=['presets', 'scaling', 'reg', 'custom'], 
                             default='presets', help='比較モード')
    compare_parser.add_argument('--configs', type=str, help='カスタム設定 ("scaling1:reg1,scaling2:reg2" 形式)')
    compare_parser.add_argument('--no-viz', action='store_true', help='可視化を無効化')
    compare_parser.add_argument('--no-save', action='store_true', help='結果の保存を無効化')
    
    # 一覧表示コマンド
    list_parser = subparsers.add_parser('list', help='使用可能な設定を一覧表示')
    list_parser.add_argument('--type', choices=['all', 'scaling', 'reg', 'presets'], 
                          default='all', help='表示する情報の種類')
    
    return parser.parse_args()


def parse_params(param_str: str) -> Dict[str, Any]:
    """
    パラメータ文字列をディクショナリに変換
    
    Args:
        param_str: "key1=value1,key2=value2" 形式のパラメータ文字列
        
    Returns:
        パラメータディクショナリ
    """
    if not param_str:
        return {}
    
    params = {}
    for p in param_str.split(','):
        if not p or '=' not in p:
            continue
        key, value = p.split('=', 1)
        # 数値変換を試みる
        try:
            value = float(value)
            # 整数かどうかをチェック
            if value.is_integer():
                value = int(value)
        except ValueError:
            # 数値変換できない場合はそのまま文字列として使用
            pass
        
        params[key] = value
    
    return params


def get_solver(args, grid_config: GridConfig):
    """コマンドライン引数からソルバーを取得"""
    if args.preset:
        # プリセット設定を使用
        scaling, regularization, preset_params = get_preset_by_name(args.preset)
        # パラメータがあれば上書き
        if hasattr(args, 'params') and args.params:
            user_params = parse_params(args.params)
            preset_params.update(user_params)
    else:
        # カスタム設定を使用
        scaling = args.scaling
        regularization = args.reg
        preset_params = {}
        if hasattr(args, 'params') and args.params:
            preset_params = parse_params(args.params)
    
    # 統合ソルバーを作成
    solver = CCDCompositeSolver.create_solver(
        grid_config,
        scaling=scaling,
        regularization=regularization,
        params=preset_params
    )
    
    return solver


def parse_custom_configs(config_str: str) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """カスタム設定文字列から設定リストを生成"""
    configs = []
    if not config_str:
        return configs
    
    for config_item in config_str.split(','):
        if not config_item or ':' not in config_item:
            continue
        scaling, reg = config_item.split(':', 1)
        name = f"{scaling.capitalize()}_{reg.capitalize()}"
        configs.append((name, scaling, reg, {}))
    
    return configs


def print_available_methods():
    """使用可能な手法を表示"""
    # CCDCompositeSolverの新しい表示メソッドを使用
    CCDCompositeSolver.display_available_methods()


def run_cli():
    """プラグイン対応コマンドラインインターフェースの実行"""
    # プラグインを静かモードでロード
    CCDCompositeSolver.load_plugins(silent=True)
    
    args = parse_args()
    
    # グリッド設定
    n = args.n
    x_range = tuple(args.xrange)
    L = x_range[1] - x_range[0]
    grid_config = GridConfig(n_points=n, h=L / (n - 1))
    
    # コマンドに応じた処理
    if args.command == 'test':
        # ソルバーの生成
        solver = get_solver(args, grid_config)
        # テスターの実行
        tester = CCDMethodTester(CCDCompositeSolver, grid_config, x_range, {})
        tester.solver = solver  # 直接インスタンスを設定
        
        solver_name = args.preset.capitalize() if args.preset else f"{args.scaling.capitalize()}_{args.reg.capitalize()}"
        print(f"テスト実行中: {solver_name}")
        tester.run_tests(prefix=f"{solver_name.lower()}_", visualize=not args.no_viz)
        
    elif args.command == 'diagnostics':
        # ソルバーの生成
        solver = get_solver(args, grid_config)
        # 診断の実行
        diagnostics = CCDSolverDiagnostics(CCDCompositeSolver, grid_config, {})
        diagnostics.solver = solver  # 直接インスタンスを設定
        
        solver_name = args.preset.capitalize() if args.preset else f"{args.scaling.capitalize()}_{args.reg.capitalize()}"
        print(f"診断実行中: {solver_name}")
        diagnostics.perform_full_diagnosis(visualize=args.viz)
        
    elif args.command == 'compare':
        configs = []
        
        if args.mode == 'presets':
            # プリセット設定を比較
            configs = get_combined_presets()
        elif args.mode == 'scaling':
            # 代表的なスケーリング手法を比較
            scaling_methods = CCDCompositeSolver.available_scaling_methods()
            configs = [(s.capitalize(), s, "none", {}) for s in scaling_methods]
        elif args.mode == 'reg':
            # 代表的な正則化手法を比較
            reg_methods = CCDCompositeSolver.available_regularization_methods()
            configs = [(r.capitalize(), "none", r, {}) for r in reg_methods]
        elif args.mode == 'custom' and args.configs:
            # カスタム設定を比較
            configs = parse_custom_configs(args.configs)
        
        if not configs:
            print("エラー: 比較する設定が指定されていません")
            return
        
        # ソルバーリストの生成
        solvers_list = []
        for name, scaling, regularization, params in configs:
            # ソルバーを作成
            solver = CCDCompositeSolver.create_solver(
                grid_config,
                scaling=scaling,
                regularization=regularization,
                params=params
            )
            
            solvers_list.append((name, CCDCompositeSolver, {"solver": solver}))
        
        # 比較を実行
        print(f"比較実行中: {len(configs)}個の設定を比較します")
        comparator = SolverComparator(solvers_list, grid_config, x_range)
        comparator.run_comparison(save_results=not args.no_save, visualize=not args.no_viz)
    
    elif args.command == 'list':
        if args.type == 'all':
            print_available_methods()
        elif args.type == 'scaling':
            print("=== 使用可能なスケーリング手法 ===")
            for method in CCDCompositeSolver.available_scaling_methods():
                param_info = CCDCompositeSolver.get_scaling_param_info(method)
                if param_info:
                    params = ", ".join([f"{k} ({v['help']}, デフォルト: {v['default']})" for k, v in param_info.items()])
                    print(f"- {method} - パラメータ: {params}")
                else:
                    print(f"- {method}")
        elif args.type == 'reg':
            print("=== 使用可能な正則化手法 ===")
            for method in CCDCompositeSolver.available_regularization_methods():
                param_info = CCDCompositeSolver.get_regularization_param_info(method)
                if param_info:
                    params = ", ".join([f"{k} ({v['help']}, デフォルト: {v['default']})" for k, v in param_info.items()])
                    print(f"- {method} - パラメータ: {params}")
                else:
                    print(f"- {method}")
        elif args.type == 'presets':
            print("=== 使用可能なプリセット設定 ===")
            for name, scaling, regularization, params in get_combined_presets():
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "なし"
                print(f"- {name}: スケーリング={scaling}, 正則化={regularization}, パラメータ={param_str}")


if __name__ == "__main__":
    run_cli()