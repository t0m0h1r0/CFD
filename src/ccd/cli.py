"""
コマンドラインインターフェースモジュール

CCD法のテスト・診断・比較のためのコマンドラインツールを提供します。
"""

import argparse
import inspect
from typing import Dict, Any, List, Tuple

from ccd_core import GridConfig
from unified_solver import CCDCompositeSolver


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description='CCD法のテストと診断')
    
    # 共通オプション
    parser.add_argument('--n', type=int, default=256, help='グリッド点の数')
    parser.add_argument('--xrange', type=float, nargs=2, default=[-1.0, 1.0], 
                      help='x軸の範囲 (開始点 終了点)')
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # テストコマンド
    test_parser = subparsers.add_parser('test', help='テストを実行')
    test_parser.add_argument('--preset', type=str, default=None,
                          choices=['basic', 'normalization', 'rehu', 'tikhonov', 
                                  'landweber', 'precomputed_landweber', 'iterative'],
                          help='プリセットソルバー設定')
    test_parser.add_argument('--prefix', type=str, default='', help='出力ファイルの接頭辞')
    test_parser.add_argument('--param', nargs='+', help='プリセットのパラメータ ("名前:値" 形式)')
    test_parser.add_argument('--no-viz', action='store_true', help='可視化を無効化')
    test_parser.add_argument('--scaling', type=str, default='none',
                          choices=CCDCompositeSolver.available_scaling_methods(),
                          help='スケーリング手法')
    test_parser.add_argument('--regularization', type=str, default='none',
                          choices=CCDCompositeSolver.available_regularization_methods(),
                          help='正則化手法')
    test_parser.add_argument('--scaling-param', nargs='+', 
                          help='スケーリングパラメータ ("名前:値" 形式)')
    test_parser.add_argument('--reg-param', nargs='+',
                          help='正則化パラメータ ("名前:値" 形式)')
    
    # 診断コマンド
    diag_parser = subparsers.add_parser('diagnostics', help='診断を実行')
    diag_parser.add_argument('--preset', type=str, default=None,
                          choices=['basic', 'normalization', 'rehu', 'tikhonov', 
                                  'landweber', 'precomputed_landweber', 'iterative'],
                          help='プリセットソルバー設定')
    diag_parser.add_argument('--param', nargs='+', help='プリセットのパラメータ ("名前:値" 形式)')
    diag_parser.add_argument('--viz', action='store_true', help='可視化を有効化')
    diag_parser.add_argument('--scaling', type=str, default='none',
                          choices=CCDCompositeSolver.available_scaling_methods(),
                          help='スケーリング手法')
    diag_parser.add_argument('--regularization', type=str, default='none',
                          choices=CCDCompositeSolver.available_regularization_methods(),
                          help='正則化手法')
    diag_parser.add_argument('--scaling-param', nargs='+', 
                          help='スケーリングパラメータ ("名前:値" 形式)')
    diag_parser.add_argument('--reg-param', nargs='+',
                          help='正則化パラメータ ("名前:値" 形式)')
    
    # 比較コマンド
    compare_parser = subparsers.add_parser('compare', help='ソルバー間の比較を実行')
    compare_parser.add_argument('--configs', nargs='+', 
                              help='比較する設定 ("名前:スケーリング:正則化:パラメータ" 形式)')
    compare_parser.add_argument('--presets', action='store_true', help='プリセット設定を比較')
    compare_parser.add_argument('--custom', nargs='+', 
                              help='独自組み合わせ ("スケーリング:正則化" 形式の組み合わせのリスト)')
    compare_parser.add_argument('--no-viz', action='store_true', help='可視化を無効化')
    compare_parser.add_argument('--no-save', action='store_true', help='結果の保存を無効化')
    
    # 一覧表示コマンド
    list_parser = subparsers.add_parser('list', help='使用可能な設定を一覧表示')
    
    return parser.parse_args()


def parse_params(param_list: List[str]) -> Dict[str, Any]:
    """
    パラメータリストをディクショナリに変換
    
    Args:
        param_list: "名前:値" 形式のパラメータリスト
        
    Returns:
        パラメータディクショナリ
    """
    if not param_list:
        return {}
    
    params = {}
    for p in param_list:
        name, value = p.split(':', 1)
        # 数値変換を試みる
        try:
            value = float(value)
            # 整数かどうかをチェック
            if value.is_integer():
                value = int(value)
        except ValueError:
            # 数値変換できない場合はそのまま文字列として使用
            pass
        
        params[name] = value
    
    return params


def parse_config_spec(config_spec: str) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    設定仕様文字列をパース
    
    Args:
        config_spec: "名前:スケーリング:正則化:パラメータ" 形式の文字列
        
    Returns:
        (名前, スケーリング, 正則化, パラメータ辞書)
    """
    parts = config_spec.split(':')
    name = parts[0]
    scaling = parts[1] if len(parts) > 1 else "none"
    regularization = parts[2] if len(parts) > 2 else "none"
    
    # パラメータ部分を解析
    params = {}
    if len(parts) > 3:
        param_str = parts[3]
        for p in param_str.split(','):
            if not p:
                continue
            param_parts = p.split('=')
            if len(param_parts) == 2:
                key, value = param_parts
                # 数値変換を試みる
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
                params[key] = value
    
    return name, scaling, regularization, params


def create_preset_solver(preset: str, grid_config: GridConfig, params: Dict[str, Any] = None) -> CCDCompositeSolver:
    """
    プリセット設定からソルバーを作成
    
    Args:
        preset: プリセット名
        grid_config: グリッド設定
        params: 追加パラメータ
        
    Returns:
        設定されたソルバーインスタンス
    """
    params = params or {}
    
    if preset == "basic":
        return CCDCompositeSolver.create_basic_solver(grid_config)
    elif preset == "normalization":
        return CCDCompositeSolver.create_normalization_solver(grid_config)
    elif preset == "rehu":
        return CCDCompositeSolver.create_rehu_solver(grid_config)
    elif preset == "iterative":
        max_iter = params.get("max_iter", 10)
        tol = params.get("tol", 1e-8)
        return CCDCompositeSolver.create_iterative_solver(grid_config, max_iter, tol)
    elif preset == "tikhonov":
        alpha = params.get("alpha", 1e-6)
        return CCDCompositeSolver.create_tikhonov_solver(grid_config, alpha)
    elif preset == "landweber":
        iterations = params.get("iterations", 20)
        relaxation = params.get("relaxation", 0.1)
        return CCDCompositeSolver.create_landweber_solver(grid_config, iterations, relaxation)
    elif preset == "precomputed_landweber":
        iterations = params.get("iterations", 20)
        relaxation = params.get("relaxation", 0.1)
        return CCDCompositeSolver.create_precomputed_landweber_solver(grid_config, iterations, relaxation)
    else:
        # デフォルトは基本ソルバー
        return CCDCompositeSolver.create_basic_solver(grid_config)


def get_preset_configs() -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    すべてのプリセット設定を取得
    
    Returns:
        [(名前, スケーリング, 正則化, パラメータ辞書), ...] の形式のリスト
    """
    presets = [
        ("Basic", "none", "none", {}),
        ("Normalization", "normalization", "none", {}),
        ("Rehu", "rehu", "none", {}),
        ("Equalization", "equalization", "none", {}),
        ("Iterative", "iterative", "none", {"max_iter": 10, "tol": 1e-8}),
        ("Tikhonov", "none", "tikhonov", {"alpha": 1e-6}),
        ("Tikhonov_Strong", "none", "tikhonov", {"alpha": 1e-4}),
        ("Landweber", "none", "landweber", {"iterations": 20, "relaxation": 0.1}),
        ("PrecomputedLandweber", "none", "precomputed_landweber", {"iterations": 20, "relaxation": 0.1}),
        ("SVD", "none", "svd", {"threshold": 1e-10}),
        ("Rehu_Tikhonov", "rehu", "tikhonov", {"alpha": 1e-6}),
        ("Normalization_Landweber", "normalization", "landweber", {"iterations": 20, "relaxation": 0.1}),
        ("Iterative_SVD", "iterative", "svd", {"max_iter": 10, "tol": 1e-8, "threshold": 1e-10})
    ]
    
    return presets


def get_custom_configs(custom_specs: List[str]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    カスタム設定仕様からソルバー設定を生成
    
    Args:
        custom_specs: "スケーリング:正則化" 形式の設定リスト
        
    Returns:
        [(名前, スケーリング, 正則化, パラメータ辞書), ...] の形式のリスト
    """
    configs = []
    
    for spec in custom_specs:
        parts = spec.split(':')
        scaling = parts[0]
        regularization = parts[1] if len(parts) > 1 else "none"
        
        # デフォルトパラメータ
        params = {}
        if regularization == "tikhonov":
            params = {"alpha": 1e-6}
        elif regularization in ["landweber", "precomputed_landweber"]:
            params = {"iterations": 20, "relaxation": 0.1}
        elif regularization == "svd":
            params = {"threshold": 1e-10}
        
        # 名前を設定
        name = f"{scaling.capitalize()}_{regularization.capitalize()}"
        
        configs.append((name, scaling, regularization, params))
    
    return configs


def run_cli():
    """コマンドラインインターフェースの実行"""
    args = parse_args()
    
    # グリッド設定
    n = args.n
    x_range = tuple(args.xrange)
    L = x_range[1] - x_range[0]
    grid_config = GridConfig(n_points=n, h=L / (n - 1))
    
    # コマンドに応じた処理
    if args.command == 'test':
        from ccd_tester import CCDMethodTester
        
        if args.preset:
            # プリセット設定を使用
            params = parse_params(args.param)
            solver = create_preset_solver(args.preset, grid_config, params)
            solver_name = args.preset.capitalize()
            
            # ソルバーインスタンスを直接使用する場合、grid_configは渡さない
            tester = CCDMethodTester(solver.__class__, grid_config, x_range, {})
            tester.solver = solver  # 直接インスタンスを設定
            tester.run_tests(prefix=f"{args.prefix}{solver_name.lower()}_", visualize=not args.no_viz)
        # カスタム設定を使用する部分の修正
        else:
            # カスタム設定を使用
            scaling_params = parse_params(args.scaling_param)
            reg_params = parse_params(args.reg_param)
            
            # ソルバーの設定
            solver_name = f"{args.scaling.capitalize()}_{args.regularization.capitalize()}"
            print(f"ソルバー設定: スケーリング={args.scaling}, 正則化={args.regularization}")
            
            solver = CCDCompositeSolver(
                grid_config,
                scaling=args.scaling,
                regularization=args.regularization,
                scaling_params=scaling_params,
                regularization_params=reg_params
            )
            
            # 同様に空の辞書を渡し、直接インスタンスを設定
            tester = CCDMethodTester(solver.__class__, grid_config, x_range, {})
            tester.solver = solver  # 直接インスタンスを設定
            tester.run_tests(prefix=f"{args.prefix}{solver_name.lower()}_", visualize=not args.no_viz)
        
    elif args.command == 'diagnostics':
        from ccd_diagnostics import CCDSolverDiagnostics
        
        if args.preset:
            # プリセット設定を使用
            params = parse_params(args.param)
            solver = create_preset_solver(args.preset, grid_config, params)
            solver_name = args.preset.capitalize()
            
            # ソルバークラスとインスタンスを直接渡す
            diagnostics = CCDSolverDiagnostics(solver.__class__, grid_config, 
                                              {"grid_config": grid_config})
            diagnostics.solver = solver  # 直接インスタンスを設定
            diagnostics.perform_full_diagnosis(visualize=args.viz)
        else:
            # カスタム設定を使用
            scaling_params = parse_params(args.scaling_param)
            reg_params = parse_params(args.reg_param)
            
            # ソルバーの設定
            solver_name = f"{args.scaling.capitalize()}_{args.regularization.capitalize()}"
            print(f"ソルバー設定: スケーリング={args.scaling}, 正則化={args.regularization}")
            
            solver = CCDCompositeSolver(
                grid_config,
                scaling=args.scaling,
                regularization=args.regularization,
                scaling_params=scaling_params,
                regularization_params=reg_params
            )
            
            # ソルバークラスとインスタンスを直接渡す
            diagnostics = CCDSolverDiagnostics(solver.__class__, grid_config, 
                                              {"grid_config": grid_config})
            diagnostics.solver = solver  # 直接インスタンスを設定
            diagnostics.perform_full_diagnosis(visualize=args.viz)
        
    elif args.command == 'compare':
        from solver_comparator import SolverComparator
        
        if args.presets:
            # プリセット設定を比較
            configs = get_preset_configs()
        elif args.custom:
            # カスタム設定を比較
            configs = get_custom_configs(args.custom)
        elif args.configs:
            # 指定された設定を比較
            configs = [parse_config_spec(spec) for spec in args.configs]
        else:
            print("エラー: --presets, --custom, または --configs のいずれかを指定してください")
            return
        
        # ソルバーリストを生成
        solvers_list = []
        for name, scaling, regularization, params in configs:
            # スケーリングと正則化のパラメータを分離
            scaling_params = {}
            reg_params = {}
            for key, value in params.items():
                if key in ["alpha", "threshold", "iterations", "relaxation"]:
                    reg_params[key] = value
                else:
                    scaling_params[key] = value
            
            # ソルバーを作成
            solver = CCDCompositeSolver(
                grid_config,
                scaling=scaling,
                regularization=regularization,
                scaling_params=scaling_params,
                regularization_params=reg_params
            )
            
            solvers_list.append((name, solver.__class__, {"grid_config": grid_config, "solver": solver}))
        
        # 比較を実行
        comparator = SolverComparator(solvers_list, grid_config, x_range)
        comparator.run_comparison(save_results=not args.no_save, visualize=not args.no_viz)
    
    elif args.command == 'list':
        # 利用可能な設定を一覧表示
        print("=== CCDCompositeSolver - 使用可能な設定 ===")
        
        print("\n=== スケーリング手法 ===")
        for i, method in enumerate(CCDCompositeSolver.available_scaling_methods(), 1):
            print(f"{i}. {method}")
        
        print("\n=== 正則化手法 ===")
        for i, method in enumerate(CCDCompositeSolver.available_regularization_methods(), 1):
            print(f"{i}. {method}")
        
        print("\n=== プリセット設定 ===")
        for name, scaling, regularization, params in get_preset_configs():
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "なし"
            print(f"- {name}: スケーリング={scaling}, 正則化={regularization}, パラメータ={param_str}")
        
        print("\n=== 使用例 ===")
        print("# プリセット設定を使用")
        print("python main.py test --preset tikhonov --param alpha:1e-5")
        print("python main.py test --preset rehu")
        print("\n# カスタム設定を使用")
        print("python main.py test --scaling rehu --regularization tikhonov --reg-param alpha:1e-6")
        print("\n# 複数設定の比較")
        print("python main.py compare --presets")
        print("python main.py compare --custom \"rehu:tikhonov\" \"normalization:landweber\"")
    
    else:
        print("有効なコマンドを指定してください: test, diagnostics, compare, または list")


if __name__ == "__main__":
    run_cli()