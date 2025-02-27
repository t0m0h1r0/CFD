"""
簡素化されたコマンドラインインターフェースモジュール

CCD法のテスト・診断・比較のためのシンプルなコマンドラインツールを提供します。
0-2階微分の組み合わせ指定をサポートしています。
"""

import argparse

from ccd_core import GridConfig
from unified_solver import CCDCompositeSolver
from presets import get_combined_presets, get_preset_by_name
from ccd_tester import CCDMethodTester
from ccd_diagnostics import CCDSolverDiagnostics
from solver_comparator import SolverComparator


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description='CCD法のテストと診断')
    
    # 共通オプション
    parser.add_argument('--n', type=int, default=256, help='グリッド点の数')
    parser.add_argument('--xrange', type=float, nargs=2, default=[-1.0, 1.0], 
                       help='x軸の範囲 (開始点 終了点)')
    
    # 微分係数の指定方法
    coeff_group = parser.add_mutually_exclusive_group()
    coeff_group.add_argument('--coeffs', type=float, nargs='+', 
                           help='[a, b, c, d] 係数リスト (f = a*psi + b*psi\' + c*psi\'\' + d*psi\'\'\')')
    coeff_group.add_argument('--diff-mode', choices=['psi', 'psi1', 'psi2', 'psi01', 'psi02', 'psi12', 'psi012'], 
                           help='微分の組み合わせ (psi=0階のみ, psi1=1階のみ, psi2=2階のみ, psi01=0階+1階, ...)')
    
    # サブコマンド
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド', required=True)
    
    # テストコマンド
    test_parser = subparsers.add_parser('test', help='テストを実行')
    test_parser.add_argument('--preset', type=str, help='プリセット設定')
    test_parser.add_argument('--scaling', type=str, default='none', help='スケーリング手法')
    test_parser.add_argument('--reg', type=str, default='none', help='正則化手法')
    test_parser.add_argument('--no-viz', action='store_true', help='可視化を無効化')
    
    # 診断コマンド
    diag_parser = subparsers.add_parser('diagnostics', help='診断を実行')
    diag_parser.add_argument('--preset', type=str, help='プリセット設定')
    diag_parser.add_argument('--scaling', type=str, default='none', help='スケーリング手法')
    diag_parser.add_argument('--reg', type=str, default='none', help='正則化手法')
    diag_parser.add_argument('--viz', action='store_true', help='可視化を有効化')
    
    # 比較コマンド
    compare_parser = subparsers.add_parser('compare', help='ソルバー間の比較を実行')
    compare_parser.add_argument('--mode', choices=['presets', 'scaling', 'reg', 'diff-modes'], 
                              default='presets', help='比較モード')
    compare_parser.add_argument('--no-viz', action='store_true', help='可視化を無効化')
    
    # 一覧表示コマンド
    list_parser = subparsers.add_parser('list', help='使用可能な設定を一覧表示')
    list_parser.add_argument('--type', choices=['all', 'scaling', 'reg', 'presets', 'diff-modes'], 
                           default='all', help='表示する情報の種類')
    
    return parser.parse_args()


def get_coefficients_from_diff_mode(diff_mode, coeffs=None):
    """微分モードから係数を取得"""
    if coeffs:
        # 直接指定された係数があればそれを使用（不足分は0で補完）
        coeff_list = list(coeffs)
        while len(coeff_list) < 4:
            coeff_list.append(0.0)
        return coeff_list[:4]
    
    # 微分モードに基づいて係数を設定
    if diff_mode == 'psi':
        return [1.0, 0.0, 0.0, 0.0]
    elif diff_mode == 'psi1':
        return [0.0, 1.0, 0.0, 0.0]
    elif diff_mode == 'psi2':
        return [0.0, 0.0, 1.0, 0.0]
    elif diff_mode == 'psi01':
        return [1.0, 1.0, 0.0, 0.0]
    elif diff_mode == 'psi02':
        return [1.0, 0.0, 1.0, 0.0]
    elif diff_mode == 'psi12':
        return [0.0, 1.0, 1.0, 0.0]
    elif diff_mode == 'psi012':
        return [1.0, 1.0, 1.0, 0.0]
    
    # デフォルトは関数値のみ
    return [1.0, 0.0, 0.0, 0.0]


def get_solver(args, grid_config: GridConfig):
    """コマンドライン引数からソルバーを取得"""
    # プリセットまたはカスタム設定
    if args.preset:
        scaling, regularization, params = get_preset_by_name(args.preset)
    else:
        scaling = args.scaling
        regularization = args.reg
        params = {}
    
    # 係数の設定
    coeffs = get_coefficients_from_diff_mode(args.diff_mode, args.coeffs)
    
    # ソルバーの作成
    return CCDCompositeSolver.create_solver(
        grid_config,
        scaling=scaling,
        regularization=regularization,
        params=params,
        coeffs=coeffs
    )


def get_diff_mode_configs():
    """微分モードの設定リストを生成"""
    return [
        ("PSI", "none", "none", {"coeffs": [1.0, 0.0, 0.0, 0.0]}),
        ("PSI1", "none", "none", {"coeffs": [0.0, 1.0, 0.0, 0.0]}),
        ("PSI2", "none", "none", {"coeffs": [0.0, 0.0, 1.0, 0.0]}),
        ("PSI01", "none", "none", {"coeffs": [1.0, 1.0, 0.0, 0.0]}),
        ("PSI02", "none", "none", {"coeffs": [1.0, 0.0, 1.0, 0.0]}),
        ("PSI12", "none", "none", {"coeffs": [0.0, 1.0, 1.0, 0.0]}),
        ("PSI012", "none", "none", {"coeffs": [1.0, 1.0, 1.0, 0.0]})
    ]


def print_diff_modes():
    """微分モードの説明を表示"""
    print("=== 利用可能な微分モード ===")
    print("- psi:    関数値（0階微分）のみ使用")
    print("- psi1:   1階微分のみ使用")
    print("- psi2:   2階微分のみ使用")
    print("- psi01:  関数値と1階微分の組み合わせ")
    print("- psi02:  関数値と2階微分の組み合わせ")
    print("- psi12:  1階微分と2階微分の組み合わせ")
    print("- psi012: 関数値、1階微分、2階微分の組み合わせ")


def run_cli():
    """コマンドラインインターフェースの実行"""
    # プラグインを読み込み
    CCDCompositeSolver.load_plugins(silent=True)
    
    args = parse_args()
    
    # グリッド設定
    n = args.n
    x_range = tuple(args.xrange)
    L = x_range[1] - x_range[0]
    grid_config = GridConfig(n_points=n, h=L / (n - 1))
    
    # 微分係数の準備
    coeffs = get_coefficients_from_diff_mode(args.diff_mode, args.coeffs)
    
    # コマンドの実行
    if args.command == 'test':
        # ソルバー作成とテスト実行
        solver = get_solver(args, grid_config)
        tester = CCDMethodTester(CCDCompositeSolver, grid_config, x_range, {}, coeffs=coeffs)
        tester.solver = solver
        
        name = args.preset or f"{args.scaling}_{args.reg}"
        print(f"テスト実行中: {name.capitalize()} (coeffs={coeffs})")
        prefix = f"{name.lower()}_{'diff_' + args.diff_mode + '_' if args.diff_mode else ''}"
        tester.run_tests(prefix=prefix, visualize=not args.no_viz)
    
    elif args.command == 'diagnostics':
        # ソルバー作成と診断実行
        solver = get_solver(args, grid_config)
        diagnostics = CCDSolverDiagnostics(CCDCompositeSolver, grid_config, {})
        diagnostics.solver = solver
        
        name = args.preset or f"{args.scaling}_{args.reg}"
        print(f"診断実行中: {name.capitalize()} (coeffs={coeffs})")
        diagnostics.perform_full_diagnosis(visualize=args.viz)
    
    elif args.command == 'compare':
        # 比較モードの設定
        if args.mode == 'presets':
            configs = get_combined_presets()
        elif args.mode == 'scaling':
            scaling_methods = CCDCompositeSolver.available_scaling_methods()
            configs = [(s.capitalize(), s, "none", {}) for s in scaling_methods]
        elif args.mode == 'reg':
            reg_methods = CCDCompositeSolver.available_regularization_methods()
            configs = [(r.capitalize(), "none", r, {}) for r in reg_methods]
        elif args.mode == 'diff-modes':
            configs = get_diff_mode_configs()
        else:
            print("エラー: 無効な比較モードです")
            return
        
        # ソルバーリストの作成
        solvers_list = []
        for name, scaling, regularization, params in configs:
            params_copy = params.copy()
            
            # diff-modesモード以外では指定された係数を使用
            if args.mode != 'diff-modes' and 'coeffs' not in params_copy:
                params_copy['coeffs'] = coeffs
            
            tester = CCDMethodTester(
                CCDCompositeSolver, 
                grid_config, 
                x_range, 
                solver_kwargs={
                    'scaling': scaling,
                    'regularization': regularization,
                    **params_copy
                }
            )
            solvers_list.append((name, tester))
        
        # 比較実行
        print(f"比較実行中: {len(configs)}個の設定を比較します")
        prefix = "diff_mode_" if args.mode == 'diff-modes' else ""
        comparator = SolverComparator(solvers_list, grid_config, x_range)
        comparator.run_comparison(save_results=True, visualize=not args.no_viz, prefix=prefix)
    
    elif args.command == 'list':
        if args.type in ['all', 'scaling', 'reg', 'presets']:
            # プラグイン情報の表示
            CCDCompositeSolver.display_available_methods()
        if args.type in ['all', 'diff-modes']:
            # 微分モードの表示
            print("\n")
            print_diff_modes()


if __name__ == "__main__":
    run_cli()