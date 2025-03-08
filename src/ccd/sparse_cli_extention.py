# sparse_cli_extension.py
"""
スパース行列機能をCLIに追加するための拡張モジュール
"""

import argparse
from typing import Dict, Any


def add_sparse_arguments(parser: argparse.ArgumentParser) -> None:
    """
    スパース行列関連の引数をコマンドラインパーサーに追加
    
    Args:
        parser: 引数パーサー
    """
    sparse_group = parser.add_argument_group('スパース行列オプション')
    
    # ソルバー関連
    sparse_group.add_argument(
        "--solver", 
        type=str, 
        choices=['direct', 'gmres', 'bicgstab', 'cg'], 
        default='direct',
        help="使用するスパースソルバー (デフォルト: direct)"
    )
    
    # 反復法関連のオプション
    sparse_group.add_argument(
        "--solver-tol", 
        type=float, 
        default=1e-10,
        help="反復ソルバーの収束許容誤差 (デフォルト: 1e-10)"
    )
    
    sparse_group.add_argument(
        "--solver-maxiter", 
        type=int, 
        default=1000,
        help="反復ソルバーの最大反復回数 (デフォルト: 1000)"
    )
    
    sparse_group.add_argument(
        "--solver-restart", 
        type=int, 
        default=100,
        help="GMRESのリスタート値 (デフォルト: 100)"
    )
    
    sparse_group.add_argument(
        "--no-preconditioner", 
        action="store_true",
        help="前処理を使用しない"
    )
    
    # 行列分析
    sparse_group.add_argument(
        "--analyze-matrix", 
        action="store_true",
        help="行列の疎性を分析して表示"
    )
    
    sparse_group.add_argument(
        "--export-matrix", 
        type=str, 
        default=None,
        help="行列をファイルにエクスポート (MatrixMarket形式)"
    )


def get_solver_options(args: argparse.Namespace) -> Dict[str, Any]:
    """
    コマンドライン引数からソルバーオプションを取得
    
    Args:
        args: コマンドライン引数の名前空間
        
    Returns:
        Dict[str, Any]: ソルバーオプション辞書
    """
    return {
        "tol": args.solver_tol,
        "maxiter": args.solver_maxiter,
        "restart": args.solver_restart,
        "use_preconditioner": not args.no_preconditioner,
    }


def export_matrix(A, filename: str) -> None:
    """
    行列をMatrixMarket形式でエクスポート
    
    Args:
        A: エクスポートする行列
        filename: 出力ファイル名
    """
    import cupyx.scipy.io as spio
    try:
        spio.mmwrite(filename, A)
        print(f"行列を {filename} にエクスポートしました")
    except Exception as e:
        print(f"行列のエクスポートに失敗しました: {e}")


# cli.pyに以下のコードを追加する例:
"""
from sparse_cli_extension import add_sparse_arguments, get_solver_options, export_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="1D CCD法の実装 (スパース行列最適化版)")
    
    # 既存の引数追加
    ...
    
    # スパース行列関連の引数を追加
    add_sparse_arguments(parser)
    
    return parser.parse_args()

def run_cli():
    args = parse_args()
    
    # グリッドとテスターの設定
    ...
    
    # ソルバーの設定
    solver.set_solver(method=args.solver, options=get_solver_options(args))
    
    # 行列分析
    if args.analyze_matrix:
        solver.analyze_system()
    
    # 行列エクスポート
    if args.export_matrix:
        A, _ = system.build_matrix_system()
        export_matrix(A, args.export_matrix)
"""