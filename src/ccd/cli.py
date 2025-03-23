#!/usr/bin/env python3
"""
高精度コンパクト差分法 (CCD) コマンドラインインターフェース

このスクリプトは、コンパクト差分法(CCD)を使用した数値計算のためのCLIツールです。
1次元、2次元、3次元のCCD計算を簡便に実行できるようにします。
"""

import os
import sys
import time
import argparse
import numpy as np
import importlib
import inspect
from typing import List, Dict, Any, Optional, Tuple

# 相対インポートのためのパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CCD関連モジュールのインポート
from core.grid.grid1d import Grid1D
from core.grid.grid2d import Grid2D
from core.grid.grid3d import Grid3D
from tester.tester1d import CCDTester1D
from tester.tester2d import CCDTester2D
from tester.tester3d import CCDTester3D
from equation_set.equation_sets import EquationSet
from linear_solver import create_solver
from test_function.test_function1d import TestFunction1DFactory
from test_function.test_function2d import TestFunction2DFactory
from test_function.test_function3d import TestFunction3DFactory

# スケーリングプラグインのインポート（存在する場合）
try:
    from scaling import plugin_manager as scaling_plugins
    HAS_SCALING = True
except ImportError:
    HAS_SCALING = False
    scaling_plugins = None
    print("警告: スケーリングプラグインがインポートできませんでした。")

def list_available_solvers(dim: int = None) -> List[str]:
    """
    利用可能なソルバーの一覧を取得
    
    Args:
        dim: 次元 (1, 2, 3)
        
    Returns:
        利用可能なソルバー名のリスト
    """
    # バックエンド定義 (名前、モジュール、クラス、依存ライブラリ)
    backends = [
        ("CPU", "cpu_solver", "CPULinearSolver", None),
        ("GPU", "gpu_solver", "GPULinearSolver", "cupy"),
        ("JAX", "jax_solver", "JAXLinearSolver", "jax"),
        ("最適化GPU", "opt_solver", "OptimizedGPULinearSolver", "cupy")
    ]
    
    # 一般的なソルバーメソッド (フォールバック用)
    fallbacks = {
        "CPULinearSolver": ["direct", "gmres", "lgmres", "cg", "cgs", "bicg", "bicgstab", "minres", "lsqr"],
        "GPULinearSolver": ["direct", "gmres", "cg", "cgs", "minres", "lsqr", "lsmr"],
        "JAXLinearSolver": ["direct", "gmres", "cg", "bicgstab"],
        "OptimizedGPULinearSolver": ["direct", "gmres", "cg", "cgs", "minres", "bicgstab"]
    }
    
    all_solvers = []
    
    for name, mod, cls, dep in backends:
        try:
            # 依存ライブラリのチェックと必要なモジュールのインポート
            if dep and not importlib.util.find_spec(dep):
                continue
                
            # モジュールとクラスをインポート
            module = importlib.import_module(f"linear_solver.{mod}")
            solver_class = getattr(module, cls)
            
            # ソルバーメソッドを取得
            methods = [m.replace('_solve_', '') for m, _ in inspect.getmembers(solver_class, 
                      predicate=lambda x: inspect.isfunction(x) and x.__name__.startswith('_solve_'))]
            
            # 結果を追加
            if methods:
                all_solvers.extend(methods)
            else:
                all_solvers.extend(fallbacks.get(cls, []))
        except Exception:
            continue
    
    # 重複を削除して整理
    return sorted(list(set(all_solvers)))

def list_available_solvers_print(dim: int = None) -> None:
    """利用可能なソルバーの一覧を表示"""
    print("=== 利用可能なソルバー ===")
    
    # バックエンド定義 (名前、モジュール、クラス、依存ライブラリ)
    backends = [
        ("CPU", "cpu_solver", "CPULinearSolver", None),
        ("GPU", "gpu_solver", "GPULinearSolver", "cupy"),
        ("JAX", "jax_solver", "JAXLinearSolver", "jax"),
        ("最適化GPU", "opt_solver", "OptimizedGPULinearSolver", "cupy")
    ]
    
    # 一般的なソルバーメソッド (フォールバック用)
    fallbacks = {
        "CPULinearSolver": ["direct", "gmres", "lgmres", "cg", "cgs", "bicg", "bicgstab", "minres", "lsqr"],
        "GPULinearSolver": ["direct", "gmres", "cg", "cgs", "minres", "lsqr", "lsmr"],
        "JAXLinearSolver": ["direct", "gmres", "cg", "bicgstab"],
        "OptimizedGPULinearSolver": ["direct", "gmres", "cg", "cgs", "minres", "bicgstab"]
    }
    
    for name, mod, cls, dep in backends:
        print(f"{name} バックエンド:")
        try:
            # 依存ライブラリのチェックと必要なモジュールのインポート
            if dep and not importlib.util.find_spec(dep):
                print(f"  {dep} が利用できません")
                continue
                
            # モジュールとクラスをインポート
            module = importlib.import_module(f"linear_solver.{mod}")
            solver_class = getattr(module, cls)
            
            # ソルバーメソッドを取得
            methods = [m.replace('_solve_', '') for m, _ in inspect.getmembers(solver_class, 
                      predicate=lambda x: inspect.isfunction(x) and x.__name__.startswith('_solve_'))]
            
            # 結果表示 (メソッドが見つからない場合はフォールバック)
            print("  " + ", ".join(sorted(methods) if methods else fallbacks.get(cls, [])))
        except Exception:
            print(f"  エラー: {cls}の情報取得に失敗")

def get_available_scaling_methods() -> List[str]:
    """
    利用可能なスケーリングメソッドの一覧を取得
    
    Returns:
        利用可能なスケーリングメソッド名のリスト
    """
    if not HAS_SCALING:
        return []
    
    methods = []
    
    try:
        # 1. よく使われるスケーリングメソッドをチェック
        for m in ["NoScaling", "RowScaling", "ColumnScaling", "DiagonalScaling", "SymmetricScaling"]:
            try:
                if scaling_plugins.get_plugin(m): methods.append(m)
            except: pass
            
        # 2. ファイルシステムから追加のメソッドを検出
        try:
            for f in os.listdir(os.path.dirname(scaling_plugins.__file__)):
                if f.endswith('.py') and f not in ['__init__.py', 'plugin_manager.py', 'base.py']:
                    m = f[:-3]  # .pyを削除
                    if m not in methods:
                        try:
                            if scaling_plugins.get_plugin(m): methods.append(m)
                        except: pass
        except: pass
        
        # 結果を整理
        return sorted(methods)
    except Exception as e:
        print(f"スケーリングメソッド取得エラー: {e}")
        return []

def list_available_scaling_methods() -> None:
    """利用可能なスケーリングメソッドの一覧を表示"""
    print("=== 利用可能なスケーリングメソッド ===")
    
    if not HAS_SCALING:
        print("スケーリングプラグインが利用できません。")
        return
    
    methods = get_available_scaling_methods()
    
    # 結果表示
    if methods:
        for m in methods: print(f"- {m}")
    else:
        print("利用可能なスケーリングメソッドが見つかりません。")

def get_available_equation_sets(dim: int) -> List[str]:
    """
    指定された次元で利用可能な方程式セットの一覧を取得
    
    Args:
        dim: 次元 (1, 2, 3)
        
    Returns:
        利用可能な方程式セット名のリスト
    """
    equation_sets = EquationSet.get_available_sets(dim)
    if isinstance(equation_sets, dict):
        return sorted(list(equation_sets.keys()))
    return []

def list_available_equation_sets() -> None:
    """利用可能な方程式セットの一覧を表示"""
    print("=== 利用可能な方程式セット ===")
    
    for dim in [1, 2, 3]:
        print(f"\n{dim}D 方程式セット:")
        equation_sets = get_available_equation_sets(dim)
        if equation_sets:
            for name in equation_sets:
                print(f"- {name}")
        else:
            print("方程式セットが見つかりません。")

def get_available_test_functions(dim: int) -> List[str]:
    """
    指定された次元で利用可能なテスト関数の一覧を取得
    
    Args:
        dim: 次元 (1, 2, 3)
        
    Returns:
        利用可能なテスト関数名のリスト
    """
    if dim == 1:
        funcs = TestFunction1DFactory.create_standard_functions()
    elif dim == 2:
        funcs = TestFunction2DFactory.create_standard_functions()
    else:  # dim == 3
        funcs = TestFunction3DFactory.create_standard_functions()
    
    return sorted([func.name for func in funcs])

def list_available_test_functions() -> None:
    """利用可能なテスト関数の一覧を表示"""
    print("=== 利用可能なテスト関数 ===")
    
    # 1D テスト関数
    print("\n1D テスト関数:")
    funcs_1d = TestFunction1DFactory.create_standard_functions()
    for func in funcs_1d:
        print(f"- {func.name}")
    
    # 2D テスト関数
    print("\n2D テスト関数:")
    funcs_2d = TestFunction2DFactory.create_standard_functions()
    for func in funcs_2d:
        print(f"- {func.name}")
    
    # 3D テスト関数
    print("\n3D テスト関数:")
    funcs_3d = TestFunction3DFactory.create_standard_functions()
    for func in funcs_3d:
        print(f"- {func.name}")

def create_grid(dim: int, nx: int, ny: int = None, nz: int = None, 
                x_range: Tuple[float, float] = (-1.0, 1.0),
                y_range: Tuple[float, float] = None, 
                z_range: Tuple[float, float] = None) -> Any:
    """
    適切な次元のグリッドを作成
    
    Args:
        dim: 次元 (1, 2, 3)
        nx: x方向の格子点数
        ny: y方向の格子点数 (2D/3Dのみ)
        nz: z方向の格子点数 (3Dのみ)
        x_range: x方向の範囲
        y_range: y方向の範囲 (2D/3Dのみ)
        z_range: z方向の範囲 (3Dのみ)
    
    Returns:
        Grid1D, Grid2D, or Grid3D: 指定された次元のグリッド
    """
    if dim == 1:
        return Grid1D(nx, x_range=x_range)
    elif dim == 2:
        y_range = y_range or x_range
        return Grid2D(nx, ny or nx, x_range=x_range, y_range=y_range)
    elif dim == 3:
        y_range = y_range or x_range
        z_range = z_range or x_range
        return Grid3D(nx, ny or nx, nz or nx, x_range=x_range, y_range=y_range, z_range=z_range)
    else:
        raise ValueError(f"不正な次元: {dim}")

def create_tester(dim: int, grid: Any) -> Any:
    """
    適切な次元のテスターを作成
    
    Args:
        dim: 次元 (1, 2, 3)
        grid: 対応する次元のグリッド
    
    Returns:
        CCDTester1D, CCDTester2D, or CCDTester3D: 指定された次元のテスター
    """
    if dim == 1:
        return CCDTester1D(grid)
    elif dim == 2:
        return CCDTester2D(grid)
    elif dim == 3:
        return CCDTester3D(grid)
    else:
        raise ValueError(f"不正な次元: {dim}")

def format_errors(errors: List[float], dim: int) -> str:
    """
    エラー結果を整形された文字列に変換
    
    Args:
        errors: エラーのリスト
        dim: 次元
    
    Returns:
        str: 整形された文字列
    """
    if dim == 1:
        labels = ["ψ", "ψ'", "ψ''", "ψ'''"]
    elif dim == 2:
        labels = ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
    else:  # dim == 3
        labels = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
    
    result = []
    for label, error in zip(labels, errors):
        result.append(f"{label}: {error:.6e}")
    
    return ", ".join(result)

def run_single_test(tester, test_func, args, verbose=False):
    """
    単一のテスト実行
    
    Args:
        tester: テスターオブジェクト
        test_func: テスト関数
        args: コマンドライン引数
        verbose: 詳細表示フラグ
        
    Returns:
        dict: テスト結果
    """
    # 時間計測開始
    start_time = time.time()
    
    # テスト実行
    result = tester.run_test(test_func)
    
    # 経過時間
    elapsed = time.time() - start_time
    
    # 反復回数を取得
    iterations = None
    if hasattr(tester, 'solver') and hasattr(tester.solver, 'last_iterations'):
        iterations = tester.solver.last_iterations
    
    # 結果を拡張
    result['elapsed_time'] = elapsed
    result['iterations'] = iterations
    
    # 詳細表示
    if verbose:
        print(f"  関数: {result['function']}")
        print(f"  計算時間: {elapsed:.4f} 秒")
        print(f"  エラー: {format_errors(result['errors'], args.dim)}")
        if iterations is not None:
            print(f"  反復回数: {iterations}")
    
    return result

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="CCD (高精度コンパクト差分法) CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本パラメータ
    parser.add_argument("--dim", type=int, choices=[1, 2, 3], default=1,
                        help="次元 (1, 2, または 3)")
    parser.add_argument("--nx", type=int, default=16,
                        help="x方向の格子点数")
    parser.add_argument("--ny", type=int, default=None,
                        help="y方向の格子点数 (2D/3Dのみ、指定しない場合はnxと同じ)")
    parser.add_argument("--nz", type=int, default=None,
                        help="z方向の格子点数 (3Dのみ、指定しない場合はnxと同じ)")
    parser.add_argument("--xrange", type=float, nargs=2, default=(-1.0, 1.0),
                        metavar=('XMIN', 'XMAX'),
                        help="x方向の範囲")
    parser.add_argument("--yrange", type=float, nargs=2, default=None,
                        metavar=('YMIN', 'YMAX'),
                        help="y方向の範囲 (指定しない場合はx方向と同じ)")
    parser.add_argument("--zrange", type=float, nargs=2, default=None,
                        metavar=('ZMIN', 'ZMAX'),
                        help="z方向の範囲 (指定しない場合はx方向と同じ)")
    
    # 方程式セットと関数
    parser.add_argument("-e", "--equation", type=str, default="poisson",
                        help="使用する方程式セット名 ('all'で全て)")
    parser.add_argument("-f", "--function", type=str, default="Sine",
                        help="使用するテスト関数名 ('all'で全て)")
    
    # 演算器とソルバー設定
    parser.add_argument("--backend", type=str, choices=["cpu", "cuda", "jax", "opt"], 
                        default="cpu", help="計算バックエンド")
    parser.add_argument("--solver", type=str, default="direct",
                        help="使用するソルバー (例: direct, gmres, cg, 'all'で全て)")
    parser.add_argument("--scaling", type=str, default=None,
                        help="使用するスケーリング手法 ('all'で全て)")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="反復法の収束許容値")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="反復法の最大反復回数")
    parser.add_argument("--restart", type=int, default=None,
                        help="GMRESの再起動パラメータ")
    parser.add_argument("--perturbation", type=float, default=None,
                        help="初期値の摂動レベル（0.0～1.0）、Noneは初期値なし")
    
    # 出力オプション
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="詳細な出力を表示")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="結果の出力先ファイル")
    parser.add_argument("--visualize", action="store_true",
                        help="行列構造も可視化")
    parser.add_argument("--no-vis", action="store_true",
                        help="解の可視化を無効化")
    parser.add_argument("--max-combinations", type=int, default=100,
                        help="'all'使用時の最大組み合わせ数")
    
    # リスト表示オプション
    parser.add_argument("--list", action="store_true",
                        help="利用可能な方程式セットを一覧表示")
    parser.add_argument("--list-solvers", action="store_true",
                        help="利用可能なソルバーを一覧表示")
    parser.add_argument("--list-scaling", action="store_true",
                        help="利用可能なスケーリング手法を一覧表示")
    parser.add_argument("--list-functions", action="store_true",
                        help="利用可能なテスト関数を一覧表示")
    
    args = parser.parse_args()
    
    # リスト表示オプションを最初にチェック
    if args.list:
        list_available_equation_sets()
        return
        
    if args.list_solvers:
        list_available_solvers_print(args.dim)
        return
        
    if args.list_scaling:
        list_available_scaling_methods()
        return
        
    if args.list_functions:
        list_available_test_functions()
        return
    
    # 'all'オプションの処理
    function_list = [args.function]
    if args.function.lower() == 'all':
        function_list = get_available_test_functions(args.dim)
    
    equation_list = [args.equation]
    if args.equation.lower() == 'all':
        equation_list = get_available_equation_sets(args.dim)
    
    solver_list = [args.solver]
    if args.solver.lower() == 'all':
        solver_list = list_available_solvers(args.dim)
    
    scaling_list = [args.scaling]
    if args.scaling and args.scaling.lower() == 'all':
        scaling_list = get_available_scaling_methods()
        # NoScalingを必ず含める
        if 'NoScaling' not in scaling_list:
            scaling_list.append('NoScaling')
    
    # 組み合わせ数をチェック
    total_combinations = len(function_list) * len(equation_list) * len(solver_list) * len(scaling_list)
    
    if total_combinations > 1:
        print(f"実行する組み合わせ数: {total_combinations}")
        if total_combinations > args.max_combinations:
            print(f"警告: 組み合わせ数が多すぎます（最大: {args.max_combinations}）")
            print("--max-combinations オプションで最大数を増やすか、パラメータを絞ってください")
            return 1
    
    # グリッドを作成
    try:
        grid = create_grid(
            args.dim, args.nx, args.ny, args.nz, 
            args.xrange, args.yrange, args.zrange
        )
    except Exception as e:
        print(f"グリッド作成エラー: {e}")
        return 1
    
    # ソルバーオプションのベース設定
    solver_options = {
        "backend": args.backend,
        "tol": args.tol,
        "maxiter": args.maxiter
    }
    
    if args.restart is not None:
        solver_options["restart"] = args.restart
    
    # 全組み合わせの結果を保存
    all_results = []
    
    # 組み合わせごとに実行
    for equation_name in equation_list:
        for function_name in function_list:
            for solver_name in solver_list:
                for scaling_name in scaling_list:
                    if scaling_name is None and args.scaling is None:
                        # None の場合は1回だけ実行
                        scaling_to_use = None
                    else:
                        scaling_to_use = scaling_name
                    
                    print(f"\n実行: {function_name}, {equation_name}, {solver_name}, {scaling_to_use or 'スケーリングなし'}")
                    
                    try:
                        # テスターを作成
                        tester = create_tester(args.dim, grid)
                        
                        # 方程式セットを設定
                        tester.set_equation_set(equation_name)
                        
                        # ソルバーを設定
                        tester.set_solver_options(solver_name, solver_options)
                        
                        # スケーリング手法を設定
                        tester.scaling_method = scaling_to_use
                        
                        # 初期値摂動を設定
                        if args.perturbation is not None:
                            tester.perturbation_level = args.perturbation
                        
                        # テスト関数を取得
                        test_func = tester.get_test_function(function_name)
                        
                        # テスト実行
                        result = run_single_test(tester, test_func, args, args.verbose)
                        
                        # 結果をメタデータで拡張
                        result['equation'] = equation_name
                        result['solver'] = solver_name
                        result['scaling'] = scaling_to_use
                        
                        # 結果を保存
                        all_results.append(result)
                        
                        # 解の可視化（--no-visで無効化された場合を除く）
                        if not args.no_vis:
                            try:
                                # 各次元に応じた可視化処理
                                if args.dim == 1:
                                    from visualizer.visualizer1d import CCDVisualizer1D
                                    visualizer = CCDVisualizer1D()
                                    vis_file = visualizer.visualize_derivatives(
                                        grid, result['function'], result['numerical'], 
                                        result['exact'], result['errors'],
                                        prefix=f"{function_name}_{equation_name}_{solver_name}_{scaling_to_use or 'noscale'}"
                                    )
                                elif args.dim == 2:
                                    from visualizer.visualizer2d import CCDVisualizer2D
                                    visualizer = CCDVisualizer2D()
                                    vis_file = visualizer.visualize_solution(
                                        grid, result['function'], result['numerical'], 
                                        result['exact'], result['errors'],
                                        prefix=f"{function_name}_{equation_name}_{solver_name}_{scaling_to_use or 'noscale'}"
                                    )
                                elif args.dim == 3:
                                    from visualizer.visualizer3d import CCDVisualizer3D
                                    visualizer = CCDVisualizer3D()
                                    vis_file = visualizer.visualize_solution(
                                        grid, result['function'], result['numerical'], 
                                        result['exact'], result['errors'],
                                        prefix=f"{function_name}_{equation_name}_{solver_name}_{scaling_to_use or 'noscale'}"
                                    )
                                
                                if vis_file:
                                    print(f"  解の可視化を保存しました: {vis_file}")
                            except Exception as e:
                                print(f"  解の可視化エラー: {e}")
                        
                        # 行列システム可視化（--visualizeが指定された場合のみ）
                        if args.visualize:
                            try:
                                # 元のメソッドシグネチャに合わせる
                                vis_output = tester.visualize_matrix_system(test_func)
                                print(f"  行列システムの可視化: {vis_output}")
                            except Exception as e:
                                print(f"  行列可視化エラー: {e}")
                    
                    except Exception as e:
                        print(f"エラー: {e}")
                        if args.verbose:
                            import traceback
                            traceback.print_exc()
    
    # 結果サマリーを表示
    if len(all_results) > 1:
        print("\n=== 結果サマリー ===")
        print(f"実行した組み合わせ数: {len(all_results)}")
        
        # 最良の結果
        best_result = min(all_results, key=lambda x: x['errors'][0])
        print(f"\n最良の結果（ψ誤差で比較）:")
        print(f"  関数: {best_result['function']}")
        print(f"  方程式: {best_result['equation']}")
        print(f"  ソルバー: {best_result['solver']}")
        print(f"  スケーリング: {best_result['scaling'] or 'なし'}")
        print(f"  エラー: {format_errors(best_result['errors'], args.dim)}")
        print(f"  計算時間: {best_result['elapsed_time']:.4f}秒")
        if best_result['iterations'] is not None:
            print(f"  反復回数: {best_result['iterations']}")
        
        # 計算時間の統計
        avg_time = sum(r['elapsed_time'] for r in all_results) / len(all_results)
        min_time = min(all_results, key=lambda x: x['elapsed_time'])
        max_time = max(all_results, key=lambda x: x['elapsed_time'])
        
        print(f"\n計算時間統計:")
        print(f"  平均時間: {avg_time:.4f}秒")
        print(f"  最短時間: {min_time['elapsed_time']:.4f}秒 ({min_time['function']}, {min_time['equation']}, {min_time['solver']}, {min_time['scaling'] or 'なし'})")
        print(f"  最長時間: {max_time['elapsed_time']:.4f}秒 ({max_time['function']}, {max_time['equation']}, {max_time['solver']}, {max_time['scaling'] or 'なし'})")
    
    # 出力ファイルに保存（リクエストされた場合）
    if args.output:
        try:
            import pickle
            with open(args.output, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"\n結果を保存しました: {args.output}")
        except Exception as e:
            print(f"\n結果保存エラー: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())