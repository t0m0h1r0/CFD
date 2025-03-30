#!/usr/bin/env python3
"""CCD(高精度コンパクト差分法)CLI - 1D/2D/3D数値計算を簡便に実行"""

import os
import sys
import time
import argparse
import importlib
import inspect

# 相対インポートのためのパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_options(option_type, dim=None, print_mode=False):
    """ソルバー/方程式セット/テスト関数の一覧を取得または表示"""
    if option_type == "solvers":
        backends = [
            ("CPU", "cpu_solver", "CPULinearSolver", None),
            ("GPU", "gpu_solver", "GPULinearSolver", "cupy"),
            ("JAX", "jax_solver", "JAXLinearSolver", "jax"),
            ("最適化GPU", "opt_solver", "OptimizedGPULinearSolver", "cupy")
        ]
        fallbacks = {
            "CPULinearSolver": ["direct", "gmres", "lgmres", "cg", "cgs", "bicg", "bicgstab", "minres", "lsqr"],
            "GPULinearSolver": ["direct", "gmres", "cg", "cgs", "minres", "lsqr", "lsmr"],
            "JAXLinearSolver": ["direct", "gmres", "cg", "bicgstab"],
            "OptimizedGPULinearSolver": ["direct", "gmres", "cg", "cgs", "minres", "bicgstab"]
        }
        
        all_solvers = []
        if print_mode: print("=== 利用可能なソルバー ===")
        
        for name, mod, cls, dep in backends:
            try:
                if dep and not importlib.util.find_spec(dep):
                    if print_mode: print(f"  {dep} が利用できません")
                    continue
                
                module = importlib.import_module(f"linear_solver.{mod}")
                solver_class = getattr(module, cls)
                methods = [m.replace('_solve_', '') for m, _ in inspect.getmembers(solver_class, 
                          predicate=lambda x: inspect.isfunction(x) and x.__name__.startswith('_solve_'))]
                
                if print_mode:
                    print(f"{name} バックエンド:")
                    print("  " + ", ".join(sorted(methods if methods else fallbacks.get(cls, []))))
                else:
                    all_solvers.extend(methods if methods else fallbacks.get(cls, []))
            except Exception:
                if print_mode: print(f"  エラー: {cls}の情報取得に失敗")
        
        return sorted(list(set(all_solvers))) if not print_mode else None
    
    elif option_type == "equation_sets":
        from equation_set.equation_sets import EquationSet
        equation_sets = EquationSet.get_available_sets(dim)
        
        if print_mode:
            print("=== 利用可能な方程式セット ===")
            for d in [1, 2, 3] if dim is None else [dim]:
                print(f"\n{d}D 方程式セット:")
                sets = EquationSet.get_available_sets(d)
                if isinstance(sets, dict):
                    for name in sorted(sets.keys()): print(f"- {name}")
                else:
                    print("方程式セットが見つかりません。")
            return None
        else:
            return sorted(list(equation_sets.keys())) if isinstance(equation_sets, dict) else []
    
    elif option_type == "test_functions":
        if print_mode:
            print("=== 利用可能なテスト関数 ===")
            for d in [1, 2, 3]:
                print(f"\n{d}D テスト関数:")
                factory_class = {
                    1: "test_function.test_function1d.TestFunction1DFactory",
                    2: "test_function.test_function2d.TestFunction2DFactory",
                    3: "test_function.test_function3d.TestFunction3DFactory"
                }[d]
                module_path, class_name = factory_class.rsplit('.', 1)
                factory = getattr(importlib.import_module(module_path), class_name)
                for func in factory.create_standard_functions(): print(f"- {func.name}")
            return None
        else:
            factory_class = {
                1: "test_function.test_function1d.TestFunction1DFactory",
                2: "test_function.test_function2d.TestFunction2DFactory",
                3: "test_function.test_function3d.TestFunction3DFactory"
            }[dim]
            module_path, class_name = factory_class.rsplit('.', 1)
            factory = getattr(importlib.import_module(module_path), class_name)
            return sorted([func.name for func in factory.create_standard_functions()])

def create_grid(dim, nx, ny=None, nz=None, x_range=(-1.0, 1.0), y_range=None, z_range=None):
    """適切な次元のグリッドを作成"""
    grid_classes = {
        1: ("core.grid.grid1d", "Grid1D"),
        2: ("core.grid.grid2d", "Grid2D"),
        3: ("core.grid.grid3d", "Grid3D")
    }
    module_path, class_name = grid_classes[dim]
    grid_class = getattr(importlib.import_module(module_path), class_name)
    
    if dim == 1:
        return grid_class(nx, x_range=x_range)
    elif dim == 2:
        return grid_class(nx, ny or nx, x_range=x_range, y_range=y_range or x_range)
    else:  # dim == 3
        return grid_class(nx, ny or nx, nz or nx, x_range=x_range, 
                         y_range=y_range or x_range, z_range=z_range or x_range)

def create_tester(dim, grid):
    """適切な次元のテスターを作成"""
    tester_classes = {
        1: ("tester.tester1d", "CCDTester1D"),
        2: ("tester.tester2d", "CCDTester2D"),
        3: ("tester.tester3d", "CCDTester3D")
    }
    module_path, class_name = tester_classes[dim]
    tester_class = getattr(importlib.import_module(module_path), class_name)
    return tester_class(grid)

def format_errors(errors, dim):
    """エラー結果を整形"""
    labels = {
        1: ["ψ", "ψ'", "ψ''", "ψ'''"],
        2: ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"],
        3: ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
    }[dim]
    return ", ".join(f"{label}: {error:.6e}" for label, error in zip(labels, errors))

def run_single_test(tester, test_func, args, verbose=False):
    """単一のテスト実行"""
    start_time = time.time()
    result = tester.run_test(test_func)
    elapsed = time.time() - start_time
    
    # 反復回数の取得と結果拡張
    iterations = getattr(tester.solver, 'last_iterations', None) if hasattr(tester, 'solver') else None
    result.update({'elapsed_time': elapsed, 'iterations': iterations})
    
    if verbose:
        print(f"  関数: {result['function']}")
        print(f"  計算時間: {elapsed:.4f} 秒")
        print(f"  エラー: {format_errors(result['errors'], args.dim)}")
        if iterations is not None: print(f"  反復回数: {iterations}")
    
    return result

def visualize_result(dim, grid, result, function_name, equation_name, solver_name):
    """テスト結果を可視化"""
    try:
        prefix = f"{function_name}_{equation_name}_{solver_name}"
        
        visualizer_classes = {
            1: ("visualizer.visualizer1d", "CCDVisualizer1D", "visualize_derivatives"),
            2: ("visualizer.visualizer2d", "CCDVisualizer2D", "visualize_solution"),
            3: ("visualizer.visualizer3d", "CCDVisualizer3D", "visualize_solution")
        }
        
        module_path, class_name, method_name = visualizer_classes[dim]
        visualizer_class = getattr(importlib.import_module(module_path), class_name)
        visualizer = visualizer_class(output_dir="results")
        
        visualize_method = getattr(visualizer, method_name)
        vis_file = visualize_method(
            grid, result['function'], result['numerical'], 
            result['exact'], result['errors'], prefix=prefix
        )
        
        if vis_file: print(f"  解の可視化を保存しました: {vis_file}")
    except Exception as e:
        print(f"  解の可視化エラー: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="CCD (高精度コンパクト差分法) CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本パラメータ
    parser.add_argument("--dim", type=int, choices=[1, 2, 3], default=1, help="次元 (1, 2, または 3)")
    parser.add_argument("--nx", type=int, default=16, help="x方向の格子点数")
    parser.add_argument("--ny", type=int, default=None, help="y方向の格子点数 (2D/3Dのみ)")
    parser.add_argument("--nz", type=int, default=None, help="z方向の格子点数 (3Dのみ)")
    parser.add_argument("--xrange", type=float, nargs=2, default=(-1.0, 1.0), metavar=('XMIN', 'XMAX'), help="x方向の範囲")
    parser.add_argument("--yrange", type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'), help="y方向の範囲")
    parser.add_argument("--zrange", type=float, nargs=2, default=None, metavar=('ZMIN', 'ZMAX'), help="z方向の範囲")
    
    # 方程式と関数
    parser.add_argument("-e", "--equation", type=str, default="poisson", help="方程式セット名 ('all'で全て)")
    parser.add_argument("-f", "--function", type=str, default="Sine", help="テスト関数名 ('all'で全て)")
    
    # ソルバー設定
    parser.add_argument("-b","--backend", type=str, choices=["cpu", "cuda", "jax", "opt"], default="cpu", help="計算バックエンド")
    parser.add_argument("-s","--solver", type=str, default="direct", help="ソルバー名 ('all'で全て)")
    parser.add_argument("--tol", type=float, default=1e-10, help="収束許容値")
    parser.add_argument("--maxiter", type=int, default=1000, help="最大反復回数")
    parser.add_argument("--restart", type=int, default=None, help="GMRES再起動パラメータ")
    parser.add_argument("--perturbation", type=float, default=None, help="初期値の摂動レベル")
    
    # 出力オプション
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細表示")
    parser.add_argument("--output", "-o", type=str, default=None, help="結果出力ファイル")
    parser.add_argument("--visualize", action="store_true", help="行列構造も可視化")
    parser.add_argument("--no-vis", action="store_true", help="解の可視化を無効化")
    parser.add_argument("--max-combinations", type=int, default=100, help="最大組み合わせ数")
    
    # リスト表示オプション
    parser.add_argument("--list", action="store_true", help="方程式セット一覧表示")
    parser.add_argument("--list-solvers", action="store_true", help="ソルバー一覧表示")
    parser.add_argument("--list-functions", action="store_true", help="テスト関数一覧表示")
    
    args = parser.parse_args()
    
    # リスト表示オプション
    if args.list: return get_options("equation_sets", print_mode=True) or 0
    if args.list_solvers: return get_options("solvers", print_mode=True) or 0
    if args.list_functions: return get_options("test_functions", print_mode=True) or 0
    
    # 'all'オプションの処理
    function_list = get_options("test_functions", args.dim) if args.function.lower() == 'all' else [args.function]
    equation_list = get_options("equation_sets", args.dim) if args.equation.lower() == 'all' else [args.equation]
    solver_list = get_options("solvers") if args.solver.lower() == 'all' else [args.solver]
    
    # 組み合わせ数のチェック
    total_combinations = len(function_list) * len(equation_list) * len(solver_list)
    if total_combinations > 1:
        print(f"実行する組み合わせ数: {total_combinations}")
        if total_combinations > args.max_combinations:
            print(f"警告: 組み合わせ数が多すぎます（最大: {args.max_combinations}）")
            print("--max-combinations オプションを増やすか、パラメータを絞ってください")
            return 1
    
    # グリッドを作成
    try:
        grid = create_grid(args.dim, args.nx, args.ny, args.nz, args.xrange, args.yrange, args.zrange)
    except Exception as e:
        print(f"グリッド作成エラー: {e}")
        return 1
    
    # ソルバーオプションの設定
    solver_options = {"backend": args.backend, "tol": args.tol, "maxiter": args.maxiter}
    if args.restart is not None: solver_options["restart"] = args.restart
    
    # 全組み合わせの結果を保存
    all_results = []
    
    # 方程式・ソルバー・関数の組み合わせでループ
    for equation_name in equation_list:
        for solver_name in solver_list:
            print(f"\nセットアップ: {equation_name}, {solver_name}")
            
            try:
                # テスター作成と設定
                tester = create_tester(args.dim, grid)
                tester.set_equation_set(equation_name)
                tester.setup(equation=equation_name, method=solver_name, 
                           options=solver_options, backend=args.backend)
                if args.perturbation is not None:
                    tester.perturbation_level = args.perturbation
                
                # 各関数に対してテスト実行
                for function_name in function_list:
                    print(f"  テスト関数: {function_name}")
                    test_func = tester.get_test_function(function_name)
                    
                    # テスト実行と結果保存
                    result = run_single_test(tester, test_func, args, args.verbose)
                    result.update({'equation': equation_name, 'solver': solver_name})
                    all_results.append(result)
                    
                    # 可視化
                    if not args.no_vis:
                        visualize_result(args.dim, grid, result, function_name, equation_name, solver_name)
                    
                    # 行列可視化
                    if args.visualize:
                        try:
                            tester.set_solver(method=solver_name, options=solver_options)
                            vis_output = tester.visualize_matrix_system(test_func)
                            print(f"  行列システムの可視化: {vis_output}")
                        except Exception as e:
                            print(f"  行列可視化エラー: {e}")
                            if args.verbose: import traceback; traceback.print_exc()
                
            except Exception as e:
                print(f"エラー: {e}")
                if args.verbose: import traceback; traceback.print_exc()
    
    # 結果サマリーの表示
    if len(all_results) > 1:
        print("\n=== 結果サマリー ===")
        print(f"実行した組み合わせ数: {len(all_results)}")
        
        # 最良の結果
        best_result = min(all_results, key=lambda x: x['errors'][0])
        print("\n最良の結果（ψ誤差で比較）:")
        print(f"  関数: {best_result['function']}")
        print(f"  方程式: {best_result['equation']}")
        print(f"  ソルバー: {best_result['solver']}")
        print(f"  エラー: {format_errors(best_result['errors'], args.dim)}")
        print(f"  計算時間: {best_result['elapsed_time']:.4f}秒")
        if best_result['iterations'] is not None:
            print(f"  反復回数: {best_result['iterations']}")
        
        # 計算時間の統計
        avg_time = sum(r['elapsed_time'] for r in all_results) / len(all_results)
        min_time = min(all_results, key=lambda x: x['elapsed_time'])
        max_time = max(all_results, key=lambda x: x['elapsed_time'])
        
        print("\n計算時間統計:")
        print(f"  平均時間: {avg_time:.4f}秒")
        print(f"  最短時間: {min_time['elapsed_time']:.4f}秒 ({min_time['function']}, {min_time['equation']}, {min_time['solver']})")
        print(f"  最長時間: {max_time['elapsed_time']:.4f}秒 ({max_time['function']}, {max_time['equation']}, {max_time['solver']})")
    
    # 結果の保存
    if args.output and all_results:
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