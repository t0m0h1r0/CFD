import argparse
import os
import time
import numpy as np
from core.grid.grid1d import Grid1D
from core.grid.grid2d import Grid2D
from core.grid.grid3d import Grid3D
from tester.tester1d import CCDTester1D
from tester.tester2d import CCDTester2D
from tester.tester3d import CCDTester3D
from test_function.test_function1d import TestFunction1DFactory
from test_function.test_function2d import TestFunction2DFactory
from test_function.test_function3d import TestFunction3DFactory
from visualizer.visualizer1d import CCDVisualizer1D
from visualizer.visualizer2d import CCDVisualizer2D
from visualizer.visualizer3d import CCDVisualizer3D
from scaling import plugin_manager
from linear_solver import create_solver

def parse_args():
    parser = argparse.ArgumentParser(description="CCD法ソルバー")
    # 基本設定
    parser.add_argument("--dim", type=int, choices=[1, 2, 3], default=1, help="次元")
    parser.add_argument("--func", type=str, default="Sine", help="テスト関数('all'=全関数)")
    parser.add_argument("-e", "--equation", type=str, default="poisson", help="方程式セット")
    parser.add_argument("-o", "--out", type=str, default="results", help="出力ディレクトリ")
    parser.add_argument("--prefix", type=str, default="", help="出力ファイル接頭辞")
    
    # グリッド
    parser.add_argument("--nx", type=int, default=21, help="x方向点数")
    parser.add_argument("--ny", type=int, default=21, help="y方向点数(2D/3Dのみ)")
    parser.add_argument("--nz", type=int, default=21, help="z方向点数(3Dのみ)")
    parser.add_argument("--xrange", type=float, nargs=2, default=[-1.0, 1.0], help="x範囲")
    parser.add_argument("--yrange", type=float, nargs=2, default=[-1.0, 1.0], help="y範囲(2D/3Dのみ)")
    parser.add_argument("--zrange", type=float, nargs=2, default=[-1.0, 1.0], help="z範囲(3Dのみ)")
    
    # ソルバー
    parser.add_argument("--solver", type=str, default='direct', help="解法('all'=全解法)")
    parser.add_argument("--scaling", type=str, default=None, help="スケーリング手法('all'=全手法)")
    parser.add_argument("--backend", type=str, choices=['cpu', 'cuda', 'jax'], default='cpu', help="計算バックエンド")
    parser.add_argument("--tol", type=float, default=1e-10, help="許容誤差")
    parser.add_argument("--maxiter", type=int, default=1000, help="最大反復数")
    parser.add_argument("--perturbation", type=float, default=None, 
                      help="厳密解を初期値とし、この値の比率で乱数摂動を加える (0=摂動なし、None=厳密解を使用しない)")
    
    # モード
    parser.add_argument("--list", action="store_true", help="関数一覧表示")
    parser.add_argument("--list-scaling", action="store_true", help="スケーリング手法一覧")
    parser.add_argument("--list-solvers", action="store_true", help="ソルバー一覧")
    parser.add_argument("--converge", action="store_true", help="格子収束テスト")
    parser.add_argument("--verify", action="store_true", help="行列構造検証")
    parser.add_argument("--verbose", action="store_true", help="詳細情報表示")
    
    return parser.parse_args()

def create_tester(args):
    grid = _create_grid(args.nx, 
                       args.ny if args.dim >= 2 else None,
                       args.nz if args.dim == 3 else None,
                       x_range=tuple(args.xrange), 
                       y_range=tuple(args.yrange) if args.dim >= 2 else None,
                       z_range=tuple(args.zrange) if args.dim == 3 else None)
    
    if args.dim == 3:
        tester = CCDTester3D(grid)
    elif args.dim == 2:
        tester = CCDTester2D(grid)
    else:
        tester = CCDTester1D(grid)
        
    tester.set_equation_set(args.equation)
    tester.backend = args.backend
    
    # ソルバーオプションを設定
    tester.set_solver_options(args.solver, {
        "tol": args.tol, 
        "maxiter": args.maxiter, 
        "backend": args.backend
    })
    
    tester.scaling_method = args.scaling
    tester.perturbation_level = args.perturbation
    
    return tester

def get_functions(args):
    if args.func == "all":
        if args.dim == 3:
            funcs = TestFunction1DFactory.create_standard_functions()
        elif args.dim == 2:
            funcs = TestFunction2DFactory.create_standard_functions()
        else:
            funcs = TestFunction3DFactory.create_standard_functions()
            
        if args.list:
            print(f"\n利用可能な{args.dim}D関数:")
            for f in funcs:
                print(f"- {f.name}")
        return funcs
    else:
        tester = create_tester(args)
        return [tester.get_test_function(args.func)]

def get_scaling_methods(args):
    # スケーリングプラグイン読み込み時の出力を抑制
    plugin_manager.verbose = args.verbose
    
    if args.scaling == "all":
        methods = plugin_manager.get_available_plugins()
        if args.list_scaling:
            print("\n利用可能なスケーリング手法:")
            for m in methods:
                print(f"- {m}")
        return methods
    elif args.scaling:
        return [args.scaling]
    else:
        return [None]

def get_solvers(args):
    if args.solver == "all":
        try:
            # テスト実行に最小限必要なダミー行列
            if args.dim == 3:
                dummy_size = 20 * 10  # 3D: 10変数/点
            elif args.dim == 2:
                dummy_size = 20 * 7   # 2D: 7変数/点
            else:
                dummy_size = 20 * 4   # 1D: 4変数/点
                
            dummy_A = np.eye(dummy_size)
            
            # ソルバー作成
            solver = create_solver(dummy_A, backend=args.backend)
            methods = solver.get_available_solvers()
            
            # 空の場合はデフォルト
            if not methods:
                methods = ["direct"]
                
            if args.list_solvers:
                print(f"\n{args.backend}バックエンドのソルバー:")
                for m in methods:
                    print(f"- {m}")
                    
            return methods
        except Exception as e:
            if args.verbose:
                print(f"ソルバー情報取得エラー: {e}")
            return ["direct"]
    else:
        return [args.solver]

def generate_filename(args, func_name, solver, scaling, extension="png"):
    """標準化されたファイル名を生成"""
    scaling_str = scaling if scaling else "no_scaling"
    
    if args.dim == 3:
        size_str = f"{args.nx}x{args.ny}x{args.nz}"
    elif args.dim == 2:
        size_str = f"{args.nx}x{args.ny}"
    else:
        size_str = f"{args.nx}"
        
    perturbation_str = f"_pert{args.perturbation}" if args.perturbation is not None else ""
    
    # 標準ファイル名パターン: バックエンド_解法_関数_スケーリング_サイズ_摂動
    filename = f"{args.backend}_{solver}_{func_name}_{scaling_str}_{size_str}{perturbation_str}.{extension}"
    
    # プレフィックスがある場合は追加
    if args.prefix:
        filename = f"{args.prefix}_{filename}"
        
    return os.path.join(args.out, filename)

def _create_grid(nx, ny=None, nz=None, x_range=(-1.0, 1.0), y_range=None, z_range=None):
    """次元に応じたグリッドを作成"""
    if ny is None:
        return Grid1D(nx, x_range=x_range)
    elif nz is None:
        return Grid2D(nx, ny, x_range=x_range, y_range=y_range or x_range)
    else:
        return Grid3D(nx, ny, nz, x_range=x_range, y_range=y_range or x_range, z_range=z_range or x_range)

def run_tests(args):
    """関数・ソルバー・スケーリングでのテスト実行"""
    # テスト対象取得
    functions = get_functions(args)
    solvers = get_solvers(args)
    scaling_methods = get_scaling_methods(args)
    
    # テスト状況表示
    multi_test = len(functions) > 1 or len(solvers) > 1 or len(scaling_methods) > 1
    if multi_test and args.verbose:
        print(f"\nCombination test: {len(functions)} functions x {len(solvers)} solvers x {len(scaling_methods)} scaling methods")
    
    # 実行結果格納
    results = []
    
    # テスト実行
    for func in functions:
        for scaling in scaling_methods:
            for solver in solvers:
                # テスト作成
                tester = create_tester(args)
                tester.scaling_method = scaling
                tester.solver_method = solver
                
                # 実行
                perturbation_info = ""
                if args.perturbation is not None and args.verbose:
                    perturbation_info = f", Perturbation: {args.perturbation}"
                
                print(f"\nFunction: {func.name}, Scaling: {scaling or 'None'}, Solver: {solver}{perturbation_info}")
                try:
                    start_time = time.time()
                    result = tester.run_test(func)
                    elapsed = time.time() - start_time
                    
                    # 結果情報
                    max_error = max(result['errors'])
                    iterations = getattr(tester.solver, 'last_iterations', None)
                    print(f"Time: {elapsed:.4f}s, Max Error: {max_error:.6e}" + 
                         (f", Iterations: {iterations}" if iterations is not None else ""))
                    
                    # 結果データ
                    result_data = {
                        'function': func.name,
                        'scaling': scaling or "None",
                        'solver': solver,
                        'time': elapsed,
                        'max_error': max_error,
                        'iterations': iterations,
                        'errors': result['errors']
                    }
                    
                    # 結果保存
                    results.append(result_data)
                    
                    # 単一テスト時は可視化
                    if args.dim == 3:
                        visualizer = CCDVisualizer3D(output_dir=args.out)
                    elif args.dim == 2:
                        visualizer = CCDVisualizer2D(output_dir=args.out)
                    else:
                        visualizer = CCDVisualizer1D(output_dir=args.out)
                        
                    # 標準化されたファイル名を生成
                    out_filename = generate_filename(args, func.name, solver, scaling, "png")
                    out_prefix = os.path.basename(out_filename).split('.')[0]
                    
                    if args.dim == 3:
                        visualizer.visualize_solution(
                            tester.grid, func.name, result["numerical"],
                            result["exact"], result["errors"], prefix=out_prefix
                        )
                    elif args.dim == 2:
                        visualizer.visualize_solution(
                            tester.grid, func.name, result["numerical"],
                            result["exact"], result["errors"], prefix=out_prefix
                        )
                    else:
                        visualizer.visualize_derivatives(
                            tester.grid, func.name, result["numerical"],
                            result["exact"], result["errors"], prefix=out_prefix
                        )
                    
                    if args.verbose:
                        print(f"Result visualization saved to: {out_filename}")
                    
                except Exception as e:
                    print(f"Test error: {str(e)}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
        
    return results

def run_convergence_test(args):
    """格子収束性テスト実行"""
    tester = create_tester(args)
    functions = get_functions(args)
    
    # 収束テスト用グリッドサイズ
    if args.dim == 3:
        grid_sizes = [11, 16, 21, 26]  # 3Dは計算コストが高いので小さめのサイズ
    elif args.dim == 2:
        grid_sizes = [11, 21, 31, 41]  # 中間サイズ
    else:
        grid_sizes = [11, 21, 41, 81]  # 1Dは高解像度でも計算が早い
    
    for func in functions:
        print(f"\nGrid convergence test for {func.name}...")
        results = tester.run_convergence_test(
            func, grid_sizes, args.xrange, 
            args.yrange if args.dim >= 2 else None,
            args.zrange if args.dim == 3 else None
        )
        
        # 可視化
        if args.dim == 3:
            visualizer = CCDVisualizer3D(output_dir=args.out)
        elif args.dim == 2:
            visualizer = CCDVisualizer2D(output_dir=args.out)
        else:
            visualizer = CCDVisualizer1D(output_dir=args.out)
            
        # 標準化ファイル名
        out_filename = generate_filename(args, func.name, "convergence", args.scaling, "png")
        out_prefix = os.path.basename(out_filename).split('.')[0]
        
        # 可視化実行
        visualizer.visualize_grid_convergence(func.name, grid_sizes, results, prefix=out_prefix)
        if args.verbose:
            print(f"Convergence visualization saved with prefix: {out_prefix}")

def run_verification(args):
    """行列構造検証の実行"""
    verify_dir = os.path.join(args.out, "verify")
    os.makedirs(verify_dir, exist_ok=True)
    
    # テスターを作成
    tester = create_tester(args)
    tester.results_dir = verify_dir
    functions = get_functions(args)
    
    for func in functions:
        print(f"\nMatrix structure verification for {args.equation} equation in {args.dim}D: {func.name}")
        
        # 標準化ファイル名をベース名として設定
        out_filename = generate_filename(args, func.name, "matrix", args.scaling, "png")
        out_basename = os.path.basename(out_filename).split('.')[0]
        
        # ベース名設定
        tester.matrix_basename = out_basename
        
        # 検証実行
        output_path = tester.visualize_matrix_system(func)
        if args.verbose:
            print(f"Matrix visualization saved to: {output_path}")

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    # プラグインマネージャーに修正を適用
    import scaling.plugin_manager
    scaling.plugin_manager.verbose = args.verbose
    
    # 3D固有の問題を修正（エッジ名の形式を修正）
    if args.dim == 3:
        # Grid3DクラスのエッジとCornerの命名を補正
        from core.grid.grid3d import Grid3D
        original_get_boundary_type = Grid3D.get_boundary_type
        
        def patched_get_boundary_type(self, i, j=None, k=None):
            """エッジ名のフォーマットを修正するパッチ"""
            boundary_type = original_get_boundary_type(self, i, j, k)
            
            # エッジ名の形式が誤っている場合の修正
            if boundary_type.startswith('edge_x_min_y_z_min'):
                # 例: 'edge_x_min_y_z_min' → 'edge_x_y_min_z_min'
                boundary_type = 'edge_x_y_min_z_min'
            # 他のエッジ形式も必要に応じて修正
            
            return boundary_type
        
        # グリッドクラスにパッチを適用（実験的 - 実際にはより適切な修正が必要）
        Grid3D.get_boundary_type = patched_get_boundary_type
    
    # 一覧表示モードの処理
    if args.list:
        get_functions(args)
        return
    
    if args.list_scaling:
        get_scaling_methods(args)
        return
    
    if args.list_solvers:
        get_solvers(args)
        return
    
    # 各モードの実行
    if args.verify:
        run_verification(args)
    elif args.converge:
        run_convergence_test(args)
    else:
        run_tests(args)

if __name__ == "__main__":
    main()