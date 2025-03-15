#!/usr/bin/env python3
"""
CCD Matrix Structure Verification Tool
高精度コンパクト差分法(CCD)の行列構造を可視化・検証するツール
"""

import os
import argparse
from typing import Dict, Tuple, Optional, Any, Union, List

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from grid import Grid
from equation_system import EquationSystem
from test_functions import TestFunctionFactory
from equation_sets import EquationSet
from tester import CCDTester1D, CCDTester2D
from scaling import plugin_manager


class CCDVerifier:
    """CCDメソッドの行列構造を検証・可視化するクラス"""
    
    def __init__(self, output_dir="verify_results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス（デフォルト: "verify_results"）
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_filename(self, title, scaling_method=None):
        """ファイル名を生成"""
        scaling_suffix = f"_{scaling_method.lower()}" if scaling_method else ""
        return os.path.join(self.output_dir, f"{title}{scaling_suffix}_matrix.png")
    
    def verify_equation_set(self, equation_set_name: str, dimension: int, 
                           nx: int, ny: Optional[int] = None,
                           scaling_method: Optional[str] = None,
                           solver_method: str = "direct") -> Dict:
        """
        方程式セットを検証
        
        Args:
            equation_set_name: 方程式セット名
            dimension: 次元（1または2）
            nx: x方向のグリッドサイズ
            ny: y方向のグリッドサイズ (2Dのみ)
            scaling_method: スケーリング手法名（オプション）
            solver_method: ソルバー手法名
            
        Returns:
            検証結果を含む辞書
        """
        # グリッド作成
        if dimension == 1:
            grid = Grid(nx, x_range=(-1.0, 1.0))
            test_func = TestFunctionFactory.create_standard_1d_functions()[3]  # Sine
        else:
            # 2Dの場合、nyが指定されていなければnxと同じにする
            if ny is None:
                ny = nx
            grid = Grid(nx, ny, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
            test_func = TestFunctionFactory.create_standard_2d_functions()[0]  # Sine2D
        
        # 方程式セット作成
        equation_set = EquationSet.create(equation_set_name, dimension=dimension)
        
        # テスター作成
        tester_class = CCDTester1D if dimension == 1 else CCDTester2D
        tester = tester_class(grid)
        
        # 重要: テスターにソルバーオプションを設定
        solver_options = {
            "tol": 1e-10,
            "maxiter": 1000,
            "analyze_matrix": True
        }
        tester.set_solver_options(solver_method, solver_options, analyze_matrix=True)
        tester.set_equation_set(equation_set_name)
        
        if scaling_method:
            tester.scaling_method = scaling_method
        
        # ソルバーの初期化を確実にするためにダミーテストを実行
        # これにより tester.solver が確実に初期化される
        dummy_result = tester.run_test_with_options(test_func)
        
        # 行列分析
        A = tester.solver.matrix_A
        
        # 右辺ベクトル構築
        b = self._build_rhs_vector(tester.solver, grid, test_func)
        
        # スケーリング適用（有効な場合）
        if scaling_method:
            # 正しいスケーリング処理を行う
            scaler = plugin_manager.get_plugin(scaling_method)
            if scaler:
                print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
            else:
                print(f"警告: スケーリング手法 {scaling_method} が見つかりません。スケーリングなしで処理します。")
                A_scaled, b_scaled = A, b
                scaling_info = None
                scaler = None
        else:
            A_scaled, b_scaled = A, b
            scaling_info = None
            scaler = None
        
        # 解ベクトルの計算
        try:
            if solver_method == "direct":
                from cupyx.scipy.sparse.linalg import spsolve
                x = spsolve(A_scaled, b_scaled)
            else:
                # 反復解法を使用
                from cupyx.scipy.sparse.linalg import gmres, cg, cgs, lsqr, minres, lsmr
                
                if solver_method == "gmres":
                    x, _ = gmres(A_scaled, b_scaled)
                elif solver_method == "cg":
                    x, _ = cg(A_scaled, b_scaled)
                elif solver_method == "cgs":
                    x, _ = cgs(A_scaled, b_scaled)
                elif solver_method == "lsqr":
                    x, _, _, _, _, _, _, _ = lsqr(A_scaled, b_scaled)
                elif solver_method == "minres":
                    x, _ = minres(A_scaled, b_scaled)
                elif solver_method == "lsmr":
                    x, _, _, _, _, _, _, _, _, _ = lsmr(A_scaled, b_scaled)
                else:
                    # デフォルトは直接解法
                    from cupyx.scipy.sparse.linalg import spsolve
                    x = spsolve(A_scaled, b_scaled)
                    
            # スケーリングされていた場合はアンスケール
            if scaling_method and scaler and scaling_info:
                x = scaler.unscale(x, scaling_info)
        except Exception as e:
            print(f"警告: ソルバー {solver_method} での解法に失敗しました: {e}")
            print("解と厳密解の比較は行いません。")
            x = None
            
        # 厳密解の計算
        if x is not None:
            try:
                if dimension == 1:
                    # 1Dの場合
                    exact_solution = cp.zeros_like(x)
                    
                    # 各成分（ψ, ψ', ψ'', ψ'''）ごとに計算
                    n_points = grid.n_points
                    for i in range(n_points):
                        xi = grid.get_point(i)
                        exact_solution[i*4] = test_func.f(xi)         # ψ
                        exact_solution[i*4+1] = test_func.df(xi)      # ψ'
                        exact_solution[i*4+2] = test_func.d2f(xi)     # ψ''
                        exact_solution[i*4+3] = test_func.d3f(xi)     # ψ'''
                else:
                    # 2Dの場合
                    exact_solution = cp.zeros_like(x)
                    
                    # 各成分（ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy）ごとに計算
                    nx, ny = grid.nx_points, grid.ny_points
                    for j in range(ny):
                        for i in range(nx):
                            idx = (j * nx + i) * 7  # 7は2Dでの1点あたりの未知数
                            x_val, y_val = grid.get_point(i, j)
                            exact_solution[idx] = test_func.f(x_val, y_val)            # ψ
                            exact_solution[idx+1] = test_func.df_dx(x_val, y_val)      # ψ_x
                            exact_solution[idx+2] = test_func.d2f_dx2(x_val, y_val)    # ψ_xx
                            exact_solution[idx+3] = test_func.d3f_dx3(x_val, y_val)    # ψ_xxx
                            exact_solution[idx+4] = test_func.df_dy(x_val, y_val)      # ψ_y
                            exact_solution[idx+5] = test_func.d2f_dy2(x_val, y_val)    # ψ_yy
                            exact_solution[idx+6] = test_func.d3f_dy3(x_val, y_val)    # ψ_yyy
            except Exception as e:
                print(f"警告: 厳密解の計算に失敗しました: {e}")
                print("解と厳密解の比較は行いません。")
                exact_solution = None
        else:
            exact_solution = None
        
        # 系の可視化
        title = f"{equation_set_name.capitalize()}{dimension}D_{test_func.name}"
        output_path = self.generate_filename(title, scaling_method)
        
        self.visualize_system(A_scaled, b_scaled, x, exact_solution, title, output_path)
        
        # 結果を返す
        sparsity = None
        if tester.solver.sparsity_info:
            sparsity = tester.solver.sparsity_info.get("sparsity")
        
        # エラー計算
        if x is not None and exact_solution is not None:
            error = cp.max(cp.abs(x - exact_solution))
            error = float(error.get() if hasattr(error, 'get') else error)
        else:
            error = None
        
        return {
            "equation_set": equation_set_name,
            "dimension": dimension,
            "nx": nx,
            "ny": ny if dimension == 2 else None,
            "matrix_size": A.shape[0],
            "non_zeros": A.nnz if hasattr(A, "nnz") else np.count_nonzero(A),
            "sparsity": sparsity,
            "scaling_method": scaling_method,
            "max_error": error,
            "output_path": output_path
        }
    
    def _build_rhs_vector(self, solver, grid, test_func):
        """
        右辺ベクトルの構築
        
        Args:
            solver: ソルバーインスタンス
            grid: グリッド
            test_func: テスト関数
            
        Returns:
            右辺ベクトル
        """
        if grid.is_2d:
            # 2Dケース
            nx, ny = grid.nx_points, grid.ny_points
            x_min, x_max = grid.x_min, grid.x_max
            y_min, y_max = grid.y_min, grid.y_max
            
            # ソース項
            f_values = cp.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    x, y = grid.get_point(i, j)
                    f_values[i, j] = test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)
            
            # 境界値
            boundary = {
                'left_dirichlet': cp.array([test_func.f(x_min, y) for y in grid.y]),
                'right_dirichlet': cp.array([test_func.f(x_max, y) for y in grid.y]),
                'bottom_dirichlet': cp.array([test_func.f(x, y_min) for x in grid.x]),
                'top_dirichlet': cp.array([test_func.f(x, y_max) for x in grid.x]),
                'left_neumann': cp.array([test_func.df_dx(x_min, y) for y in grid.y]),
                'right_neumann': cp.array([test_func.df_dx(x_max, y) for y in grid.y]),
                'bottom_neumann': cp.array([test_func.df_dy(x, y_min) for x in grid.x]),
                'top_neumann': cp.array([test_func.df_dy(x, y_max) for x in grid.x])
            }
            
            return solver._build_rhs_vector(f_values=f_values, **boundary)
        else:
            # 1Dケース
            x_min, x_max = grid.x_min, grid.x_max
            
            # ソース項とソリューション
            f_values = cp.array([test_func.d2f(x) for x in grid.x])
            
            # 境界値
            boundary = {
                'left_dirichlet': test_func.f(x_min),
                'right_dirichlet': test_func.f(x_max),
                'left_neumann': test_func.df(x_min),
                'right_neumann': test_func.df(x_max)
            }
            
            return solver._build_rhs_vector(f_values=f_values, **boundary)
    
    def visualize_system(self, A: Any, b: Any, x: Any, exact_x: Any, 
                         title: str, output_path: str) -> str:
        """
        システム Ax = b と解x、厳密解exact_xを可視化
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            x: 解ベクトル
            exact_x: 厳密解ベクトル
            title: 図のタイトル
            output_path: 出力ファイルのパス
            
        Returns:
            生成されたファイルのパス
        """
        # CuPy/SciPy配列をNumPy配列に変換
        def to_numpy(arr):
            if hasattr(arr, 'toarray'):
                return arr.toarray().get() if hasattr(arr, 'get') else arr.toarray()
            elif hasattr(arr, 'get'):
                return arr.get()
            return arr

        # 行列とベクトルを変換・整形
        A_np = to_numpy(A)
        b_np = to_numpy(b).reshape(-1, 1)
        x_np = to_numpy(x).reshape(-1, 1) if x is not None else None
        exact_x_np = to_numpy(exact_x).reshape(-1, 1) if exact_x is not None else None
        
        # 図の準備（2x3のグリッド）
        fig = plt.figure(figsize=(18, 12))
        
        # A行列の可視化
        ax1 = fig.add_subplot(2, 3, 1)
        abs_A = np.abs(A_np)
        non_zero_A = abs_A[abs_A > 0]
        
        if len(non_zero_A) > 0:
            im1 = ax1.imshow(
                abs_A, 
                norm=LogNorm(vmin=non_zero_A.min(), vmax=abs_A.max()),
                cmap='viridis', 
                aspect='auto', 
                interpolation='nearest'
            )
            plt.colorbar(im1, ax=ax1, label='Absolute Value (Log Scale)')
        ax1.set_title("Matrix A")
        ax1.set_xlabel("Column Index")
        ax1.set_ylabel("Row Index")
        
        # x（数値解）の可視化
        if x_np is not None:
            ax2 = fig.add_subplot(2, 3, 2)
            abs_x = np.abs(x_np)
            non_zero_x = abs_x[abs_x > 0]
            
            if len(non_zero_x) > 0:
                im2 = ax2.imshow(
                    abs_x, 
                    norm=LogNorm(vmin=non_zero_x.min(), vmax=abs_x.max()),
                    cmap='plasma', 
                    aspect='auto', 
                    interpolation='nearest'
                )
                plt.colorbar(im2, ax=ax2, label='Absolute Value (Log Scale)')
            ax2.set_title("Solution x")
            ax2.set_xlabel("Component")
            ax2.set_ylabel("Value Index")
        
        # b（右辺ベクトル）の可視化
        ax3 = fig.add_subplot(2, 3, 3)
        abs_b = np.abs(b_np)
        non_zero_b = abs_b[abs_b > 0]
        
        if len(non_zero_b) > 0:
            im3 = ax3.imshow(
                abs_b, 
                norm=LogNorm(vmin=non_zero_b.min(), vmax=abs_b.max()),
                cmap='cividis', 
                aspect='auto', 
                interpolation='nearest'
            )
            plt.colorbar(im3, ax=ax3, label='Absolute Value (Log Scale)')
        ax3.set_title("Right-hand side b")
        ax3.set_xlabel("Component")
        ax3.set_ylabel("Value Index")
        
        # exact_x（厳密解）の可視化
        if exact_x_np is not None:
            ax4 = fig.add_subplot(2, 3, 4)
            abs_exact_x = np.abs(exact_x_np)
            non_zero_exact_x = abs_exact_x[abs_exact_x > 0]
            
            if len(non_zero_exact_x) > 0:
                im4 = ax4.imshow(
                    abs_exact_x, 
                    norm=LogNorm(vmin=non_zero_exact_x.min(), vmax=abs_exact_x.max()),
                    cmap='plasma', 
                    aspect='auto', 
                    interpolation='nearest'
                )
                plt.colorbar(im4, ax=ax4, label='Absolute Value (Log Scale)')
            ax4.set_title("Exact Solution")
            ax4.set_xlabel("Component")
            ax4.set_ylabel("Value Index")
        
        # x - exact_x（残差）の可視化
        if x_np is not None and exact_x_np is not None:
            ax5 = fig.add_subplot(2, 3, 5)
            residual = np.abs(x_np - exact_x_np)
            non_zero_residual = residual[residual > 0]
            
            if len(non_zero_residual) > 0:
                im5 = ax5.imshow(
                    residual, 
                    norm=LogNorm(vmin=non_zero_residual.min(), vmax=residual.max()),
                    cmap='hot', 
                    aspect='auto', 
                    interpolation='nearest'
                )
                plt.colorbar(im5, ax=ax5, label='Absolute Value (Log Scale)')
            ax5.set_title("Solution Error (|x - exact_x|)")
            ax5.set_xlabel("Component")
            ax5.set_ylabel("Value Index")
            
            # 残差の統計情報
            max_error = np.max(residual)
            avg_error = np.mean(residual)
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.axis('off')  # 枠を非表示
            ax6.text(0.05, 0.9, "Error Statistics:", fontsize=12, weight='bold')
            ax6.text(0.05, 0.7, f"Maximum Error: {max_error:.4e}", fontsize=10)
            ax6.text(0.05, 0.6, f"Average Error: {avg_error:.4e}", fontsize=10)
            
            # 各成分の最大誤差
            if x_np.shape[0] > 4:  # 複数の点が含まれる場合
                ax6.text(0.05, 0.4, "Component-wise Max Error:", fontsize=10)
                
                if len(x_np) % 4 == 0:  # 1Dの場合（各点4成分）
                    component_names = ["ψ", "ψ'", "ψ''", "ψ'''"]
                    for i, name in enumerate(component_names):
                        indices = range(i, len(x_np), 4)
                        comp_error = np.max(np.abs(x_np[indices] - exact_x_np[indices]))
                        ax6.text(0.05, 0.3 - i*0.1, f"{name}: {comp_error:.4e}", fontsize=10)
                elif len(x_np) % 7 == 0:  # 2Dの場合（各点7成分）
                    component_names = ["ψ", "ψ_x", "ψ_xx", "ψ_xxx", "ψ_y", "ψ_yy", "ψ_yyy"]
                    for i, name in enumerate(component_names):
                        indices = range(i, len(x_np), 7)
                        comp_error = np.max(np.abs(x_np[indices] - exact_x_np[indices]))
                        ax6.text(0.05, 0.3 - i*0.07, f"{name}: {comp_error:.4e}", fontsize=9)
        
        plt.suptitle(f"System Values & Solutions: {title}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # タイトル用の余白を確保
        
        # 保存して後片付け
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def run_all_verifications(self, scaling_methods=None, solver_method="direct"):
        """
        すべての検証を実行
        
        Args:
            scaling_methods: 検証するスケーリング手法のリスト
            solver_method: ソルバー手法
            
        Returns:
            検証結果のリスト
        """
        equation_set_types = ["poisson", "derivative"]
        dimensions = [1, 2]
        grid_sizes = [10, 20]
        
        results = []
        
        for eq_type in equation_set_types:
            for dim in dimensions:
                for size in grid_sizes:
                    try:
                        # スケーリングなし検証
                        result = self.verify_equation_set(
                            eq_type, dim, size, 
                            solver_method=solver_method
                        )
                        results.append(result)
                        
                        # スケーリング有り検証
                        if scaling_methods:
                            for scaling in scaling_methods:
                                try:
                                    result = self.verify_equation_set(
                                        eq_type, dim, size, size, scaling,
                                        solver_method=solver_method
                                    )
                                    results.append(result)
                                except Exception as e:
                                    print(f"検証エラー ({eq_type}, {dim}D, サイズ={size}, スケーリング={scaling}): {e}")
                    except Exception as e:
                        print(f"検証エラー ({eq_type}, {dim}D, サイズ={size}): {e}")
        
        return results
    
    def summarize_results(self, results):
        """
        検証結果の要約表示
        
        Args:
            results: 検証結果のリスト
        """
        print("\n検証結果のまとめ:")
        print("{:<15} {:<5} {:<12} {:<15} {:<10} {:<10} {:<15}".format(
            "方程式セット", "次元", "グリッド", "スケーリング", "行列サイズ", "疎性率", "最大誤差"
        ))
        print("-" * 85)
        
        for result in results:
            equation_set = result["equation_set"]
            dimension = result["dimension"]
            
            # グリッドサイズ表示の調整
            if dimension == 1:
                grid_size = f"{result['nx']}"
            else:
                grid_size = f"{result['nx']}x{result['ny']}"
                
            scaling = result.get("scaling_method", "なし")
            matrix_size = result["matrix_size"]
            sparsity = result.get("sparsity", "N/A")
            sparsity_str = f"{sparsity:.4f}" if isinstance(sparsity, (int, float)) else "N/A"
            
            error = result.get("max_error", "N/A")
            error_str = f"{error:.4e}" if isinstance(error, (int, float)) else "N/A"
            
            print("{:<15} {:<5d} {:<12} {:<15} {:<10d} {:<10} {:<15}".format(
                equation_set, dimension, grid_size, str(scaling), matrix_size, sparsity_str, error_str
            ))


def run_verification(equation_set_name="poisson", dimension=1, nx=10, ny=None, 
                    scaling_method=None, output_dir="verify_results", solver_method="direct"):
    """
    検証処理のエントリポイント
    
    Args:
        equation_set_name: 方程式セット名
        dimension: 次元（1または2）
        nx: x方向のグリッドサイズ
        ny: y方向のグリッドサイズ（2Dのみ）
        scaling_method: スケーリング手法名
        output_dir: 出力ディレクトリ
        solver_method: ソルバー手法
        
    Returns:
        検証結果
    """
    verifier = CCDVerifier(output_dir=output_dir)
    
    result = verifier.verify_equation_set(
        equation_set_name, dimension, nx, ny,
        scaling_method, solver_method
    )
    
    # 結果表示
    dimension = result["dimension"]
    eqs = result["equation_set"]
    
    # グリッドサイズ表示の調整
    if dimension == 1:
        grid_info = f"{result['nx']}"
    else:
        grid_info = f"{result['nx']}x{result['ny']}"
        
    matrix_size = result["matrix_size"]
    scaling = result.get("scaling_method", "なし")
    sparsity = result.get("sparsity", "N/A")
    sparsity_str = f"{sparsity:.4f}" if isinstance(sparsity, (int, float)) else "N/A"
    error = result.get("max_error", "N/A")
    error_str = f"{error:.4e}" if isinstance(error, (int, float)) else "N/A"
    output_path = result.get("output_path", "")
    
    print(f"\n{dimension}D {eqs} 方程式の行列検証結果:")
    print(f"  グリッドサイズ: {grid_info}")
    print(f"  スケーリング: {scaling}")
    print(f"  行列サイズ: {matrix_size}")
    print(f"  疎性率: {sparsity_str}")
    print(f"  最大誤差: {error_str}")
    print(f"  可視化結果: {output_path}")
    
    return result


def run_all_verifications(output_dir="verify_results", scaling_methods=None, solver_method="direct"):
    """
    すべての検証を実行するエントリポイント
    
    Args:
        output_dir: 出力ディレクトリ
        scaling_methods: スケーリング手法名のリスト
        solver_method: ソルバー手法
        
    Returns:
        検証結果のリスト
    """
    print("==== CCD行列構造検証ツール ====")
    
    verifier = CCDVerifier(output_dir=output_dir)
    
    # デフォルトのスケーリング手法
    if scaling_methods is None:
        scaling_methods = ["SymmetricScaling"]
    
    # 検証実行
    results = verifier.run_all_verifications(scaling_methods, solver_method)
    
    # 結果表示
    verifier.summarize_results(results)
    
    print(f"\n検証が完了しました。結果は {output_dir} に保存されています。")
    
    return results


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="CCD行列構造検証ツール")
    
    # 基本設定
    parser.add_argument("--dim", type=int, choices=[1, 2], default=1, help="次元 (1 or 2)")
    parser.add_argument("-e", "--equation", type=str, default="poisson", help="方程式セット名")
    parser.add_argument("-o", "--out", type=str, default="verify_results", help="出力ディレクトリ")
    
    # グリッド設定
    parser.add_argument("--nx", type=int, default=10, help="x方向の格子点数")
    parser.add_argument("--ny", type=int, default=None, help="y方向の格子点数 (2Dのみ)")
    
    # ソルバー設定
    parser.add_argument("--solver", type=str, choices=['direct', 'gmres', 'cg', 'cgs', 'lsqr', 'minres', 'lsmr'], 
                       default='direct', help="ソルバー手法")
    
    # スケーリング設定
    parser.add_argument("--scaling", type=str, default=None, help="スケーリング手法")
    parser.add_argument("--list-scaling", action="store_true", help="スケーリング手法一覧を表示")
    
    # 検証モード
    parser.add_argument("--all", action="store_true", help="全方程式・次元・スケーリング手法で検証実行")
    
    return parser.parse_args()


def list_scaling_methods():
    """利用可能なスケーリング手法一覧表示"""
    plugins = plugin_manager.get_available_plugins()
    print("\n利用可能なスケーリング手法:")
    for name in plugins:
        print(f"- {name}")


if __name__ == "__main__":
    args = parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.out, exist_ok=True)
    
    # スケーリング手法一覧表示
    if args.list_scaling:
        list_scaling_methods()
        exit(0)
    
    if args.all:
        # すべての検証実行
        scaling_methods = [args.scaling] if args.scaling else None
        run_all_verifications(
            output_dir=args.out, 
            scaling_methods=scaling_methods,
            solver_method=args.solver
        )
    else:
        # 単一検証実行
        run_verification(
            equation_set_name=args.equation,
            dimension=args.dim,
            nx=args.nx,
            ny=args.ny,
            scaling_method=args.scaling,
            output_dir=args.out,
            solver_method=args.solver
        )