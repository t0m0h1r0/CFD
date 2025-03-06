#!/usr/bin/env python3
"""
CCD法 コマンドラインインターフェース（改良版）

1次元と2次元のCCD法を統一的に扱うコマンドラインツール
エラーハンドリングを強化した改良版
"""

import argparse
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod

# 1次元CCDのインポート
from grid_config import GridConfig
from ccd_solver import CCDSolver
from test_functions import TestFunction, TestFunctionFactory

# 2次元CCDのインポート（存在する場合）
try:
    from grid2d_config import Grid2DConfig
    from ccd2d_solver import CCD2DSolver
    from test_functions_adapter import TestFunction2D, create_test_functions
    ccd2d_available = True
except ImportError as e:
    print(f"2次元CCD関連のインポートエラー: {e}")
    ccd2d_available = False


class CCDTestRunner(ABC):
    """CCD法のテスト実行を行う抽象基底クラス (SOLID - OCP, ISP)"""

    def __init__(self, args):
        """引数を受け取って初期化"""
        self.args = args
    
    @abstractmethod
    def setup(self) -> bool:
        """テスト環境のセットアップ。成功したらTrueを返す"""
        pass
    
    @abstractmethod
    def run(self) -> Optional[Dict[str, Any]]:
        """テストを実行"""
        pass
    
    @abstractmethod
    def visualize(self, results: Dict[str, Any]) -> None:
        """結果を可視化"""
        pass


class CCD1DTestRunner(CCDTestRunner):
    """1次元CCD法のテスト実行クラス"""

    def setup(self) -> bool:
        """テスト環境のセットアップ"""
        print("1次元CCD法のテストを実行します")
        
        # テスト関数の取得
        self.test_functions = TestFunctionFactory.create_standard_functions()
        
        # 関数名から対象の関数を検索
        func_name = self.args.function
        self.test_func = None
        
        for tf in self.test_functions:
            if tf.name.lower() == func_name.lower():
                self.test_func = tf
                break
        
        if self.test_func is None:
            print(f"エラー: テスト関数 '{func_name}' が見つかりません")
            print("利用可能な関数:")
            for tf in self.test_functions:
                print(f"  {tf.name.lower()}: {tf.name}")
            return False
        
        print(f"テスト関数: {self.test_func.name}")
        return True
    
    def run(self) -> Optional[Dict[str, Any]]:
        """テストを実行して結果を返す"""
        try:
            # グリッドの設定
            n = self.args.n_points
            x_range = self.args.domain[:2]  # [x_min, x_max]
            L = x_range[1] - x_range[0]
            h = L / (n - 1)
            
            # グリッド点の生成
            x = cp.linspace(x_range[0], x_range[1], n)
            
            # 関数値の計算
            f_values = cp.array([self.test_func.f(float(xi)) for xi in x])
            
            # 解析解の計算
            df_analytical = cp.array([self.test_func.df(float(xi)) for xi in x])
            d2f_analytical = cp.array([self.test_func.d2f(float(xi)) for xi in x])
            d3f_analytical = cp.array([self.test_func.d3f(float(xi)) for xi in x])
            
            # グリッド設定
            grid_config = GridConfig(
                n_points=n,
                h=h,
                dirichlet_values=[float(f_values[0]), float(f_values[-1])],
                neumann_values=[float(df_analytical[0]), float(df_analytical[-1])],
                coeffs=[1.0, 0.0, 0.0, 0.0]  # f = ψ (デフォルト)
            )
            
            # ソルバーの設定
            solver_kwargs = {
                "use_iterative": not self.args.direct,
            }
            
            if self.args.iterative_solver:
                solver_kwargs["solver_type"] = self.args.iterative_solver
            
            # ソルバーの初期化
            solver = CCDSolver(
                grid_config,
                **solver_kwargs
            )
            
            # 解の計算（時間計測）
            start_time = time.time()
            psi, df_numerical, d2f_numerical, d3f_numerical = solver.solve(f_values)
            elapsed_time = time.time() - start_time
            
            print(f"計算時間: {elapsed_time:.4f}秒")
            
            # 誤差の計算
            df_error = cp.sqrt(cp.mean((df_numerical - df_analytical) ** 2))
            d2f_error = cp.sqrt(cp.mean((d2f_numerical - d2f_analytical) ** 2))
            d3f_error = cp.sqrt(cp.mean((d3f_numerical - d3f_analytical) ** 2))
            
            print(f"1階導関数のL2誤差: {df_error:.2e}")
            print(f"2階導関数のL2誤差: {d2f_error:.2e}")
            print(f"3階導関数のL2誤差: {d3f_error:.2e}")
            
            # 結果を辞書に格納
            return {
                "x": x,
                "f_values": f_values,
                "df_analytical": df_analytical,
                "df_numerical": df_numerical,
                "d2f_analytical": d2f_analytical,
                "d2f_numerical": d2f_numerical,
                "d3f_analytical": d3f_analytical,
                "d3f_numerical": d3f_numerical,
                "df_error": df_error,
                "d2f_error": d2f_error,
                "d3f_error": d3f_error,
                "elapsed_time": elapsed_time,
            }
        except Exception as e:
            print(f"1次元CCDテスト実行中にエラーが発生しました: {e}")
            traceback.print_exc()
            return None
    
    def visualize(self, results: Dict[str, Any]) -> None:
        """結果を可視化"""
        if not self.args.visualize or results is None:
            return
        
        try:
            # 出力ディレクトリの作成
            os.makedirs("results", exist_ok=True)
            
            # NumPy配列に変換
            x_np = cp.asnumpy(results["x"])
            f_np = cp.asnumpy(results["f_values"])
            df_analytical_np = cp.asnumpy(results["df_analytical"])
            df_numerical_np = cp.asnumpy(results["df_numerical"])
            d2f_analytical_np = cp.asnumpy(results["d2f_analytical"])
            d2f_numerical_np = cp.asnumpy(results["d2f_numerical"])
            d3f_analytical_np = cp.asnumpy(results["d3f_analytical"])
            d3f_numerical_np = cp.asnumpy(results["d3f_numerical"])
            
            # プロットの作成
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"1D CCD Results for {self.test_func.name} Function", fontsize=16)
            
            # 関数値
            axs[0, 0].plot(x_np, f_np, 'b-', label='Function')
            axs[0, 0].set_title("Function")
            axs[0, 0].set_xlabel("x")
            axs[0, 0].set_ylabel("f(x)")
            axs[0, 0].grid(True)
            axs[0, 0].legend()
            
            # 1階導関数
            axs[0, 1].plot(x_np, df_analytical_np, 'b-', label='Analytical')
            axs[0, 1].plot(x_np, df_numerical_np, 'r--', label='Numerical')
            axs[0, 1].set_title(f"First Derivative (Error: {results['df_error']:.2e})")
            axs[0, 1].set_xlabel("x")
            axs[0, 1].set_ylabel("df/dx")
            axs[0, 1].grid(True)
            axs[0, 1].legend()
            
            # 2階導関数
            axs[1, 0].plot(x_np, d2f_analytical_np, 'b-', label='Analytical')
            axs[1, 0].plot(x_np, d2f_numerical_np, 'r--', label='Numerical')
            axs[1, 0].set_title(f"Second Derivative (Error: {results['d2f_error']:.2e})")
            axs[1, 0].set_xlabel("x")
            axs[1, 0].set_ylabel("d²f/dx²")
            axs[1, 0].grid(True)
            axs[1, 0].legend()
            
            # 3階導関数
            axs[1, 1].plot(x_np, d3f_analytical_np, 'b-', label='Analytical')
            axs[1, 1].plot(x_np, d3f_numerical_np, 'r--', label='Numerical')
            axs[1, 1].set_title(f"Third Derivative (Error: {results['d3f_error']:.2e})")
            axs[1, 1].set_xlabel("x")
            axs[1, 1].set_ylabel("d³f/dx³")
            axs[1, 1].grid(True)
            axs[1, 1].legend()
            
            plt.tight_layout()
            
            # 保存と表示
            n = self.args.n_points
            save_path = f"results/ccd1d_{self.test_func.name.lower()}_{n}points.png"
            plt.savefig(save_path, dpi=150)
            print(f"結果を '{save_path}' に保存しました")
            
            if not self.args.no_show:
                plt.show()
        except Exception as e:
            print(f"可視化中にエラーが発生しました: {e}")
            traceback.print_exc()


class CCD2DTestRunner(CCDTestRunner):
    """2次元CCD法のテスト実行クラス"""

    def setup(self) -> bool:
        """テスト環境のセットアップ"""
        if not ccd2d_available:
            print("エラー: 2次元CCD法のモジュールがインポートできません。")
            print("必要なファイル (grid2d_config.py, ccd2d_solver.py など) が利用可能であることを確認してください。")
            return False
        
        print("2次元CCD法のテストを実行します")
        
        try:
            # テスト関数の取得
            self.test_functions = create_test_functions()
            
            # 関数名から対象の関数を検索
            func_name = self.args.function
            self.test_func = None
            
            for tf in self.test_functions:
                if tf.name.lower() == func_name.lower():
                    self.test_func = tf
                    break
            
            if self.test_func is None:
                print(f"エラー: テスト関数 '{func_name}' が見つかりません")
                print("利用可能な関数:")
                for tf in self.test_functions:
                    print(f"  {tf.name.lower()}: {tf.name}")
                return False
            
            print(f"テスト関数: {self.test_func.name}")
            return True
        except Exception as e:
            print(f"2次元CCDテストのセットアップでエラーが発生しました: {e}")
            traceback.print_exc()
            return False
    
    def run(self) -> Optional[Dict[str, Any]]:
        """テストを実行して結果を返す"""
        try:
            # グリッドの設定
            nx = self.args.n_points
            ny = self.args.n_points
            x_min, x_max, y_min, y_max = self.args.domain  # [x_min, x_max, y_min, y_max]
            hx = (x_max - x_min) / (nx - 1)
            hy = (y_max - y_min) / (ny - 1)
            
            # グリッド点の生成
            x = cp.linspace(x_min, x_max, nx)
            y = cp.linspace(y_min, y_max, ny)
            
            # 関数値の計算 (2D配列)
            f_values = cp.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    f_values[i, j] = self.test_func.f(x[i], y[j])
            
            # 解析解の計算
            analytical_results = {}
            for deriv_name, func in {
                "f": self.test_func.f,
                "f_x": self.test_func.f_x,
                "f_y": self.test_func.f_y,
                "f_xx": self.test_func.f_xx,
                "f_yy": self.test_func.f_yy,
                "f_xy": self.test_func.f_xy
            }.items():
                result = cp.zeros((nx, ny))
                for i in range(nx):
                    for j in range(ny):
                        result[i, j] = func(x[i], y[j])
                analytical_results[deriv_name] = result
            
            # グリッド設定
            # ディリクレ境界条件の設定（単純化のため、境界上の関数値を使用）
            x_dirichlet_values = []
            for j in range(ny):
                left_val = float(self.test_func.f(x_min, y[j]))
                right_val = float(self.test_func.f(x_max, y[j]))
                x_dirichlet_values.append((left_val, right_val))
            
            y_dirichlet_values = []
            for i in range(nx):
                bottom_val = float(self.test_func.f(x[i], y_min))
                top_val = float(self.test_func.f(x[i], y_max))
                y_dirichlet_values.append((bottom_val, top_val))
            
            # x方向とy方向のノイマン境界条件値
            x_neumann_values = []
            for j in range(ny):
                left_val = float(self.test_func.f_x(x_min, y[j]))
                right_val = float(self.test_func.f_x(x_max, y[j]))
                x_neumann_values.append((left_val, right_val))
            
            y_neumann_values = []
            for i in range(nx):
                bottom_val = float(self.test_func.f_y(x[i], y_min))
                top_val = float(self.test_func.f_y(x[i], y_max))
                y_neumann_values.append((bottom_val, top_val))
            
            grid_config = Grid2DConfig(
                nx=nx,
                ny=ny,
                hx=hx,
                hy=hy,
                x_deriv_order=2,  # 2次までの偏導関数を考慮
                y_deriv_order=2,
                mixed_deriv_order=1,
                x_dirichlet_values=x_dirichlet_values,
                y_dirichlet_values=y_dirichlet_values,
                x_neumann_values=x_neumann_values,
                y_neumann_values=y_neumann_values,
            )
            
            # ソルバーの設定
            solver_kwargs = {
                "use_iterative": not self.args.direct,
            }
            
            if self.args.iterative_solver:
                solver_kwargs["solver_type"] = self.args.iterative_solver
            
            # ソルバーの初期化
            try:
                print("CCD2Dソルバーを初期化中...")
                solver = CCD2DSolver(
                    grid_config,
                    **solver_kwargs
                )
                print("ソルバーの初期化完了")
            except Exception as e:
                print(f"ソルバー初期化エラー: {e}")
                traceback.print_exc()
                return None
            
            # 解の計算（時間計測）
            try:
                print("方程式を解いています...")
                start_time = time.time()
                numerical_results = solver.solve(f_values)
                elapsed_time = time.time() - start_time
                print("方程式の解法完了")
            except Exception as e:
                print(f"ソルバー実行エラー: {e}")
                traceback.print_exc()
                return None
            
            print(f"計算時間: {elapsed_time:.4f}秒")
            
            # 誤差の計算
            errors = {}
            for deriv_name in ["f_x", "f_y", "f_xx", "f_yy", "f_xy"]:
                if deriv_name in numerical_results and deriv_name in analytical_results:
                    error = float(cp.sqrt(cp.mean((numerical_results[deriv_name] - analytical_results[deriv_name]) ** 2)))
                    errors[deriv_name] = error
                    print(f"{deriv_name}のL2誤差: {error:.2e}")
            
            # 結果を辞書に格納
            return {
                "x": x,
                "y": y,
                "f_values": f_values,
                "analytical_results": analytical_results,
                "numerical_results": numerical_results,
                "errors": errors,
                "elapsed_time": elapsed_time,
            }
        except Exception as e:
            print(f"2次元CCDテスト実行中にエラーが発生しました: {e}")
            traceback.print_exc()
            return None
    
    def visualize(self, results: Dict[str, Any]) -> None:
        """結果を可視化"""
        if not self.args.visualize or results is None:
            return
        
        try:
            # 出力ディレクトリの作成
            os.makedirs("results", exist_ok=True)
            
            # NumPy配列に変換
            x_np = cp.asnumpy(results["x"])
            y_np = cp.asnumpy(results["y"])
            X, Y = np.meshgrid(x_np, y_np, indexing='ij')
            
            # プロットの作成
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f"2D CCD Results for {self.test_func.name} Function", fontsize=16)
            
            plot_components = ["f", "f_x", "f_y", "f_xx", "f_yy", "f_xy"]
            plot_titles = {
                "f": "Function",
                "f_x": "∂f/∂x",
                "f_y": "∂f/∂y",
                "f_xx": "∂²f/∂x²",
                "f_yy": "∂²f/∂y²",
                "f_xy": "∂²f/∂x∂y"
            }
            
            for i, comp in enumerate(plot_components):
                if (comp in results["numerical_results"] and 
                    comp in results["analytical_results"]):
                    # NumPy配列に変換
                    Z_num = cp.asnumpy(results["numerical_results"][comp])
                    Z_ana = cp.asnumpy(results["analytical_results"][comp])
                    
                    # エラー計算
                    error_str = f" (Error: {results['errors'].get(comp, 0.0):.2e})" if comp in results["errors"] else ""
                    
                    # サブプロット
                    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
                    
                    # ワイヤーフレームプロット
                    ax.plot_wireframe(X, Y, Z_num, color='red', alpha=0.5, label='Numerical')
                    ax.plot_wireframe(X, Y, Z_ana, color='blue', alpha=0.5, label='Analytical')
                    
                    ax.set_title(f"{plot_titles[comp]}{error_str}")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.legend()
            
            plt.tight_layout()
            
            # 保存と表示
            nx = self.args.n_points
            save_path = f"results/ccd2d_{self.test_func.name.lower()}_{nx}x{nx}points.png"
            plt.savefig(save_path, dpi=150)
            print(f"結果を '{save_path}' に保存しました")
            
            if not self.args.no_show:
                plt.show()
        except Exception as e:
            print(f"可視化中にエラーが発生しました: {e}")
            traceback.print_exc()


class CCDTestRunnerFactory:
    """テストランナーを生成するファクトリークラス (SOLID - DIP)"""
    
    @staticmethod
    def create_runner(command: str, args) -> Optional[CCDTestRunner]:
        """コマンドに基づいてテストランナーを作成"""
        if command == "1d":
            return CCD1DTestRunner(args)
        elif command == "2d":
            return CCD2DTestRunner(args)
        else:
            return None


def list_available_functions():
    """利用可能なテスト関数を一覧表示"""
    try:
        print("利用可能なテスト関数:")
        
        print("\n1次元テスト関数:")
        for func in TestFunctionFactory.create_standard_functions():
            print(f"  {func.name.lower()}: {func.name}")
        
        print("\n2次元テスト関数:")
        if ccd2d_available:
            for func in create_test_functions():
                print(f"  {func.name.lower()}: {func.name}")
        else:
            print("  (2次元CCD法モジュールが利用できません)")
    except Exception as e:
        print(f"関数一覧表示中にエラーが発生しました: {e}")
        traceback.print_exc()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="CCD法テストツール")
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")
    
    # 1次元CCDコマンド
    parser_1d = subparsers.add_parser("1d", help="1次元CCDテスト")
    parser_1d.add_argument("--function", "-f", default="sine", help="テスト関数名")
    parser_1d.add_argument("--n-points", "-n", type=int, default=51, help="格子点数")
    parser_1d.add_argument("--domain", type=float, nargs="+", default=[-1.0, 1.0], help="計算領域 [x_min x_max]")
    parser_1d.add_argument("--direct", action="store_true", help="直接法を使用")
    parser_1d.add_argument("--iterative-solver", choices=["gmres", "cg", "bicgstab", "lsqr"], help="反復ソルバーの種類")
    parser_1d.add_argument("--visualize", "-v", action="store_true", help="結果を可視化")
    parser_1d.add_argument("--no-show", action="store_true", help="結果の表示を抑制（保存のみ）")
    
    # 2次元CCDコマンド
    parser_2d = subparsers.add_parser("2d", help="2次元CCDテスト")
    parser_2d.add_argument("--function", "-f", default="gaussian", help="テスト関数名")
    parser_2d.add_argument("--n-points", "-n", type=int, default=21, help="各方向の格子点数")
    parser_2d.add_argument("--domain", type=float, nargs="+", default=[-1.0, 1.0, -1.0, 1.0], help="計算領域 [x_min x_max y_min y_max]")
    parser_2d.add_argument("--direct", action="store_true", help="直接法を使用")
    parser_2d.add_argument("--iterative-solver", choices=["gmres", "cg", "bicgstab", "lsqr"], help="反復ソルバーの種類")
    parser_2d.add_argument("--visualize", "-v", action="store_true", help="結果を可視化")
    parser_2d.add_argument("--no-show", action="store_true", help="結果の表示を抑制（保存のみ）")
    
    # 関数一覧コマンド
    parser_list = subparsers.add_parser("list", help="利用可能な関数を一覧表示")
    
    # コマンドライン引数の解析
    args = parser.parse_args()
    
    # コマンドの実行
    if args.command == "list":
        list_available_functions()
    else:
        # テストランナーの作成
        runner = CCDTestRunnerFactory.create_runner(args.command, args)
        
        if runner:
            # セットアップ
            if runner.setup():
                # テスト実行
                results = runner.run()
                if results:
                    # 結果の可視化
                    runner.visualize(results)
                else:
                    print("テスト実行に失敗しました。")
        else:
            parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"実行中にエラーが発生しました: {e}")
        traceback.print_exc()