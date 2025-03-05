"""
2次元CCD法テスターモジュール

2次元CCDソルバー実装のテスト機能を提供します。
1次元CCDテスターを2次元に拡張したものです。
"""

import jax.numpy as jnp
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable, Any, Type

from grid_2d_config import Grid2DConfig
from ccd2d_solver import CCD2DSolver


class TestFunction2D:
    """2次元テスト関数とその導関数を保持するクラス"""
    
    def __init__(
        self,
        name: str,
        f: Callable[[float, float], float],
        fx: Callable[[float, float], float],
        fy: Callable[[float, float], float],
        fxx: Callable[[float, float], float],
        fyy: Callable[[float, float], float],
        fxy: Optional[Callable[[float, float], float]] = None,
        laplacian: Optional[Callable[[float, float], float]] = None
    ):
        """
        初期化
        
        Args:
            name: 関数名
            f: 関数 f(x,y)
            fx: x方向偏微分 ∂f/∂x
            fy: y方向偏微分 ∂f/∂y
            fxx: x方向2階偏微分 ∂²f/∂x²
            fyy: y方向2階偏微分 ∂²f/∂y²
            fxy: 交差偏微分 ∂²f/∂x∂y
            laplacian: ラプラシアン ∇²f = ∂²f/∂x² + ∂²f/∂y²
        """
        self.name = name
        self.f = f
        self.fx = fx
        self.fy = fy
        self.fxx = fxx
        self.fyy = fyy
        
        # 交差偏微分がなければ自動生成
        if fxy is None:
            self.fxy = lambda x, y: 0.0  # デフォルトはゼロ
        else:
            self.fxy = fxy
            
        # ラプラシアンがなければ自動生成
        if laplacian is None:
            self.laplacian = lambda x, y: self.fxx(x, y) + self.fyy(x, y)
        else:
            self.laplacian = laplacian


class TestFunctionFactory2D:
    """2次元テスト関数を生成するファクトリークラス"""
    
    @staticmethod
    def create_standard_functions() -> List[TestFunction2D]:
        """標準的な2次元テスト関数セットを生成"""
        
        pi = jnp.pi
        
        functions = [
            TestFunction2D(
                name="Zero",
                f=lambda x, y: 0.0,
                fx=lambda x, y: 0.0,
                fy=lambda x, y: 0.0,
                fxx=lambda x, y: 0.0,
                fyy=lambda x, y: 0.0
            ),
            TestFunction2D(
                name="SinProduct",
                f=lambda x, y: jnp.sin(pi * x) * jnp.sin(pi * y),
                fx=lambda x, y: pi * jnp.cos(pi * x) * jnp.sin(pi * y),
                fy=lambda x, y: pi * jnp.sin(pi * x) * jnp.cos(pi * y),
                fxx=lambda x, y: -pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y),
                fyy=lambda x, y: -pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y),
                fxy=lambda x, y: pi**2 * jnp.cos(pi * x) * jnp.cos(pi * y),
                laplacian=lambda x, y: -2 * pi**2 * jnp.sin(pi * x) * jnp.sin(pi * y)
            ),
            TestFunction2D(
                name="Polynomial",
                f=lambda x, y: x**2 * y**2 * (1 - x) * (1 - y),
                fx=lambda x, y: 2 * x * y**2 * (1 - x) * (1 - y) - x**2 * y**2 * (1 - y),
                fy=lambda x, y: 2 * x**2 * y * (1 - x) * (1 - y) - x**2 * y**2 * (1 - x),
                fxx=lambda x, y: 2 * y**2 * (1 - x) * (1 - y) - 4 * x * y**2 * (1 - y) + 2 * x**2 * y**2 * (1 - y),
                fyy=lambda x, y: 2 * x**2 * (1 - x) * (1 - y) - 4 * x**2 * y * (1 - x) + 2 * x**2 * y**2 * (1 - x),
                fxy=lambda x, y: (2*x*y*(1-x)*(1-y) - 2*x*y**2*(1-x) - 2*x**2*y*(1-y) + 2*x**2*y**2)
            ),
            TestFunction2D(
                name="GaussianPeak",
                f=lambda x, y: jnp.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)),
                fx=lambda x, y: -20 * (x - 0.5) * jnp.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)),
                fy=lambda x, y: -20 * (y - 0.5) * jnp.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)),
                fxx=lambda x, y: (-20 + 400 * (x - 0.5)**2) * jnp.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)),
                fyy=lambda x, y: (-20 + 400 * (y - 0.5)**2) * jnp.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)),
                fxy=lambda x, y: 400 * (x - 0.5) * (y - 0.5) * jnp.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2)),
                laplacian=lambda x, y: (
                    (-40 + 400 * ((x - 0.5)**2 + (y - 0.5)**2)) * 
                    jnp.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2))
                )
            )
        ]
        
        return functions


class CCD2DMethodTester:
    """2次元CCD法のテストを実行するクラス"""
    
    def __init__(
        self,
        grid_config: Grid2DConfig,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        test_functions: Optional[List[TestFunction2D]] = None,
        coeffs: Optional[List[float]] = None,
    ):
        """
        Args:
            grid_config: 2次元グリッド設定
            solver_kwargs: ソルバーの初期化パラメータ
            test_functions: テスト関数のリスト（指定がなければ標準関数セットを使用）
            coeffs: 係数リスト（指定がなければgrid_configから使用）
        """
        self.grid_config = grid_config
        self.solver_kwargs = solver_kwargs or {}
        
        # グリッドパラメータ
        self.nx, self.ny = grid_config.nx, grid_config.ny
        self.hx, self.hy = grid_config.hx, grid_config.hy
        
        # x、y座標配列
        self.x = jnp.linspace(0, 1, self.nx)  # 標準区間 [0,1] を使用
        self.y = jnp.linspace(0, 1, self.ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        
        # テスト関数の設定
        if test_functions is None:
            self.test_functions = TestFunctionFactory2D.create_standard_functions()
        else:
            self.test_functions = test_functions
            
        # 係数の設定
        if coeffs is not None:
            self.coeffs = coeffs
            grid_config.coeffs = coeffs
        else:
            self.coeffs = grid_config.coeffs
        
        # ソルバーキーワード引数の確認と調整
        # CCD2DSolverに渡せるパラメータのみを抽出
        filtered_solver_kwargs = {}
        for key, value in self.solver_kwargs.items():
            if key in ["use_direct_solver"]:  # CCD2DSolverが受け付けるパラメータのみ
                filtered_solver_kwargs[key] = value
        
        # ソルバーの初期化
        self.solver = CCD2DSolver(grid_config, **filtered_solver_kwargs)
    
    def compute_errors(
        self, test_func: TestFunction2D
    ) -> Tuple[Dict[str, float], float]:
        """
        各導関数の誤差を計算
        
        Args:
            test_func: テスト関数
            
        Returns:
            (誤差の辞書, 計算時間)
        """
        # 解析解
        f_values = jnp.vectorize(test_func.f)(self.X, self.Y)
        fx_values = jnp.vectorize(test_func.fx)(self.X, self.Y)
        fy_values = jnp.vectorize(test_func.fy)(self.X, self.Y)
        fxx_values = jnp.vectorize(test_func.fxx)(self.X, self.Y)
        fyy_values = jnp.vectorize(test_func.fyy)(self.X, self.Y)
        
        if hasattr(test_func, 'fxy') and test_func.fxy is not None:
            fxy_values = jnp.vectorize(test_func.fxy)(self.X, self.Y)
        else:
            fxy_values = jnp.zeros_like(f_values)
        
        # ディリクレ境界条件を設定
        dirichlet_values = {
            'left': f_values[:, 0],
            'right': f_values[:, -1],
            'bottom': f_values[0, :],
            'top': f_values[-1, :]
        }
        
        # グリッド設定を更新
        self.grid_config.dirichlet_values = dirichlet_values
        
        # 右辺関数を計算（係数に基づく）
        a, b, c, d, e, f = (self.coeffs + [0, 0, 0, 0, 0, 0])[:6]
        
        # 右辺関数 = a*f + b*fx + c*fxx + d*fy + e*fyy + f*fxy
        rhs = (a * f_values + b * fx_values + c * fxx_values + 
              d * fy_values + e * fyy_values + f * fxy_values)
        
        # ソルバーを初期化
        solver = CCD2DSolver(self.grid_config, **self.solver_kwargs)
        
        # 時間計測
        start_time = time.time()
        
        # 数値解を計算
        solution = solver.solve(rhs)
        
        # 計測終了
        elapsed_time = time.time() - start_time
        
        # 数値解の各成分
        u_numerical = solution['u']
        ux_numerical = solution.get('ux', None)
        uy_numerical = solution.get('uy', None)
        uxx_numerical = solution.get('uxx', None)
        
        # 誤差計算（L2ノルム）
        errors = {}
        
        # 関数値の誤差
        errors['u'] = float(jnp.sqrt(jnp.mean((u_numerical - f_values)**2)))
        
        # 微分値の誤差（ソルバーから対応する値が得られた場合のみ）
        if ux_numerical is not None:
            errors['ux'] = float(jnp.sqrt(jnp.mean((ux_numerical - fx_values)**2)))
        
        if uy_numerical is not None:
            errors['uy'] = float(jnp.sqrt(jnp.mean((uy_numerical - fy_values)**2)))
        
        if uxx_numerical is not None:
            errors['uxx'] = float(jnp.sqrt(jnp.mean((uxx_numerical - fxx_values)**2)))
        
        return errors, elapsed_time
    
    def run_tests(self, prefix: str = "", output_dir: str = "results", visualize: bool = True) -> Dict[str, Tuple[Dict[str, float], float]]:
        """
        すべてのテスト関数に対してテストを実行
        
        Args:
            prefix: 出力ファイルの接頭辞
            output_dir: 出力ディレクトリ
            visualize: 可視化を行うかどうか
            
        Returns:
            テスト結果の辞書 {関数名: (誤差辞書, 計算時間)}
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        avg_errors = {'u': 0.0, 'ux': 0.0, 'uy': 0.0, 'uxx': 0.0}
        total_time = 0.0
        
        # 結果テーブルのヘッダーを表示
        coeff_str = f" (coeffs={self.coeffs})"
        print(f"2次元CCD法のテスト結果{coeff_str}:")
        print("-" * 80)
        print(f"{'関数':<15} {'u誤差':<12} {'ux誤差':<12} {'uy誤差':<12} {'uxx誤差':<12} {'計算時間(秒)':<12}")
        print("-" * 80)
        
        for test_func in self.test_functions:
            # 誤差と時間を計算
            errors, elapsed_time = self.compute_errors(test_func)
            results[test_func.name] = (errors, elapsed_time)
            
            # 結果を表示
            error_u = errors.get('u', float('nan'))
            error_ux = errors.get('ux', float('nan'))
            error_uy = errors.get('uy', float('nan'))
            error_uxx = errors.get('uxx', float('nan'))
            
            print(f"{test_func.name:<15} {error_u:<12.2e} {error_ux:<12.2e} {error_uy:<12.2e} {error_uxx:<12.2e} {elapsed_time:<12.4f}")
            
            # 誤差を累積（存在する場合のみ）
            for k in avg_errors.keys():
                if k in errors:
                    avg_errors[k] += errors[k]
            
            total_time += elapsed_time
            
            # 可視化
            if visualize:
                self._visualize_results(test_func, output_dir, prefix)
        
        # 平均誤差を計算
        count = len(self.test_functions)
        for k in avg_errors.keys():
            avg_errors[k] /= count
        avg_time = total_time / count
        
        # 平均結果を表示
        print("-" * 80)
        print(f"{'平均':<15} {avg_errors['u']:<12.2e} {avg_errors['ux']:<12.2e} {avg_errors['uy']:<12.2e} {avg_errors['uxx']:<12.2e} {avg_time:<12.4f}")
        print("-" * 80)
        
        return results
    
    def _visualize_results(self, test_func: TestFunction2D, output_dir: str, prefix: str):
        """
        テスト結果を可視化
        
        Args:
            test_func: 現在のテスト関数
            output_dir: 出力ディレクトリ
            prefix: ファイル名の接頭辞
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析解
        f_values = jnp.vectorize(test_func.f)(self.X, self.Y)
        fx_values = jnp.vectorize(test_func.fx)(self.X, self.Y)
        fy_values = jnp.vectorize(test_func.fy)(self.X, self.Y)
        fxx_values = jnp.vectorize(test_func.fxx)(self.X, self.Y)
        fyy_values = jnp.vectorize(test_func.fyy)(self.X, self.Y)
        
        if hasattr(test_func, 'fxy') and test_func.fxy is not None:
            fxy_values = jnp.vectorize(test_func.fxy)(self.X, self.Y)
        else:
            fxy_values = jnp.zeros_like(f_values)
        
        # 境界条件を設定
        dirichlet_values = {
            'left': f_values[:, 0],
            'right': f_values[:, -1],
            'bottom': f_values[0, :],
            'top': f_values[-1, :]
        }
        
        # グリッド設定を更新
        self.grid_config.dirichlet_values = dirichlet_values
        
        # 右辺関数を計算（係数に基づく）
        a, b, c, d, e, f = (self.coeffs + [0, 0, 0, 0, 0, 0])[:6]
        
        # 右辺関数 = a*f + b*fx + c*fxx + d*fy + e*fyy + f*fxy
        rhs = (a * f_values + b * fx_values + c * fxx_values + 
              d * fy_values + e * fyy_values + f * fxy_values)
        
        # 数値解を計算
        solution = self.solver.solve(rhs)
        u_numerical = solution['u']
        
        # 誤差を計算
        error = jnp.abs(u_numerical - f_values)
        l2_error = jnp.sqrt(jnp.mean(error**2))
        max_error = jnp.max(error)
        
        # プロットの作成
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 数値解
        im0 = axes[0].pcolormesh(self.X, self.Y, u_numerical, cmap='viridis', shading='auto')
        axes[0].set_title(f'数値解 ({test_func.name})')
        fig.colorbar(im0, ax=axes[0])
        
        # 解析解
        im1 = axes[1].pcolormesh(self.X, self.Y, f_values, cmap='viridis', shading='auto')
        axes[1].set_title('解析解')
        fig.colorbar(im1, ax=axes[1])
        
        # 誤差
        im2 = axes[2].pcolormesh(self.X, self.Y, error, cmap='hot', shading='auto')
        axes[2].set_title(f'誤差 (L2: {l2_error:.2e}, Max: {max_error:.2e})')
        fig.colorbar(im2, ax=axes[2])
        
        # すべてのプロットに座標軸ラベルを設定
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        # ファイル名を生成
        func_name = test_func.name.lower()
        coeff_str = '_'.join(map(str, self.coeffs))
        filename = f"{prefix}{func_name}_2d_coeff_{coeff_str}.png"
        filepath = os.path.join(output_dir, filename)
        
        # プロットを保存
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        axes[1].set_title('解析解')
        fig.colorbar(im1, ax=axes[1])
        
        # 誤差
        im2 = axes[2].pcolormesh(self.X, self.Y, error, cmap='hot', shading='auto')
        axes[2].set_title(f'誤差 (L2: {l2_error:.2e}, Max: {max_error:.2e})')
        fig.colorbar(im2, ax=axes[2])
        
        # すべてのプロットに座標軸ラベルを設定
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        # ファイル名を生成
        func_name = test_func.name.lower()
        coeff_str = '_'.join(map(str, self.coeffs))
        filename = f"{prefix}{func_name}_2d_coeff_{coeff_str}.png"
        filepath = os.path.join(output_dir, filename)
        
        # プロットを保存
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        
    def analyze_single_function(
        self, 
        test_func: TestFunction2D, 
        output_dir: str = "results", 
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        単一のテスト関数に対して詳細な分析を実行
        
        Args:
            test_func: テスト関数
            output_dir: 出力ディレクトリ
            prefix: ファイル名の接頭辞
            
        Returns:
            分析結果の辞書
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析解
        f_values = jnp.vectorize(test_func.f)(self.X, self.Y)
        fx_values = jnp.vectorize(test_func.fx)(self.X, self.Y)
        fy_values = jnp.vectorize(test_func.fy)(self.X, self.Y)
        fxx_values = jnp.vectorize(test_func.fxx)(self.X, self.Y)
        fyy_values = jnp.vectorize(test_func.fyy)(self.X, self.Y)
        
        if hasattr(test_func, 'fxy') and test_func.fxy is not None:
            fxy_values = jnp.vectorize(test_func.fxy)(self.X, self.Y)
        else:
            fxy_values = jnp.zeros_like(f_values)
        
        # ラプラシアン
        if hasattr(test_func, 'laplacian') and test_func.laplacian is not None:
            laplacian_values = jnp.vectorize(test_func.laplacian)(self.X, self.Y)
        else:
            laplacian_values = fxx_values + fyy_values
        
        # 境界条件を設定
        dirichlet_values = {
            'left': f_values[:, 0],
            'right': f_values[:, -1],
            'bottom': f_values[0, :],
            'top': f_values[-1, :]
        }
        
        # グリッド設定を更新
        self.grid_config.dirichlet_values = dirichlet_values
        
        # 右辺関数を計算（係数に基づく）
        a, b, c, d, e, f = (self.coeffs + [0, 0, 0, 0, 0, 0])[:6]
        
        # 右辺関数 = a*f + b*fx + c*fxx + d*fy + e*fyy + f*fxy
        rhs = (a * f_values + b * fx_values + c * fxx_values + 
              d * fy_values + e * fyy_values + f * fxy_values)
        
        # 時間計測
        start_time = time.time()
        
        # 数値解を計算
        solution = self.solver.solve(rhs)
        
        # 計測終了
        elapsed_time = time.time() - start_time
        
        # 数値解の各成分
        u_numerical = solution['u']
        ux_numerical = solution.get('ux', None)
        uy_numerical = solution.get('uy', None)
        uxx_numerical = solution.get('uxx', None)
        
        # 誤差計算
        errors = {}
        errors['u'] = float(jnp.sqrt(jnp.mean((u_numerical - f_values)**2)))
        
        if ux_numerical is not None:
            errors['ux'] = float(jnp.sqrt(jnp.mean((ux_numerical - fx_values)**2)))
        
        if uy_numerical is not None:
            errors['uy'] = float(jnp.sqrt(jnp.mean((uy_numerical - fy_values)**2)))
        
        if uxx_numerical is not None:
            errors['uxx'] = float(jnp.sqrt(jnp.mean((uxx_numerical - fxx_values)**2)))
        
        # ラプラシアンの計算（数値解から）
        # 注: この部分は数値微分を使用していますが、より洗練された方法も可能です
        if ux_numerical is not None and uxx_numerical is not None:
            if 'uyy' in solution:
                uyy_numerical = solution['uyy']
            else:
                # 中央差分を使用して数値的に計算
                uyy_numerical = jnp.zeros_like(u_numerical)
                uyy_numerical[1:-1, 1:-1] = (
                    u_numerical[2:, 1:-1] - 2 * u_numerical[1:-1, 1:-1] + u_numerical[:-2, 1:-1]
                ) / (self.hy ** 2)
            
            laplacian_numerical = uxx_numerical + uyy_numerical
            errors['laplacian'] = float(jnp.sqrt(jnp.mean(
                (laplacian_numerical - laplacian_values)**2)))
        
        # 詳細なプロットの作成
        self._create_detailed_plots(
            test_func, f_values, u_numerical, rhs, 
            solution, errors, elapsed_time, output_dir, prefix
        )
        
        # 分析結果を返す
        return {
            'function_name': test_func.name,
            'errors': errors,
            'elapsed_time': elapsed_time,
            'grid_size': (self.nx, self.ny),
            'coefficients': self.coeffs
        }
        
    def _create_detailed_plots(
        self, 
        test_func: TestFunction2D,
        f_values: jnp.ndarray,
        u_numerical: jnp.ndarray,
        rhs: jnp.ndarray,
        solution: Dict[str, jnp.ndarray],
        errors: Dict[str, float],
        elapsed_time: float,
        output_dir: str,
        prefix: str
    ):
        """
        詳細な分析プロットを作成
        
        Args:
            test_func: テスト関数
            f_values: 解析解
            u_numerical: 数値解
            rhs: 右辺関数
            solution: ソルバーからの解の辞書
            errors: 誤差の辞書
            elapsed_time: 計算時間
            output_dir: 出力ディレクトリ
            prefix: ファイル名の接頭辞
        """
        # 誤差を計算
        error = jnp.abs(u_numerical - f_values)
        
        # 1. 基本的な可視化（解析解、数値解、誤差）
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
        
        # 数値解
        im1 = axes1[0, 0].pcolormesh(self.X, self.Y, u_numerical, cmap='viridis', shading='auto')
        axes1[0, 0].set_title('数値解')
        fig1.colorbar(im1, ax=axes1[0, 0])
        
        # 解析解
        im2 = axes1[0, 1].pcolormesh(self.X, self.Y, f_values, cmap='viridis', shading='auto')
        axes1[0, 1].set_title('解析解')
        fig1.colorbar(im2, ax=axes1[0, 1])
        
        # 誤差（対数スケール）
        im3 = axes1[1, 0].pcolormesh(self.X, self.Y, error, cmap='hot', shading='auto',
                                   norm=plt.cm.colors.LogNorm(vmin=max(1e-15, error.min()), 
                                                             vmax=max(1e-14, error.max())))
        axes1[1, 0].set_title(f'誤差 (対数スケール)')
        fig1.colorbar(im3, ax=axes1[1, 0])
        
        # 右辺関数
        im4 = axes1[1, 1].pcolormesh(self.X, self.Y, rhs, cmap='RdBu', shading='auto', 
                                   vmin=-max(abs(rhs.min()), abs(rhs.max())),
                                   vmax=max(abs(rhs.min()), abs(rhs.max())))
        axes1[1, 1].set_title('右辺関数')
        fig1.colorbar(im4, ax=axes1[1, 1])
        
        # すべてのプロットにラベルを設定
        for ax in axes1.flat:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        # 情報テキストを追加
        plt.figtext(0.5, 0.01, 
                   f"関数: {test_func.name}, L2誤差: {errors['u']:.2e}, 計算時間: {elapsed_time:.4f}秒\n"
                   f"グリッドサイズ: {self.nx}×{self.ny}, 係数: {self.coeffs}",
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # ファイル名を生成して保存
        func_name = test_func.name.lower()
        coeff_str = '_'.join(map(str, self.coeffs))
        filename1 = f"{prefix}{func_name}_2d_analysis_{coeff_str}.png"
        filepath1 = os.path.join(output_dir, filename1)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filepath1, dpi=150)
        plt.close(fig1)
        
        # 2. 微分値の可視化（利用可能な場合）
        if 'ux' in solution and 'uy' in solution:
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
            
            # x方向微分
            im5 = axes2[0, 0].pcolormesh(self.X, self.Y, solution['ux'], cmap='RdBu', 
                                        shading='auto', vmin=-2, vmax=2)
            axes2[0, 0].set_title('∂u/∂x (数値解)')
            fig2.colorbar(im5, ax=axes2[0, 0])
            
            # y方向微分
            im6 = axes2[0, 1].pcolormesh(self.X, self.Y, solution['uy'], cmap='RdBu', 
                                        shading='auto', vmin=-2, vmax=2)
            axes2[0, 1].set_title('∂u/∂y (数値解)')
            fig2.colorbar(im6, ax=axes2[0, 1])
            
            # x方向微分誤差
            fx_values = jnp.vectorize(test_func.fx)(self.X, self.Y)
            ux_error = jnp.abs(solution['ux'] - fx_values)
            im7 = axes2[1, 0].pcolormesh(self.X, self.Y, ux_error, cmap='hot', 
                                        shading='auto', norm=plt.cm.colors.LogNorm())
            axes2[1, 0].set_title(f'∂u/∂x 誤差 (L2: {errors.get("ux", 0):.2e})')
            fig2.colorbar(im7, ax=axes2[1, 0])
            
            # y方向微分誤差
            fy_values = jnp.vectorize(test_func.fy)(self.X, self.Y)
            uy_error = jnp.abs(solution['uy'] - fy_values)
            im8 = axes2[1, 1].pcolormesh(self.X, self.Y, uy_error, cmap='hot', 
                                        shading='auto', norm=plt.cm.colors.LogNorm())
            axes2[1, 1].set_title(f'∂u/∂y 誤差 (L2: {errors.get("uy", 0):.2e})')
            fig2.colorbar(im8, ax=axes2[1, 1])
            
            # すべてのプロットにラベルを設定
            for ax in axes2.flat:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            
            # ファイル名を生成して保存
            filename2 = f"{prefix}{func_name}_2d_derivatives_{coeff_str}.png"
            filepath2 = os.path.join(output_dir, filename2)
            
            plt.tight_layout()
            plt.savefig(filepath2, dpi=150)
            plt.close(fig2)