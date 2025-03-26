"""
高精度コンパクト差分法 (CCD) の2次元結果可視化

このモジュールは、2次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。視覚的に分かりやすい統合ダッシュボード形式で結果を表示します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from typing import List

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer2D(BaseVisualizer):
    """CCDソルバーの2D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_2d"):
        """初期化"""
        super().__init__(output_dir)
    
    def generate_filename(self, func_name, nx_points, ny_points=None, prefix=""):
        """ファイル名を生成"""
        if ny_points is None:
            ny_points = nx_points
            
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{nx_points}x{ny_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{nx_points}x{ny_points}_points.png"
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        2Dソリューションを可視化（統合ダッシュボード形式）
        
        Args:
            grid: Grid2D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト
            exact: 厳密解のリスト
            errors: 誤差リスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            保存に成功したかどうかのブール値
        """
        # グリッド情報取得
        X, Y = grid.get_points()
        nx_points, ny_points = grid.nx_points, grid.ny_points
        
        # NumPy配列に変換
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        
        # 可視化するソリューションのリスト
        solution_names = self.get_error_types()
        
        # ダッシュボード作成 (3x3グリッド)
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 3, height_ratios=[1, 1, 0.6])
        
        # 主要な成分の可視化 (最初の6つ)
        display_count = min(6, len(solution_names))
        for i in range(display_count):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            
            # データ取得と変換
            num_np = self._to_numpy(numerical[i])
            err = errors[i]
            
            # コンターマップの表示
            im = ax.contourf(X_np, Y_np, num_np, 20, cmap='viridis')
            ax.set_title(f"{solution_names[i]} (Error: {err:.2e})")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
            
            # 等高線の追加
            ax.contour(X_np, Y_np, num_np, 8, colors='k', linewidths=0.5, alpha=0.5)
        
        # 誤差分布ヒートマップ (3行目左側)
        ax_err = fig.add_subplot(gs[2, 0])
        
        # 主要成分(ψ)の誤差
        num_np = self._to_numpy(numerical[0])
        ex_np = self._to_numpy(exact[0])
        error_np = np.abs(num_np - ex_np)
        
        # 対数スケールで誤差表示
        err_min = max(np.min(error_np[error_np > 0]) if np.any(error_np > 0) else 1e-10, 1e-15)
        im_err = ax_err.pcolormesh(
            X_np, Y_np, error_np, 
            norm=LogNorm(vmin=err_min, vmax=np.max(error_np)),
            cmap='hot', shading='auto'
        )
        
        ax_err.set_title("Error Distribution (ψ)")
        ax_err.set_xlabel('X')
        ax_err.set_ylabel('Y')
        plt.colorbar(im_err, ax=ax_err)
        
        # 誤差サマリーグラフ (3行目中央と右側に結合)
        ax_summary = fig.add_subplot(gs[2, 1:])
        self._plot_error_summary(ax_summary, solution_names, errors)
        
        # 全体タイトル
        plt.suptitle(
            f"{function_name} Function Analysis ({nx_points}x{ny_points} points)", 
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存と表示
        if save:
            filepath = self.generate_filename(function_name, nx_points, ny_points, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def _plot_error_summary(self, ax, component_names, errors):
        """誤差サマリーバーチャートを描画"""
        x_pos = np.arange(len(component_names))
        
        # 対数スケール用の前処理
        plot_errors = np.array(errors).copy()
        plot_errors[plot_errors == 0] = self.min_log_value
        
        # バーチャート描画
        bars = ax.bar(x_pos, plot_errors)
        ax.set_yscale('log')
        ax.set_title("Error Summary")
        ax.set_xlabel('Solution Component')
        ax.set_ylabel('Maximum Error (log scale)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(component_names, rotation=45, ha="right")
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 値のアノテーション
        for i, (bar, err) in enumerate(zip(bars, errors)):
            ax.annotate(
                f'{err:.2e}',
                xy=(i, max(err, self.min_log_value) * 1.1),
                ha='center', va='bottom',
                fontsize=9
            )
    
    def compare_all_functions_errors(self, results_summary, grid_size, prefix="", save=True, dpi=150, show=False):
        """
        すべてのテスト関数の誤差を比較するグラフを生成
        
        Args:
            results_summary: 関数名をキーとし、誤差リストを値とする辞書
            grid_size: グリッドサイズ
            prefix: 接頭辞
            save: 保存するかどうか
            dpi: 画像のDPI値
            show: 表示するかどうか
            
        Returns:
            出力ファイルパス
        """
        func_names = list(results_summary.keys())
        error_types = self.get_error_types()
        
        # 最大4行2列のレイアウトで各成分の誤差を表示
        rows = (len(error_types) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i, error_type in enumerate(error_types):
            if i < len(axes):
                ax = axes[i]
                # 各関数の対応する誤差を取得
                orig_errors = [results_summary[name][i] for name in func_names]
                
                # 0値を置き換え
                plot_errors = [max(err, self.min_log_value) for err in orig_errors]
                
                # バーチャート
                x_pos = np.arange(len(func_names))
                bars = ax.bar(x_pos, plot_errors)
                ax.set_yscale('log')
                ax.set_title(f"{error_type}")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(func_names, rotation=45, ha="right", fontsize=8)
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # 値のアノテーション
                for j, (err, pos) in enumerate(zip(orig_errors, x_pos)):
                    label = "0.0" if err == 0.0 else f"{err:.1e}"
                    ax.annotate(
                        label,
                        xy=(pos, plot_errors[j] * 1.1),
                        ha='center', va='bottom',
                        fontsize=7, rotation=90
                    )
        
        # 全体タイトル
        plt.suptitle(f"Error Comparison ({grid_size}x{grid_size} points)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存と表示
        if save:
            filename = f"{self.output_dir}/{prefix}_function_comparison_{grid_size}x{grid_size}.png"
            if not prefix:
                filename = f"{self.output_dir}/function_comparison_{grid_size}x{grid_size}.png"
            plt.savefig(filename, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filename
    
    def visualize_grid_convergence(self, function_name, grid_sizes, results, prefix="", save=True, show=False, dpi=150):
        """
        グリッド収束性のグラフを生成
        
        Args:
            function_name: 関数名
            grid_sizes: グリッドサイズのリスト
            results: 結果データ
            prefix: 出力ファイルの接頭辞
            save: 保存するかどうか
            show: 表示するかどうか
            dpi: 画像のDPI値
            
        Returns:
            成功したかどうかのブール値
        """
        solution_names = self.get_error_types()
        
        # 最大4行2列のレイアウトで各成分の収束性を表示
        rows = (len(solution_names) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        # グリッド間隔計算
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        for i, name in enumerate(solution_names):
            if i < len(axes):
                ax = axes[i]
                # 各グリッドサイズでの誤差を取得
                errors = [results[n][i] for n in grid_sizes]
                
                # すべて0の場合は別処理
                if all(err == 0 for err in errors):
                    ax.text(0.5, 0.5, "All errors are zero", 
                            ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # 0値を置き換え
                plot_errors = [max(err, self.min_log_value) for err in errors]
                
                # 対数-対数プロット
                ax.loglog(grid_spacings, plot_errors, 'o-', label=name)
                
                # 収束次数の参照線
                if min(plot_errors) > 0:
                    x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                    for order, style, color in zip([2, 4], ['--', '-.'], ['r', 'g']):
                        scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                        y_ref = scale * x_ref ** order
                        ax.loglog(x_ref, y_ref, style, color=color, label=f'O(h^{order})')
                
                ax.set_title(f"{name}")
                ax.set_xlabel('Grid Spacing (h)')
                ax.set_ylabel('Error')
                ax.grid(True, which='both')
                ax.legend(fontsize=8)
        
        # 全体タイトル
        plt.suptitle(f"Grid Convergence: {function_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存と表示
        if save:
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def get_dimension_label(self) -> str:
        """次元ラベルを返す"""
        return "2D"
    
    def get_error_types(self) -> List[str]:
        """エラータイプのリストを返す"""
        return ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]