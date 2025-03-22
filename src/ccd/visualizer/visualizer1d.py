"""
高精度コンパクト差分法 (CCD) の1次元結果可視化

このモジュールは、1次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。可視化結果を統合し、一つのファイルに出力します。
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer1D(BaseVisualizer):
    """CCDソルバーの1D結果を可視化するクラス"""

    def __init__(self, output_dir="results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス
        """
        super().__init__(output_dir)

    def visualize_derivatives(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        導関数の結果を可視化 (統合版)
        
        Args:
            grid: Gridオブジェクト
            function_name: テスト関数名
            numerical: 数値解のリスト [psi, psi', psi'', psi''']
            exact: 厳密解のリスト
            errors: 誤差リスト
            prefix: 出力ファイルの接頭辞
            save: 保存するかどうか
            show: 表示するかどうか
            dpi: 画像のDPI値
            
        Returns:
            出力ファイルパス
        """
        x = grid.get_points()
        n_points = grid.n_points
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]

        # NumPy配列に変換
        x_np = self._to_numpy(x)

        # 1つの大きな図を作成 (3行2列: 上2行が微分結果、下段が誤差サマリー)
        fig = plt.figure(figsize=(14, 12))
        
        # グリッド定義 - 上部2x2のグリッド(微分結果用)と下部1x1(誤差サマリー用)
        gs = plt.GridSpec(3, 2, height_ratios=[2, 2, 1.5])

        # 微分結果の表示
        for i in range(4):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, col])
            
            # データをNumPy配列に変換
            exact_data = self._to_numpy(exact[i])
            num_data = self._to_numpy(numerical[i])
            err_data = np.abs(num_data - exact_data)

            # 各成分のプロット
            ax.plot(x_np, exact_data, "b-", label="Exact")
            ax.plot(x_np, num_data, "r--", label="Numerical")
            
            # 誤差を薄い緑色で表示
            ax_err = ax.twinx()
            ax_err.semilogy(x_np, err_data, "g-", alpha=0.3, label="Error")
            ax_err.set_ylabel("Error (log scale)", color="g")
            ax_err.tick_params(axis="y", labelcolor="g")
            ax_err.grid(False)
            
            # 最大誤差をプロット上に表示
            max_err = errors[i]
            max_err_x = x_np[np.argmax(err_data)]
            max_err_y = err_data[np.argmax(err_data)]
            ax_err.plot(max_err_x, max_err_y, "go", ms=5)
            ax_err.annotate(f"Max: {max_err:.2e}", 
                           xy=(max_err_x, max_err_y),
                           xytext=(10, 10),
                           textcoords="offset points",
                           arrowprops=dict(arrowstyle="->", color="g", alpha=0.5))
            
            # タイトルとラベル
            ax.set_title(f"{titles[i]} (Max Error: {errors[i]:.2e})")
            ax.set_xlabel("x")
            ax.set_ylabel("Value")
            
            # 凡例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_err.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
            
            ax.grid(True)

        # 誤差サマリーのプロット (下段)
        ax_summary = fig.add_subplot(gs[2, :])
        
        x_pos = np.arange(len(titles))
        bars = ax_summary.bar(x_pos, errors)
        
        # 対数スケール（ゼロを小さな値に置き換え）
        error_values = errors.copy()
        for i, err in enumerate(error_values):
            if err == 0:
                error_values[i] = self.min_log_value
        
        ax_summary.set_yscale('log')
        ax_summary.set_title(f"Error Summary")
        ax_summary.set_xlabel('Component')
        ax_summary.set_ylabel('Maximum Error (log scale)')
        ax_summary.set_xticks(x_pos)
        ax_summary.set_xticklabels(titles)
        ax_summary.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # バーにラベルを追加
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax_summary.annotate(f'{err:.2e}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        # 全体のタイトル
        plt.suptitle(f"{function_name} Function Analysis ({n_points} points)", fontsize=16)
        plt.tight_layout()

        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, n_points, prefix)
            plt.savefig(filepath, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filepath

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
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]

        # グリッド間隔を計算（[-1, 1]の範囲を仮定）
        grid_spacings = [2.0 / (n - 1) for n in grid_sizes]

        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            if i < len(titles):
                # グリッドサイズごとの該当成分のエラーを取得
                errors = [results[n][i] for n in grid_sizes]
                
                # エラーが全て0の場合はグラフを描画しない
                if all(err == 0 for err in errors):
                    ax.text(0.5, 0.5, f"All errors are 0 for {title}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue
                    
                # エラーが0の場合は小さな値に置き換え（対数スケール用）
                plot_errors = [max(err, self.min_log_value) for err in errors]
                
                # 対数スケールで誤差をプロット
                ax.loglog(grid_spacings, plot_errors, 'o-', label=f"Error ({title})")
                
                # 基準となる傾きを示す参照線の追加
                x_range = [min(grid_spacings), max(grid_spacings)]
                
                # 2次、4次、6次の収束線を描画
                for order, style, color in zip([2, 4, 6], ['--', '-.', ':'], ['r', 'g', 'b']):
                    # 参照線の位置を最後のポイントに合わせる
                    ref_y0 = plot_errors[-1] * (x_range[-1] / x_range[-1]) ** order
                    y_ref = [ref_y0 * (x / x_range[-1]) ** order for x in x_range]
                    ax.loglog(x_range, y_ref, style, color=color, label=f"O(h^{order})")
                
                ax.set_title(f"{title} Error Convergence")
                ax.set_xlabel("Grid Spacing (h)")
                ax.set_ylabel("Error")
                ax.grid(True)
                ax.legend()

        plt.suptitle(f"Grid Convergence for {function_name}")
        plt.tight_layout()

        filepath = ""
        if save:
            if prefix:
                filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            else:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return True

    def get_dimension_label(self) -> str:
        """
        次元ラベルを返す
        
        Returns:
            "1D"
        """
        return "1D"
    
    def get_error_types(self) -> List[str]:
        """
        エラータイプのリストを返す
        
        Returns:
            1D用のエラータイプリスト
        """
        return ["ψ", "ψ'", "ψ''", "ψ'''"]