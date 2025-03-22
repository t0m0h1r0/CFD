"""
高精度コンパクト差分法 (CCD) の1次元結果可視化

このモジュールは、1次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。数値解と誤差分布を効率的に一つのダッシュボードに表示します。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer1D(BaseVisualizer):
    """CCDソルバーの1D結果を可視化するクラス"""

    def __init__(self, output_dir="results"):
        """初期化"""
        super().__init__(output_dir)

    def visualize_derivatives(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        導関数の結果を可視化 (統合ダッシュボード形式)
        
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
        # グリッドデータ
        x_np = self._to_numpy(grid.get_points())
        n_points = grid.n_points
        
        # 成分名
        component_names = self.get_error_types()
        
        # ダッシュボード作成 (3行2列)
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 0.7])
        
        # 各導関数の可視化
        for i in range(4):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, col])
            
            # データ変換
            exact_data = self._to_numpy(exact[i])
            num_data = self._to_numpy(numerical[i])
            
            # 解と誤差を同時に可視化
            self._plot_solution_with_error(ax, x_np, exact_data, num_data, component_names[i], errors[i])
        
        # 誤差サマリーの作成
        ax_summary = fig.add_subplot(gs[2, :])
        self._plot_error_summary(ax_summary, component_names, errors)
        
        # 全体のタイトル
        plt.suptitle(f"{function_name} Function Analysis ({n_points} points)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # タイトル用の余白確保

        # 保存と表示
        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, n_points, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filepath
    
    def _plot_solution_with_error(self, ax, x, exact, numerical, title, max_error):
        """
        解と誤差を単一のグラフに効率的に可視化
        
        Args:
            ax: プロット用のAxis
            x: x座標値
            exact: 厳密解データ
            numerical: 数値解データ
            title: プロットタイトル
            max_error: 最大誤差値
        """
        # 左側のY軸: 解の値
        ax.plot(x, exact, "b-", label="Exact", linewidth=1.5)
        ax.plot(x, numerical, "r--", label="Numerical", linewidth=1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("Value")
        
        # 右側のY軸: 誤差 (対数スケール)
        ax2 = ax.twinx()
        error = np.abs(numerical - exact)
        
        # エラーが0の場合の処理
        min_error = max(np.min(error[error > 0]) if np.any(error > 0) else 1e-15, 1e-15)
        plot_error = np.maximum(error, min_error)
        
        # 誤差プロット (塗りつぶし付き)
        ax2.semilogy(x, plot_error, "g-", alpha=0.3, label="Error")
        ax2.fill_between(x, min_error, plot_error, color='green', alpha=0.1)
        ax2.set_ylabel("Error (log)", color="g")
        ax2.tick_params(axis="y", labelcolor="g")
        
        # 最大誤差点をマーク
        max_err_idx = np.argmax(error)
        max_err_x = x[max_err_idx]
        max_err_y = error[max_err_idx]
        
        # 最大誤差値が0でない場合のみマーカー表示
        if max_err_y > 0:
            ax2.plot(max_err_x, max_err_y, "go", ms=4)
            ax2.annotate(f"Max", 
                      xy=(max_err_x, max_err_y),
                      xytext=(5, 5),
                      textcoords="offset points",
                      fontsize=8,
                      color="g")
        
        # タイトル (最大誤差含む)
        ax.set_title(f"{title} (Error: {max_error:.2e})")
        
        # 凡例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
        
        # グリッド
        ax.grid(True, alpha=0.3)
    
    def _plot_error_summary(self, ax, component_names, errors):
        """
        誤差サマリーグラフを描画
        
        Args:
            ax: プロット用のAxis
            component_names: 成分名リスト
            errors: 各成分の誤差値
        """
        x_pos = np.arange(len(component_names))
        
        # 0値を処理
        plot_errors = np.array(errors).copy()
        plot_errors[plot_errors == 0] = self.min_log_value
        
        # バープロット
        bars = ax.bar(x_pos, plot_errors)
        ax.set_yscale('log')
        ax.set_title("Error Summary")
        ax.set_xlabel('Component')
        ax.set_ylabel('Maximum Error (log scale)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(component_names)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # 値のアノテーション
        for i, (bar, err) in enumerate(zip(bars, errors)):
            height = max(err, self.min_log_value)
            ax.annotate(
                f'{err:.2e}',
                xy=(i, height * 1.1),
                ha='center', va='bottom',
                fontsize=10
            )

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
        component_names = self.get_error_types()
        
        # 2x2のレイアウト
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        # グリッド間隔計算
        grid_spacings = [2.0 / (n - 1) for n in grid_sizes]

        for i, (ax, name) in enumerate(zip(axes, component_names)):
            # 各グリッドサイズでの誤差
            errors = [results[n][i] for n in grid_sizes]
            
            # 全て0の場合はスキップ
            if all(err == 0 for err in errors):
                ax.text(0.5, 0.5, f"All errors are 0", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
                
            # 対数プロットのための前処理
            plot_errors = [max(err, self.min_log_value) for err in errors]
            
            # エラープロット
            ax.loglog(grid_spacings, plot_errors, 'o-', label=name, linewidth=1.5)
            
            # 収束次数の参照線
            if min(plot_errors) > 0:
                x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                
                # 重要な2次と4次の収束線のみ表示
                for order, style, color in zip([2, 4], ['--', '-.'], ['r', 'g']):
                    # 最後の点に合わせて参照線をスケーリング
                    scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                    y_ref = scale * x_ref ** order
                    ax.loglog(x_ref, y_ref, style, color=color, label=f"O(h^{order})")
            
            ax.set_title(f"{name} Convergence")
            ax.set_xlabel("Grid Spacing (h)")
            ax.set_ylabel("Error")
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=9)

        # 全体タイトル
        plt.suptitle(f"Grid Convergence: {function_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存と表示
        filepath = ""
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
        return "1D"
    
    def get_error_types(self) -> List[str]:
        """エラータイプのリストを返す"""
        return ["ψ", "ψ'", "ψ''", "ψ'''"]