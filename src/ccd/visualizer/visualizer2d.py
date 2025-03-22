"""
高精度コンパクト差分法 (CCD) の2次元結果可視化

このモジュールは、2次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from typing import List, Optional, Tuple, Dict

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer2D(BaseVisualizer):
    """CCDソルバーの2D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_2d"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス
        """
        super().__init__(output_dir)
        # カラーマップ設定
        self.cmap_solution = 'viridis'
        self.cmap_error = 'hot'
        # コンポーネント名
        self.solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
    
    def generate_filename(self, func_name, nx_points, ny_points=None, prefix=""):
        """
        ファイル名を生成
        
        Args:
            func_name: 関数名
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数（Noneの場合はnx_pointsと同じ）
            prefix: 接頭辞
            
        Returns:
            生成されたファイルパス
        """
        if ny_points is None:
            ny_points = nx_points
            
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{nx_points}x{ny_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{nx_points}x{ny_points}_points.png"
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        2Dソリューションを可視化（統合コンパクトビュー）
        
        Args:
            grid: Grid2D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト [psi, psi_x, psi_y, psi_xx, psi_yy, psi_xxx, psi_yyy]
            exact: 厳密解のリスト
            errors: 誤差のリスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            保存に成功したかどうかのブール値
        """
        X, Y = grid.get_points()
        nx_points, ny_points = grid.nx_points, grid.ny_points
        
        # CuPy配列をNumPyに変換
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        
        # メインコンポーネントのみを可視化 (4つのプライマリコンポーネント)
        primary_indices = [0, 1, 3, 5]  # ψ, ψ_x, ψ_xx, ψ_xxx
        primary_names = [self.solution_names[i] for i in primary_indices]
        
        # 2x4のレイアウト (各コンポーネントに対して、数値解と誤差)
        fig, axes = plt.subplots(len(primary_indices), 2, figsize=(12, 4*len(primary_indices)))
        
        for i, comp_idx in enumerate(primary_indices):
            name = self.solution_names[comp_idx]
            num = numerical[comp_idx]
            ex = exact[comp_idx]
            err = errors[comp_idx]
            
            # NumPy配列に変換
            num_np = self._to_numpy(num)
            ex_np = self._to_numpy(ex)
            error_np = self._to_numpy(np.abs(num_np - ex_np))
            
            # 最小値・最大値を取得
            vmin = min(np.min(num_np), np.min(ex_np))
            vmax = max(np.max(num_np), np.max(ex_np))
            
            # 左側: 数値解
            im1 = axes[i, 0].contourf(X_np, Y_np, num_np, 50, cmap=self.cmap_solution, 
                                    vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"{name} (Numerical)")
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Y')
            cbar1 = plt.colorbar(im1, ax=axes[i, 0])
            
            # オーバーレイとして厳密解の等高線を追加
            cont = axes[i, 0].contour(X_np, Y_np, ex_np, 10, colors='black', linewidths=0.5, alpha=0.7)
            
            # 右側: 誤差
            # 誤差が非常に小さい場合はログスケールを使用
            if np.max(error_np) < 1e-5 and np.max(error_np) > 0:
                norm = LogNorm(vmin=max(np.min(error_np[error_np > 0]), 1e-15), vmax=np.max(error_np))
            else:
                norm = Normalize(vmin=0, vmax=np.max(error_np))
                
            im2 = axes[i, 1].contourf(X_np, Y_np, error_np, 50, cmap=self.cmap_error, norm=norm)
            axes[i, 1].set_title(f"{name} Error (Max: {err:.2e})")
            axes[i, 1].set_xlabel('X')
            axes[i, 1].set_ylabel('Y')
            cbar2 = plt.colorbar(im2, ax=axes[i, 1])
            
            # 誤差の特徴を強調するためのオーバーレイコンター
            if np.max(error_np) > 0:
                err_levels = np.logspace(np.log10(max(np.min(error_np[error_np > 0]), 1e-15)), 
                                      np.log10(np.max(error_np)), 5)
                axes[i, 1].contour(X_np, Y_np, error_np, levels=err_levels, 
                                 colors='black', linewidths=0.5, alpha=0.7)
        
        plt.suptitle(f"{function_name} Function Analysis ({nx_points}x{ny_points} points)")
        plt.tight_layout()
        
        # 保存ファイル名
        if save:
            filepath = self.generate_filename(function_name, nx_points, ny_points, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # 誤差サマリー図を作成
        self._create_error_summary(function_name, self.solution_names, errors, nx_points, ny_points, prefix, save, show, dpi)
        
        return True
    
    def _create_error_summary(self, function_name, solution_names, errors, nx_points, ny_points, prefix="", save=True, show=False, dpi=150):
        """
        誤差のサマリーグラフを作成
        
        Args:
            function_name: 関数名
            solution_names: 解成分の名前リスト
            errors: 誤差リスト
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数
            prefix: 接頭辞
            save: 保存するかどうか
            show: 表示するかどうか
            dpi: 画像のDPI値
            
        Returns:
            None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x_pos = np.arange(len(solution_names))
        
        # 1. 通常バープロット
        bars = ax1.bar(x_pos, errors, color='orange', alpha=0.7)
        
        # 対数スケール用の値設定（ゼロを小さな値に置き換え）
        log_errors = []
        for err in errors:
            if err == 0 or err < self.min_log_value:
                log_errors.append(self.min_log_value)
            else:
                log_errors.append(err)
                
        # バーに値ラベルを追加
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax1.annotate(f'{err:.2e}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3ポイント上にオフセット
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
        
        ax1.set_title(f"Component Errors (Linear Scale)")
        ax1.set_ylabel('Maximum Error')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(solution_names, rotation=45, ha="right")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 対数スケールバープロット
        bars2 = ax2.bar(x_pos, log_errors, color='red', alpha=0.7)
        ax2.set_yscale('log')
        
        # バーに値ラベルを追加
        for bar, err in zip(bars2, errors):
            y_pos = bar.get_height() * 1.1  # スケーリングして位置調整
            if y_pos <= 0:
                y_pos = self.min_log_value
            ax2.annotate(f'{err:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
        
        ax2.set_title(f"Component Errors (Log Scale)")
        ax2.set_ylabel('Maximum Error (Log Scale)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(solution_names, rotation=45, ha="right")
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # x方向成分とy方向成分を異なる色で区別
        for i, name in enumerate(solution_names):
            if '_x' in name:
                bars[i].set_color('blue')
                bars2[i].set_color('blue')
            elif '_y' in name:
                bars[i].set_color('green')
                bars2[i].set_color('green')
        
        plt.suptitle(f"{function_name} Function: Error Summary ({nx_points}x{ny_points} points)")
        plt.tight_layout()
        
        if save:
            summary_filepath = self.generate_filename(
                f"{function_name}_error_summary", 
                nx_points, 
                ny_points, 
                prefix
            )
            plt.savefig(summary_filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def compare_all_functions_errors(self, results_summary, grid_size, prefix="", dpi=150, show=False):
        """
        すべてのテスト関数の誤差を比較するグラフを生成
        
        Args:
            results_summary: 関数名をキーとし、誤差リストを値とする辞書
            grid_size: グリッドサイズ
            prefix: 接頭辞
            dpi: 画像のDPI値
            show: 表示するかどうか
            
        Returns:
            出力ファイルパス
        """
        func_names = list(results_summary.keys())
        
        # 重要なコンポーネントのみを表示
        important_components = [0, 1, 2, 3, 4]  # ψ, ψ_x, ψ_y, ψ_xx, ψ_yy
        error_types = [self.solution_names[i] for i in important_components]
        
        # より効果的なレイアウト (3行2列 + ヒートマップ)
        fig = plt.figure(figsize=(16, 12))
        grid = plt.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1-5: 個別のバープロット
        bar_axes = []
        for i, comp_idx in enumerate(important_components):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(grid[row, col])
            bar_axes.append(ax)
            
            # 対数スケール用に0を適切に処理
            error_values = []
            for name in func_names:
                err = results_summary[name][comp_idx]
                if err == 0.0 or err < self.min_log_value:
                    error_values.append(self.min_log_value)
                else:
                    error_values.append(err)
            
            x_positions = np.arange(len(func_names))
            bars = ax.bar(x_positions, error_values)
            ax.set_yscale("log")
            ax.set_title(f"{error_types[comp_idx]} Error")
            ax.set_xlabel("Test Function")
            ax.set_ylabel("Error (log scale)")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(func_names, rotation=45, ha="right", fontsize=8)
            
            # 値ラベル
            for j, (bar, name) in enumerate(zip(bars, func_names)):
                orig_err = results_summary[name][comp_idx]
                label_text = "0.0" if orig_err == 0.0 else f"{orig_err:.2e}"
                y_pos = bar.get_height() * 1.1
                ax.annotate(
                    label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        
        # 6: ヒートマップ (全コンポーネント・全関数の概観)
        ax_heat = fig.add_subplot(grid[2, :])
        
        # ヒートマップデータ作成
        heat_data = np.zeros((len(func_names), len(self.solution_names)))
        for i, name in enumerate(func_names):
            for j, _ in enumerate(self.solution_names):
                heat_data[i, j] = results_summary[name][j]
        
        # ゼロを適切に処理
        heat_data_log = np.copy(heat_data)
        heat_data_log[heat_data_log <= 0] = self.min_log_value
        
        # ヒートマップ描画
        im = ax_heat.imshow(heat_data_log, cmap='viridis', aspect='auto', 
                          norm=LogNorm(vmin=np.min(heat_data_log), vmax=np.max(heat_data_log)))
        
        # 軸ラベル
        ax_heat.set_yticks(np.arange(len(func_names)))
        ax_heat.set_xticks(np.arange(len(self.solution_names)))
        ax_heat.set_yticklabels(func_names)
        ax_heat.set_xticklabels(self.solution_names)
        ax_heat.set_title("Error Heatmap (Log Scale)")
        
        # 値ラベルをヒートマップに表示
        for i in range(len(func_names)):
            for j in range(len(self.solution_names)):
                val = heat_data[i, j]
                text = ax_heat.text(j, i, f"{val:.2e}",
                                  ha="center", va="center", color="w" if val > 1e-8 else "black",
                                  fontsize=7)
        
        # カラーバー
        cbar = fig.colorbar(im, ax=ax_heat)
        cbar.set_label('Error (Log Scale)')
        
        plt.suptitle(f"Error Comparison for All Functions ({grid_size}x{grid_size} points)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # タイトル用に上部に余白を設定
        
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison_{grid_size}x{grid_size}.png"
        if not prefix:
            filename = f"{self.output_dir}/all_functions_comparison_{grid_size}x{grid_size}.png"
            
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        
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
        # 2x4レイアウト (7コンポーネント + サマリー)
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        # 重要なコンポーネント
        important_indices = [0, 1, 2, 3, 4, 5, 6]  # すべてのコンポーネント
        component_names = [self.solution_names[i] for i in important_indices]
        
        # グリッド間隔計算
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        # 収束次数の色とスタイル
        order_colors = ['r', 'g', 'b', 'purple']
        order_styles = ['--', '-.', ':', '-']
        order_values = [2, 4, 6, 8]
        
        # 各コンポーネントの収束プロット
        for i, comp_idx in enumerate(important_indices):
            if i < len(axes) - 1:  # 最後のセルはサマリー用に残す
                ax = axes[i]
                name = component_names[i]
                
                # 格子サイズごとの誤差を収集
                errors = [results[n][comp_idx] for n in grid_sizes]
                
                # ゼロ値の処理
                plot_errors = []
                for err in errors:
                    if err == 0 or err < self.min_log_value:
                        plot_errors.append(self.min_log_value)
                    else:
                        plot_errors.append(err)
                
                # 収束曲線プロット
                ax.loglog(grid_spacings, plot_errors, 'o-', color='black', linewidth=2, 
                        markersize=8, label=name)
                
                # 収束次数の参照線
                if min(plot_errors) > 0:
                    x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                    for j, order in enumerate(order_values):
                        # 最後のデータポイントに合わせて参照線を配置
                        if len(plot_errors) > 1:
                            # 実際の収束次数を推定
                            est_order = np.log(plot_errors[-2]/plot_errors[-1]) / np.log(grid_spacings[-2]/grid_spacings[-1])
                            order_label = f'O(h^{order:.1f})' if j == 0 else f'O(h^{order})'
                            
                            scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                            y_ref = scale * x_ref ** order
                            ax.loglog(x_ref, y_ref, order_styles[j], color=order_colors[j], 
                                    linewidth=1.5, label=order_label if j == 0 else f'O(h^{order})')
                
                # 実際の収束次数の計算と表示
                if len(grid_spacings) >= 2 and all(e > 0 for e in plot_errors):
                    orders = []
                    for k in range(1, len(grid_spacings)):
                        h_ratio = grid_spacings[k-1] / grid_spacings[k]
                        err_ratio = plot_errors[k-1] / plot_errors[k]
                        if err_ratio > 1.0:  # 収束している場合のみ
                            order = np.log(err_ratio) / np.log(h_ratio)
                            orders.append(order)
                    
                    if orders:
                        avg_order = np.mean(orders)
                        ax.text(0.05, 0.05, f"Est. Order: {avg_order:.2f}", 
                              transform=ax.transAxes, fontsize=9,
                              bbox=dict(facecolor='white', alpha=0.7))
                
                ax.set_title(f"{name} Convergence")
                ax.set_xlabel('Grid Spacing (h)')
                ax.set_ylabel('Error')
                ax.grid(True, which='both')
                ax.legend(fontsize=8, loc='upper left')
        
        # サマリープロット (すべての成分を一つのグラフに)
        ax_summary = axes[-1]
        
        # 統合プロット用のマーカーとカラー
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
        colors = plt.cm.tab10(np.linspace(0, 1, len(component_names)))
        
        # 各成分の誤差曲線をプロット
        for i, comp_idx in enumerate(important_indices):
            name = component_names[i]
            errors = [results[n][comp_idx] for n in grid_sizes]
            
            # ゼロ値の処理
            plot_errors = []
            for err in errors:
                if err == 0 or err < self.min_log_value:
                    plot_errors.append(self.min_log_value)
                else:
                    plot_errors.append(err)
            
            ax_summary.loglog(grid_spacings, plot_errors, marker=markers[i % len(markers)],
                           color=colors[i], linestyle='-', linewidth=1.5, 
                           markersize=6, label=name)
        
        # 参照線 (2次と4次のみ表示して簡素化)
        x_ref = np.array([min(grid_spacings), max(grid_spacings)])
        # 適切な基準点を見つける (中央値)
        mid_error = np.median([e for e in plot_errors if e > self.min_log_value])
        mid_spacing = np.median(grid_spacings)
        
        for order, style, color in zip([2, 4], ['--', '-.'], ['gray', 'black']):
            scale = mid_error / (mid_spacing ** order)
            y_ref = scale * x_ref ** order
            ax_summary.loglog(x_ref, y_ref, style, color=color, 
                           linewidth=1.0, label=f'O(h^{order})')
        
        ax_summary.set_title("All Components Convergence")
        ax_summary.set_xlabel('Grid Spacing (h)')
        ax_summary.set_ylabel('Error')
        ax_summary.grid(True, which='both')
        ax_summary.legend(fontsize=7, loc='upper left', ncol=2)
        
        plt.suptitle(f"Grid Convergence Analysis for {function_name} Function")
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def get_dimension_label(self) -> str:
        """
        次元ラベルを返す
        
        Returns:
            "2D"
        """
        return "2D"
    
    def get_error_types(self) -> List[str]:
        """
        エラータイプのリストを返す
        
        Returns:
            2D用のエラータイプリスト
        """
        return self.solution_names