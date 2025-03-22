"""
高精度コンパクト差分法 (CCD) の1次元結果可視化

このモジュールは、1次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict

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
        # 図のカラーテーマ設定
        self.color_exact = 'blue'
        self.color_numerical = 'red'
        self.color_error = 'orange'
        self.linestyle_exact = '-'
        self.linestyle_numerical = '--'
        self.linestyle_error = '-.'

    def visualize_derivatives(self, grid, function_name, numerical, exact, errors, 
                            prefix="", save=True, show=False, dpi=150):
        """
        導関数の結果を可視化 - 解と誤差を同一図内に統合
        
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

        # 2×2のサブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for i, ax in enumerate(axes.flat):
            if i < len(numerical):
                # データをNumPy配列に変換
                exact_data = self._to_numpy(exact[i])
                num_data = self._to_numpy(numerical[i])
                error_data = np.abs(num_data - exact_data)
                
                # 左Y軸 - 解プロット
                l1 = ax.plot(x_np, exact_data, self.color_exact, 
                          linestyle=self.linestyle_exact, label="Exact", linewidth=2)
                l2 = ax.plot(x_np, num_data, self.color_numerical, 
                          linestyle=self.linestyle_numerical, label="Numerical", linewidth=2)
                
                # 共通Y軸の範囲設定
                min_val = min(np.min(exact_data), np.min(num_data))
                max_val = max(np.max(exact_data), np.max(num_data))
                pad = 0.1 * (max_val - min_val)
                if abs(pad) < 1e-10:  # 値の範囲が極めて小さい場合
                    pad = 0.1
                ax.set_ylim(min_val - pad, max_val + pad)
                
                # 右Y軸 - 誤差プロット
                ax2 = ax.twinx()
                l3 = ax2.plot(x_np, error_data, self.color_error, 
                           linestyle=self.linestyle_error, label="Error", linewidth=1.5)
                
                # 誤差軸のスケール調整 (自動対数スケール)
                max_error = np.max(error_data)
                if max_error > 0 and max_error < 1e-5:
                    ax2.set_yscale('log')
                    # 下限を適切に設定して空白を減らす
                    ax2.set_ylim(max_error * 1e-3, max_error * 10)
                else:
                    # 線形スケールの場合は0から開始
                    ax2.set_ylim(0, max_error * 1.2)
                
                # 凡例の結合
                lines = l1 + l2 + l3
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper right')
                
                # タイトルに誤差情報を含める
                ax.set_title(f"{titles[i]} (Max Error: {errors[i]:.2e})")
                ax.set_xlabel("x")
                ax.set_ylabel("Value")
                ax2.set_ylabel("Error")
                ax.grid(True, alpha=0.3)

        plt.suptitle(f"{function_name} Function Analysis ({n_points} points)")
        plt.tight_layout()

        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, n_points, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)
            
        # 追加のサマリー図を生成
        self._create_error_summary(function_name, titles, errors, n_points, prefix, save, show, dpi)

        return filepath
    
    def _create_error_summary(self, function_name, titles, errors, n_points, prefix="", save=True, show=False, dpi=150):
        """
        誤差サマリーグラフを生成
        
        Args:
            function_name: 関数名
            titles: 成分タイトルリスト
            errors: 誤差リスト
            n_points: 格子点数
            prefix: 接頭辞
            save: 保存するかどうか
            show: 表示するかどうか
            dpi: 画像のDPI値
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        component_names = [t.replace('$', '') for t in titles]
        x_pos = np.arange(len(component_names))
        
        # 棒グラフでエラーを表示
        bars = ax.bar(x_pos, errors, color=self.color_error, alpha=0.7)
        
        # 対数スケール (ゼロ対応)
        if all(e > 0 for e in errors):
            ax.set_yscale('log')
        
        # タイトルと軸ラベル
        ax.set_title(f"{function_name} Function: Error Summary ({n_points} points)")
        ax.set_xlabel('Component')
        ax.set_ylabel('Maximum Error')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(component_names)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # バー上に値を表示
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax.annotate(f'{err:.2e}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3ポイント上にオフセット
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            filepath = self.generate_filename(f"{function_name}_error_summary", n_points, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)

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
        
        # 収束次数の色とスタイル
        order_colors = ['r', 'g', 'b', 'purple']
        order_styles = ['--', '-.', ':', '-']
        order_labels = ['O(h²)', 'O(h⁴)', 'O(h⁶)', 'O(h⁸)']
        
        # すべての成分の誤差値範囲を把握して共通スケールを使用
        all_errors = []
        for n in grid_sizes:
            all_errors.extend(results[n])
        valid_errors = [e for e in all_errors if e > 0]
        
        if valid_errors:
            y_min = min(valid_errors) * 0.1
            y_max = max(valid_errors) * 10
        else:
            y_min, y_max = 1e-16, 1.0
        
        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            if i < len(titles):
                # グリッドサイズごとの該当成分のエラーを取得
                errors = [results[n][i] for n in grid_sizes]
                
                # エラーが全て0の場合はメッセージを表示
                if all(err == 0 for err in errors):
                    ax.text(0.5, 0.5, f"All errors are 0 for {title}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue
                    
                # エラーが0の場合は小さな値に置き換え（対数スケール用）
                plot_errors = [max(err, self.min_log_value) for err in errors]
                
                # 対数スケールで誤差をプロット
                ax.loglog(grid_spacings, plot_errors, 'o-', color='black', linewidth=2, 
                        markersize=8, label=f"Error ({title})")
                
                # 収束次数の参照線
                for j, order in enumerate([2, 4, 6, 8]):
                    if min(plot_errors) > 0:  # エラーが正の場合のみ
                        x_range = [min(grid_spacings), max(grid_spacings)]
                        # 最後のデータポイントに合わせて参照線を配置
                        ref_y0 = plot_errors[-1] * (x_range[-1] / x_range[-1]) ** order
                        y_ref = [ref_y0 * (x / x_range[-1]) ** order for x in x_range]
                        ax.loglog(x_range, y_ref, order_styles[j], color=order_colors[j], 
                                linewidth=1.5, label=order_labels[j])
                
                # 軸の設定
                ax.set_title(f"{title} Error Convergence")
                ax.set_xlabel("Grid Spacing (h)")
                ax.set_ylabel("Error")
                ax.set_ylim(y_min, y_max)  # 共通のY軸範囲を設定
                ax.grid(True, which='both', alpha=0.3)
                ax.legend(fontsize=9)

        plt.suptitle(f"Grid Convergence Analysis for {function_name}")
        plt.tight_layout()

        filepath = ""
        if save:
            if prefix:
                filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            else:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return True

    def generate_filename(self, func_name, n_points, prefix=""):
        """
        ファイル名を生成
        
        Args:
            func_name: 関数名
            n_points: 格子点数
            prefix: 接頭辞
            
        Returns:
            生成されたファイルパス
        """
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{n_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{n_points}_points.png"

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