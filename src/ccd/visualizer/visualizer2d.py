"""
高精度コンパクト差分法 (CCD) の2次元結果可視化

このモジュールは、2次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。可視化結果を統合し、一つのファイルに出力します。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

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
        2Dソリューションを可視化（統合版）
        
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
        
        # 可視化するソリューションのリスト
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
        
        # 統合されたグラフ作成 - GridSpecを利用
        fig = plt.figure(figsize=(20, 16))
        
        # GridSpecでレイアウト定義: 主要成分表示(2行3列) + 誤差サマリー(1行)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3, height_ratios=[3, 3, 2])
        
        # カラーマップ
        cmap_sol = 'viridis'
        cmap_err = 'hot'
        
        # 主要な成分のみ表示（最大6つ）
        display_components = min(6, len(solution_names))
        
        # 各成分の可視化
        for i in range(display_components):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            
            name = solution_names[i]
            num = numerical[i]
            ex = exact[i]
            err = errors[i]
            
            # NumPy配列に変換
            num_np = self._to_numpy(num)
            
            # 数値解のコンターマップ
            im = ax.contourf(X_np, Y_np, num_np, 50, cmap=cmap_sol)
            ax.set_title(f"{name} (Error: {err:.2e})")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
            
            # コンターラインを追加
            contour = ax.contour(X_np, Y_np, num_np, 8, colors='k', linewidths=0.5, alpha=0.7)
            
        # 誤差分布の可視化 (最初の成分、ψについて）
        ax_err = fig.add_subplot(gs[2, 0])
        
        # ψ成分の誤差
        num_np = self._to_numpy(numerical[0])
        ex_np = self._to_numpy(exact[0])
        error_np = np.abs(num_np - ex_np)
        
        # 対数スケールの誤差マップ
        err_min = np.max([np.min(error_np[error_np > 0]), 1e-15])
        from matplotlib.colors import LogNorm
        im_err = ax_err.pcolormesh(X_np, Y_np, error_np, cmap=cmap_err, 
                                 norm=LogNorm(vmin=err_min, vmax=error_np.max()), 
                                 shading='auto')
        
        ax_err.set_title(f"ψ Error Distribution")
        ax_err.set_xlabel('X')
        ax_err.set_ylabel('Y')
        plt.colorbar(im_err, ax=ax_err)
        
        # 誤差サマリーグラフ
        ax_summary = fig.add_subplot(gs[2, 1:])
        
        x_pos = np.arange(len(solution_names))
        bars = ax_summary.bar(x_pos, errors)
        
        # 対数スケール（ゼロを小さな値に置き換え）
        error_values = errors.copy()
        for i, err in enumerate(error_values):
            if err == 0:
                error_values[i] = self.min_log_value
        
        ax_summary.set_yscale('log')
        ax_summary.set_title(f"Error Summary")
        ax_summary.set_xlabel('Solution Component')
        ax_summary.set_ylabel('Maximum Error (log scale)')
        ax_summary.set_xticks(x_pos)
        ax_summary.set_xticklabels(solution_names, rotation=45, ha="right")
        ax_summary.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # バーにラベルを追加
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax_summary.annotate(f'{err:.2e}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        # 全体タイトル
        plt.suptitle(f"{function_name} Function Analysis ({nx_points}x{ny_points} points)", fontsize=16)
        plt.tight_layout()
        
        # 保存
        if save:
            filepath = self.generate_filename(function_name, nx_points, ny_points, prefix)
            plt.savefig(filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
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
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        error_types = ["ψ", "ψ_x", "ψ_y", "ψ''_x", "ψ''_y", "ψ'''_x", "ψ'''_y"]
        
        for i, (ax, error_type) in enumerate(zip(axes.flat, error_types)):
            if i < len(error_types):
                original_errors = [results_summary[name][i] for name in func_names]
                
                # 対数スケール用に0を小さな値に置き換え
                errors = []
                for err in original_errors:
                    if err == 0.0:
                        errors.append(self.min_log_value)
                    else:
                        errors.append(err)
                
                x_positions = np.arange(len(func_names))
                bars = ax.bar(x_positions, errors)
                ax.set_yscale("log")
                ax.set_title(f"{error_type} Error Comparison")
                ax.set_xlabel("Test Function")
                ax.set_ylabel("Error (log scale)")
                ax.grid(True, which="both", linestyle="--", alpha=0.5)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(func_names, rotation=45, ha="right")
                
                # 値をバーの上に表示
                for j, (bar, orig_err) in enumerate(zip(bars, original_errors)):
                    height = bar.get_height()
                    label_text = "0.0" if orig_err == 0.0 else f"{orig_err:.2e}"
                    y_pos = height * 1.1
                    ax.annotate(
                        label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        
        plt.suptitle(f"Error Comparison for All Functions ({grid_size}x{grid_size} points)")
        plt.tight_layout()
        
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison_{grid_size}x{grid_size}.png"
        if not prefix:
            filename = f"{self.output_dir}/all_functions_comparison_{grid_size}x{grid_size}.png"
            
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
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
        
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        for i, (ax, name) in enumerate(zip(axes.flat, solution_names)):
            if i < len(solution_names):
                errors = [results[n][i] for n in grid_sizes]
                
                # 対数-対数プロット
                ax.loglog(grid_spacings, errors, 'o-', label=name)
                
                # 傾きの参照線
                if min(errors) > 0:  # すべてのエラーが0より大きい場合のみ
                    x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                    # 2次、4次、6次の参照線
                    for order, style in zip([2, 4, 6], ['--', '-.', ':']):
                        scale = errors[-1] / (grid_spacings[-1] ** order)
                        y_ref = scale * x_ref ** order
                        ax.loglog(x_ref, y_ref, style, label=f'O(h^{order})')
                
                ax.set_title(f"{name} Error Convergence")
                ax.set_xlabel('Grid Spacing h')
                ax.set_ylabel('Maximum Error')
                ax.grid(True, which='both')
                ax.legend()
        
        plt.suptitle(f"Grid Convergence for {function_name} Function")
        plt.tight_layout()
        
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
        return ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]