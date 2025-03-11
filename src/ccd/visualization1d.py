import os
import numpy as np
import matplotlib.pyplot as plt

class CCDVisualizer:
    """CCDソルバーの結果を可視化するクラス"""

    def __init__(self, output_dir="results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス（デフォルト: "results"）
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.min_log_value = 1e-16

    def generate_filename(self, func_name, n_points, prefix=""):
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{n_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{n_points}_points.png"

    def visualize_derivatives(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """導関数の結果を可視化"""
        x = grid.get_points()
        n_points = grid.n_points

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]

        x_np = x.get() if hasattr(x, "get") else x

        for i, ax in enumerate(axes.flat):
            exact_data = exact[i].get() if hasattr(exact[i], "get") else exact[i]
            num_data = numerical[i].get() if hasattr(numerical[i], "get") else numerical[i]

            ax.plot(x_np, exact_data, "b-", label="Exact")
            ax.plot(x_np, num_data, "r--", label="Numerical")
            ax.set_title(f"{titles[i]} (error: {errors[i]:.2e})")
            ax.legend()
            ax.grid(True)

        plt.suptitle(f"Results for {function_name} function ({n_points} points)")
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

    def compare_all_functions_errors(self, results_summary, prefix="", dpi=150, show=False):
        """全テスト関数の誤差比較グラフを生成"""
        func_names = list(results_summary.keys())

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        error_types = ["ψ", "ψ'", "ψ''", "ψ'''"]

        for i, (ax, error_type) in enumerate(zip(axes.flat, error_types)):
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

        plt.tight_layout()
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison.png" if prefix else f"{self.output_dir}/all_functions_comparison.png"
        plt.savefig(filename, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filename

    def visualize_grid_convergence(self, function_name, grid_sizes, results, prefix="", save=True, show=False, dpi=150):
        """グリッド収束性のグラフを生成"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]

        grid_spacings = []
        for n in grid_sizes:
            h = 2.0 / (n - 1)  # Assuming domain [-1, 1]
            grid_spacings.append(h)

        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            errors = [results[n][i] for n in grid_sizes]
            
            # エラーが全て0の場合はグラフを描画しない
            if all(err == 0 for err in errors):
                ax.text(0.5, 0.5, f"All errors are 0 for {title}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                continue
                
            # エラーが0の場合は小さな値に置き換え
            plot_errors = [max(err, self.min_log_value) for err in errors]
            
            ax.loglog(grid_spacings, plot_errors, 'o-', label=f"Error ({title})")
            
            # 参照傾き線を描画
            x_range = [min(grid_spacings), max(grid_spacings)]
            for order, style, color in zip([2, 4, 6], ['--', '-.', ':'], ['r', 'g', 'b']):
                # 参照線の位置を調整（最後のポイントに合わせる）
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
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png" if prefix else f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filepath