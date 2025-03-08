import os
import numpy as np
import matplotlib.pyplot as plt

class CCDVisualizer:
    """CCDソルバーの結果を可視化するクラス"""

    def __init__(self, output_dir="results"):
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
        plt.savefig(filename)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filename