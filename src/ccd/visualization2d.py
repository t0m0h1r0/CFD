import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class CCD2DVisualizer:
    """CCDソルバーの2D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_2d"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.min_log_value = 1e-16
    
    def generate_filename(self, func_name, nx_points, ny_points, prefix=""):
        """ファイル名を生成"""
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{nx_points}x{ny_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{nx_points}x{ny_points}_points.png"
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        2Dソリューションを可視化（平面図として、すべての解を1ファイルにまとめる）
        
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
        """
        X, Y = grid.get_points()
        nx_points, ny_points = grid.nx_points, grid.ny_points
        
        # CuPy配列をNumPyに変換する関数
        def to_numpy(x):
            return x.get() if hasattr(x, "get") else x
        
        X_np = to_numpy(X)
        Y_np = to_numpy(Y)
        
        # 可視化するソリューションのリスト
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
        
        # すべての解を1つの図にまとめる
        fig, axes = plt.subplots(len(solution_names), 3, figsize=(15, 4*len(solution_names)))
        
        # カラーマップ
        cmap_sol = 'viridis'
        cmap_err = 'hot'
        
        for i, (name, num, ex, err) in enumerate(zip(solution_names, numerical, exact, errors)):
            num_np = to_numpy(num)
            ex_np = to_numpy(ex)
            error_np = to_numpy(np.abs(num - ex))
            
            # 同じカラーバーの範囲を使用するため、最小値と最大値を計算
            vmin = min(np.min(num_np), np.min(ex_np))
            vmax = max(np.max(num_np), np.max(ex_np))
            
            # Numerical solution
            im1 = axes[i, 0].contourf(X_np, Y_np, num_np, 50, cmap=cmap_sol, vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"{name} (Numerical)")
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Exact solution
            im2 = axes[i, 1].contourf(X_np, Y_np, ex_np, 50, cmap=cmap_sol, vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f"{name} (Exact)")
            axes[i, 1].set_xlabel('X')
            axes[i, 1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Error
            im3 = axes[i, 2].contourf(X_np, Y_np, error_np, 50, cmap=cmap_err)
            axes[i, 2].set_title(f"{name} Error (Max: {err:.2e})")
            axes[i, 2].set_xlabel('X')
            axes[i, 2].set_ylabel('Y')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.suptitle(f"{function_name} Function Analysis ({nx_points}x{ny_points} points)")
        plt.tight_layout()
        
        # 保存ファイル名
        if save:
            filepath = self.generate_filename(function_name, nx_points, ny_points, prefix)
            plt.savefig(filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # 誤差サマリー図を作成
        self._create_error_summary(function_name, solution_names, errors, nx_points, ny_points, prefix, save, show, dpi)
        
        return True
    
    def _create_error_summary(self, function_name, solution_names, errors, nx_points, ny_points, prefix="", save=True, show=False, dpi=150):
        """誤差のサマリーグラフを作成"""
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(solution_names))
        bars = ax.bar(x_pos, errors)
        
        # 対数スケール（ゼロを小さな値に置き換え）
        error_values = errors.copy()
        for i, err in enumerate(error_values):
            if err == 0:
                error_values[i] = self.min_log_value
        
        ax.set_yscale('log')
        ax.set_title(f"{function_name} Function: Error Summary ({nx_points}x{ny_points} points)")
        ax.set_xlabel('Solution Component')
        ax.set_ylabel('Maximum Error (log scale)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(solution_names)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # バーにラベルを追加
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax.annotate(f'{err:.2e}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            summary_filepath = self.generate_filename(
                f"{function_name}_summary", 
                nx_points, 
                ny_points, 
                prefix
            )
            plt.savefig(summary_filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def compare_all_functions_errors(self, results_summary, grid_size, prefix="", dpi=150, show=False):
        """すべてのテスト関数の誤差を比較するグラフを生成"""
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
        """グリッド収束性のグラフを生成"""
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