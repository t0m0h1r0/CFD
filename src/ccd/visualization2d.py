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
        2Dソリューションを可視化
        
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
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ''_x", "ψ''_y", "ψ'''_x", "ψ'''_y"]
        
        # マルチページの図を作成
        num_solutions = len(solution_names)
        
        for sol_idx, (name, num, ex, err) in enumerate(zip(solution_names, numerical, exact, errors)):
            fig = plt.figure(figsize=(15, 5))
            
            # 数値解
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            num_np = to_numpy(num)
            surf = ax1.plot_surface(X_np, Y_np, num_np, cmap=cm.viridis, alpha=0.8)
            ax1.set_title(f"{name} (Numerical Solution)")
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            
            # 厳密解
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ex_np = to_numpy(ex)
            surf = ax2.plot_surface(X_np, Y_np, ex_np, cmap=cm.viridis, alpha=0.8)
            ax2.set_title(f"{name} (Exact Solution)")
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            
            # 誤差
            ax3 = fig.add_subplot(1, 3, 3)
            error_np = to_numpy(np.abs(num - ex))
            im = ax3.pcolormesh(X_np, Y_np, error_np, cmap='hot', shading='auto')
            fig.colorbar(im, ax=ax3)
            ax3.set_title(f"{name} Error (Max: {err:.2e})")
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            
            plt.suptitle(f"{function_name} Function: {name} ({nx_points}x{ny_points} points)")
            plt.tight_layout()
            
            # 保存ファイル名
            if save:
                # 各ソリューションタイプ用に個別のファイル名を生成
                solution_type = name.replace("ψ", "psi").replace("'", "p").replace("''", "pp").replace("'''", "ppp")
                filepath = self.generate_filename(
                    f"{function_name}_{solution_type}", 
                    nx_points, 
                    ny_points, 
                    prefix
                )
                plt.savefig(filepath, dpi=dpi)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        # 誤差サマリー図を作成
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(solution_names))
        bars = ax.bar(x_pos, errors)
        
        # 対数スケール（ゼロを小さな値に置き換え）
        for i, err in enumerate(errors):
            if err == 0:
                errors[i] = self.min_log_value
        
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
        
        return True
    
    def compare_all_functions_errors(self, results_summary, grid_size, prefix="", dpi=150, show=False):
        """
        すべてのテスト関数の誤差を比較するグラフを生成
        
        Args:
            results_summary: 各関数の誤差結果を持つ辞書
            grid_size: グリッドサイズ
            prefix: ファイル名の接頭辞
            dpi: 保存する図のDPI
            show: 図を表示するかどうか
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
            function_name: テスト関数の名前
            grid_sizes: テストしたグリッドサイズのリスト
            results: 各グリッドサイズでの誤差結果の辞書
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
        """
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ''_x", "ψ''_y", "ψ'''_x", "ψ'''_y"]
        
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