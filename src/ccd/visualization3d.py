import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class CCD3DVisualizer:
    """CCDソルバーの3D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_3d"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス（デフォルト: "results_3d"）
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.min_log_value = 1e-16
    
    def generate_filename(self, func_name, nx_points, ny_points, nz_points, prefix=""):
        """ファイル名を生成"""
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        3Dソリューションを可視化（スライス図として、主要な解を複数のスライスで表示）
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト [psi, psi_x, psi_y, psi_z, psi_xx, psi_yy, psi_zz, ...]
            exact: 厳密解のリスト
            errors: 誤差のリスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # CuPy配列をNumPyに変換する関数
        def to_numpy(x):
            return x.get() if hasattr(x, "get") else x
        
        # 可視化する主要なソリューションの選択
        # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]のうち主要な成分
        solution_indices = [0, 1, 4, 7, 2, 5, 8]  # ψ, ψ_x, ψ_y, ψ_z, ψ_xx, ψ_yy, ψ_zz
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz"]
        
        # カラーマップ
        cmap_sol = 'viridis'
        cmap_err = 'hot'
        
        # 各主要成分ごとにスライスを表示
        for sol_idx, sol_name in zip(solution_indices, solution_names):
            num = to_numpy(numerical[sol_idx])
            ex = to_numpy(exact[sol_idx])
            error = to_numpy(np.abs(num - ex))
            
            # 中央のスライスを取得
            slice_x = nx_points // 2
            slice_y = ny_points // 2
            slice_z = nz_points // 2
            
            # 値の範囲を決定
            vmin = min(np.min(num), np.min(ex))
            vmax = max(np.max(num), np.max(ex))
            
            # スライス表示するためのfigureを作成
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f"{function_name}: {sol_name} ({nx_points}x{ny_points}x{nz_points} points)", fontsize=14)
            
            # XY平面のスライス (Z中央)
            ax1 = fig.add_subplot(231)
            im1 = ax1.imshow(num[:, :, slice_z].T, origin='lower', cmap=cmap_sol, vmin=vmin, vmax=vmax)
            ax1.set_title(f"XY-Plane (z={slice_z}) - Numerical")
            plt.colorbar(im1, ax=ax1)
            
            ax2 = fig.add_subplot(232)
            im2 = ax2.imshow(ex[:, :, slice_z].T, origin='lower', cmap=cmap_sol, vmin=vmin, vmax=vmax)
            ax2.set_title(f"XY-Plane (z={slice_z}) - Exact")
            plt.colorbar(im2, ax=ax2)
            
            ax3 = fig.add_subplot(233)
            im3 = ax3.imshow(error[:, :, slice_z].T, origin='lower', cmap=cmap_err)
            ax3.set_title(f"XY-Plane (z={slice_z}) - Error")
            plt.colorbar(im3, ax=ax3)
            
            # XZ平面のスライス (Y中央)
            ax4 = fig.add_subplot(234)
            im4 = ax4.imshow(num[:, slice_y, :].T, origin='lower', cmap=cmap_sol, vmin=vmin, vmax=vmax)
            ax4.set_title(f"XZ-Plane (y={slice_y}) - Numerical")
            plt.colorbar(im4, ax=ax4)
            
            ax5 = fig.add_subplot(235)
            im5 = ax5.imshow(ex[:, slice_y, :].T, origin='lower', cmap=cmap_sol, vmin=vmin, vmax=vmax)
            ax5.set_title(f"XZ-Plane (y={slice_y}) - Exact")
            plt.colorbar(im5, ax=ax5)
            
            ax6 = fig.add_subplot(236)
            im6 = ax6.imshow(error[:, slice_y, :].T, origin='lower', cmap=cmap_err)
            ax6.set_title(f"XZ-Plane (y={slice_y}) - Error")
            plt.colorbar(im6, ax=ax6)
            
            plt.tight_layout()
            
            # 保存ファイル名
            if save:
                filepath = self.generate_filename(f"{function_name}_{sol_name}", nx_points, ny_points, nz_points, prefix)
                plt.savefig(filepath, dpi=dpi)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        # 3Dボリュームレンダリング（メインの解のみ）
        self._create_3d_visualization(grid, function_name, numerical[0], exact[0], errors[0], prefix, save, show, dpi)
        
        # 誤差サマリー図を作成
        self._create_error_summary(function_name, solution_names, errors[:7], nx_points, ny_points, nz_points, prefix, save, show, dpi)
        
        return True
    
    def _create_3d_visualization(self, grid, function_name, numerical, exact, error, prefix="", save=True, show=False, dpi=150):
        """3Dボリュームビジュアライゼーションを作成"""
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # NumPyに変換
        numerical = numerical.get() if hasattr(numerical, "get") else numerical
        exact = exact.get() if has等.a, "get") else exact
        error = np.abs(numerical - exact)
        
        # 3D等値面の可視化
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(f"{function_name}: 3D Isosurfaces ({nx_points}x{ny_points}x{nz_points} points)", fontsize=14)
        
        # 値の範囲を取得
        vmin = min(np.min(numerical), np.min(exact))
        vmax = max(np.max(numerical), np.max(exact))
        level = (vmin + vmax) / 2  # 中間値の等値面
        
        # 数値解の等値面
        ax1 = fig.add_subplot(131, projection='3d')
        x, y, z = np.meshgrid(
            np.linspace(0, 1, nx_points),
            np.linspace(0, 1, ny_points),
            np.linspace(0, 1, nz_points),
            indexing='ij'
        )
        ax1.scatter(x.flatten(), y.flatten(), z.flatten(), c=numerical.flatten(), 
                   cmap='viridis', alpha=0.02, s=0.5)
        ax1.set_title("Numerical Solution")
        
        # 厳密解の等値面
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(x.flatten(), y.flatten(), z.flatten(), c=exact.flatten(), 
                   cmap='viridis', alpha=0.02, s=0.5)
        ax2.set_title("Exact Solution")
        
        # 誤差の等値面
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(x.flatten(), y.flatten(), z.flatten(), c=error.flatten(), 
                   cmap='hot', alpha=0.05, s=0.5)
        ax3.set_title(f"Error (Max: {np.max(error):.2e})")
        
        plt.tight_layout()
        
        # 保存ファイル名
        if save:
            filepath = self.generate_filename(f"{function_name}_3d", nx_points, ny_points, nz_points, prefix)
            plt.savefig(filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _create_error_summary(self, function_name, solution_names, errors, nx_points, ny_points, nz_points, prefix="", save=True, show=False, dpi=150):
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
        ax.set_title(f"{function_name} Function: Error Summary ({nx_points}x{ny_points}x{nz_points} points)")
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
                nz_points,
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
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        error_types = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz"]
        
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
        
        plt.suptitle(f"Error Comparison for All Functions ({grid_size}x{grid_size}x{grid_size} points)")
        plt.tight_layout()
        
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison_{grid_size}x{grid_size}x{grid_size}.png"
        if not prefix:
            filename = f"{self.output_dir}/all_functions_comparison_{grid_size}x{grid_size}x{grid_size}.png"
            
        plt.savefig(filename, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filename
        
    def visualize_grid_convergence(self, function_name, grid_sizes, results, prefix="", save=True, show=False, dpi=150):
        """グリッド収束性のグラフを生成"""
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz"]
        
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
