"""
高精度コンパクト差分法 (CCD) の3次元結果可視化

このモジュールは、3次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from skimage import measure

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer3D(BaseVisualizer):
    """CCDソルバーの3D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_3d"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス
        """
        super().__init__(output_dir)
    
    def generate_filename(self, func_name, nx_points, ny_points=None, nz_points=None, prefix=""):
        """
        ファイル名を生成
        
        Args:
            func_name: 関数名
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数（Noneの場合はnx_pointsと同じ）
            nz_points: z方向の格子点数（Noneの場合はnx_pointsと同じ）
            prefix: 接頭辞
            
        Returns:
            生成されたファイルパス
        """
        if ny_points is None:
            ny_points = nx_points
        if nz_points is None:
            nz_points = nx_points
            
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        3Dソリューションを可視化（断面図として）
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト [psi, psi_x, psi_y, psi_z, psi_xx, psi_yy, psi_zz, psi_xxx, psi_yyy, psi_zzz]
            exact: 厳密解のリスト
            errors: 誤差のリスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            保存に成功したかどうかのブール値
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # 中心断面のインデックス
        mid_x = nx_points // 2
        mid_y = ny_points // 2
        mid_z = nz_points // 2
        
        # CuPy配列をNumPyに変換
        X, Y, Z = grid.get_points()
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        Z_np = self._to_numpy(Z)
        
        # 可視化するソリューションのリスト
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
        
        # 3つの主要断面について可視化
        for plane, (idx_name, idx) in enumerate([('x', mid_x), ('y', mid_y), ('z', mid_z)]):
            # 主要解成分だけを可視化
            vis_components = [0, 1, 4, 7]  # ψ, ψ_x, ψ_xx, ψ_xxx
            
            # 2x2の図を作成（コンポーネント数に合わせて）
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()  # 1次元配列に変換して扱いやすくする
            plt.suptitle(f"{function_name}: {idx_name}={idx} Plane ({nx_points}x{ny_points}x{nz_points} points)")
            
            # カラーマップ
            cmap_sol = 'viridis'
            
            for i, comp_idx in enumerate(vis_components):
                name = solution_names[comp_idx]
                num = numerical[comp_idx]
                ex = exact[comp_idx]
                err = errors[comp_idx]
                
                # NumPy配列に変換
                num_np = self._to_numpy(num)
                ex_np = self._to_numpy(ex)
                error_np = self._to_numpy(np.abs(num_np - ex_np))
                
                # 平面ごとの断面を抽出
                if idx_name == 'x':
                    num_slice = num_np[idx, :, :]
                    ex_slice = ex_np[idx, :, :]
                    error_slice = error_np[idx, :, :]
                    xlabel, ylabel = 'Y', 'Z'
                    x_data, y_data = Y_np[idx, :, 0], Z_np[idx, 0, :]
                    xx, yy = np.meshgrid(x_data, y_data, indexing='ij')
                elif idx_name == 'y':
                    num_slice = num_np[:, idx, :]
                    ex_slice = ex_np[:, idx, :]
                    error_slice = error_np[:, idx, :]
                    xlabel, ylabel = 'X', 'Z'
                    x_data, y_data = X_np[:, idx, 0], Z_np[0, idx, :]
                    xx, yy = np.meshgrid(x_data, y_data, indexing='ij')
                else:  # idx_name == 'z'
                    num_slice = num_np[:, :, idx]
                    ex_slice = ex_np[:, :, idx]
                    error_slice = error_np[:, :, idx]
                    xlabel, ylabel = 'X', 'Y'
                    x_data, y_data = X_np[:, 0, idx], Y_np[0, :, idx]
                    xx, yy = np.meshgrid(x_data, y_data, indexing='ij')
                
                # 同じカラーバーの範囲を使用するため、最小値と最大値を計算
                vmin = min(np.min(num_slice), np.min(ex_slice))
                vmax = max(np.max(num_slice), np.max(ex_slice))
                
                # 数値解
                ax = axes[i]
                im = ax.contourf(xx, yy, num_slice, 50, cmap=cmap_sol, vmin=vmin, vmax=vmax)
                ax.set_title(f"{name} (Numerical), Error: {err:.2e}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            
            # 保存ファイル名
            if save:
                plane_suffix = f"_plane_{idx_name}{idx}"
                filepath = self.generate_filename(function_name, nx_points, ny_points, nz_points, f"{prefix}{plane_suffix}")
                plt.savefig(filepath, dpi=dpi)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        # 3Dボリュームスライスの可視化（主要成分のみ）
        self._create_3d_volume_visualization(grid, function_name, numerical[0], exact[0], errors[0], prefix, save, show, dpi)
        
        # 誤差サマリー図を作成
        self._create_error_summary(function_name, solution_names, errors, nx_points, ny_points, nz_points, prefix, save, show, dpi)
        
        return True
    
    def _create_3d_volume_visualization(self, grid, function_name, numerical, exact, error, prefix="", save=True, show=False, dpi=150):
        """
        改良された3Dボリューム可視化メソッド
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解 (ψ)
            exact: 厳密解 (ψ)
            error: 誤差値
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # NumPy配列に変換
        num_np = self._to_numpy(numerical)
        
        # メッシュグリッド
        X, Y, Z = grid.get_points()
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        Z_np = self._to_numpy(Z)
        
        # 4つのサブプロットを作成: 等値面と3つの主要断面
        fig = plt.figure(figsize=(18, 12))
        
        # カラーマップの範囲
        vmin, vmax = np.min(num_np), np.max(num_np)
        
        # 1. 等値面プロット (上左)
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 複数の等値面を表示（値の範囲に応じて調整）
        levels = np.linspace(vmin + 0.05 * (vmax - vmin), vmax - 0.05 * (vmax - vmin), 4)
        
        # 等値面ごとに異なる色と不透明度を設定
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        alphas = [0.3, 0.4, 0.5, 0.6]
        
        for i, (level, color, alpha) in enumerate(zip(levels, colors, alphas)):
            # 等値面のマスク作成
            verts, faces, _, _ = measure.marching_cubes(num_np, level)
            
            # グリッド座標に変換
            x_min, x_max = grid.x_min, grid.x_max
            y_min, y_max = grid.y_min, grid.y_max
            z_min, z_max = grid.z_min, grid.z_max
            
            x_vals = x_min + (x_max - x_min) * verts[:, 0] / (nx_points - 1)
            y_vals = y_min + (y_max - y_min) * verts[:, 1] / (ny_points - 1)
            z_vals = z_min + (z_max - z_min) * verts[:, 2] / (nz_points - 1)
            
            # 等値面を描画
            mesh = ax1.plot_trisurf(x_vals, y_vals, z_vals, triangles=faces,
                                  color=color, alpha=alpha, shade=True, 
                                  label=f'Level {i+1}: {level:.2f}')
        
        ax1.set_title(f"{function_name}: Isosurfaces", fontsize=12)
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax1.legend()
        
        # 2. XY断面プロット (上右)
        ax2 = fig.add_subplot(222)
        mid_z = nz_points // 2
        z_val = Z_np[0, 0, mid_z]
        
        # 少ない等高線でより明確に
        cs = ax2.contourf(X_np[:, :, mid_z], Y_np[:, :, mid_z], num_np[:, :, mid_z], 
                      levels=15, cmap='viridis')
        plt.colorbar(cs, ax=ax2)
        
        # より明確な輪郭線を追加
        contour = ax2.contour(X_np[:, :, mid_z], Y_np[:, :, mid_z], num_np[:, :, mid_z], 
                          levels=8, colors='k', linewidths=0.5)
        ax2.clabel(contour, inline=True, fontsize=8)
        
        ax2.set_title(f"XY Plane (Z = {z_val:.2f})", fontsize=12)
        ax2.set_xlabel('X'), ax2.set_ylabel('Y')
        ax2.grid(False)  # グリッド線を非表示に
        
        # 3. XZ断面プロット (下左)
        ax3 = fig.add_subplot(223)
        mid_y = ny_points // 2
        y_val = Y_np[0, mid_y, 0]
        
        cs = ax3.contourf(X_np[:, mid_y, :], Z_np[:, mid_y, :], num_np[:, mid_y, :], 
                      levels=15, cmap='viridis')
        plt.colorbar(cs, ax=ax3)
        
        contour = ax3.contour(X_np[:, mid_y, :], Z_np[:, mid_y, :], num_np[:, mid_y, :], 
                          levels=8, colors='k', linewidths=0.5)
        ax3.clabel(contour, inline=True, fontsize=8)
        
        ax3.set_title(f"XZ Plane (Y = {y_val:.2f})", fontsize=12)
        ax3.set_xlabel('X'), ax3.set_ylabel('Z')
        ax3.grid(False)
        
        # 4. YZ断面プロット (下右)
        ax4 = fig.add_subplot(224)
        mid_x = nx_points // 2
        x_val = X_np[mid_x, 0, 0]
        
        cs = ax4.contourf(Y_np[mid_x, :, :], Z_np[mid_x, :, :], num_np[mid_x, :, :], 
                      levels=15, cmap='viridis')
        plt.colorbar(cs, ax=ax4)
        
        contour = ax4.contour(Y_np[mid_x, :, :], Z_np[mid_x, :, :], num_np[mid_x, :, :], 
                          levels=8, colors='k', linewidths=0.5)
        ax4.clabel(contour, inline=True, fontsize=8)
        
        ax4.set_title(f"YZ Plane (X = {x_val:.2f})", fontsize=12)
        ax4.set_xlabel('Y'), ax4.set_ylabel('Z')
        ax4.grid(False)
        
        # 全体タイトルを設定
        plt.suptitle(f"{function_name}: 3D Visualization (Error: {error:.2e})", fontsize=14)
        
        plt.tight_layout()
        
        # 保存ファイル名
        if save:
            volume_filepath = self.generate_filename(function_name, nx_points, ny_points, nz_points, f"{prefix}_volume")
            plt.savefig(volume_filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # 追加の可視化: 3Dボリュームレンダリング（オプション）
        self._create_volume_rendering(grid, function_name, num_np, error, prefix, save, show, dpi)

    def _create_volume_rendering(self, grid, function_name, num_np, error, prefix="", save=True, show=False, dpi=150):
        """
        3Dボリュームレンダリングを使用した可視化（より直感的）
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            num_np: 数値解データ (NumPy配列)
            error: 誤差値
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # データの正規化（0-1の範囲に）
        v_min, v_max = np.min(num_np), np.max(num_np)
        normalized_data = (num_np - v_min) / (v_max - v_min)
        
        # 図の作成
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # ボリュームレンダリングのための座標系設定
        x = np.linspace(grid.x_min, grid.x_max, nx_points)
        y = np.linspace(grid.y_min, grid.y_max, ny_points)
        z = np.linspace(grid.z_min, grid.z_max, nz_points)
        
        # スパースサンプリング（可視化を明確にするため）
        stride = max(1, min(nx_points, ny_points, nz_points) // 20)
        x_sparse = x[::stride]
        y_sparse = y[::stride]
        z_sparse = z[::stride]
        
        # 各ポイントでの値を取得（間引いたデータ）
        points = []
        values = []
        sizes = []
        
        for i in range(0, nx_points, stride):
            for j in range(0, ny_points, stride):
                for k in range(0, nz_points, stride):
                    val = normalized_data[i, j, k]
                    
                    # 低い値はスキップして視覚的なノイズを減らす
                    if val < 0.25:
                        continue
                    
                    points.append((x[i], y[j], z[k]))
                    values.append(val)
                    # 値が大きいほど、点を大きく表示
                    sizes.append(50 * val**2 + 10)
        
        if points:  # 点がある場合のみプロット
            points = np.array(points)
            
            # 3Dスキャッタープロットを使用したボリュームビジュアライゼーション
            scatter = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=values, cmap='plasma',
                s=sizes, alpha=0.6,
                edgecolors='none'
            )
            
            plt.colorbar(scatter, ax=ax, label='Normalized Value')
        
        ax.set_title(f"{function_name}: Volume Rendering (Error: {error:.2e})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # x, y, z軸の目盛りを減らして見やすく
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.zaxis.set_major_locator(plt.MaxNLocator(5))
        
        # 保存ファイル名
        if save:
            volume_filepath = self.generate_filename(function_name, nx_points, ny_points, nz_points, f"{prefix}_volume_render")
            plt.savefig(volume_filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _create_error_summary(self, function_name, solution_names, errors, nx_points, ny_points, nz_points, prefix="", save=True, show=False, dpi=150):
        """
        誤差のサマリーグラフを作成
        
        Args:
            function_name: 関数名
            solution_names: 解成分の名前リスト
            errors: 誤差リスト
            nx_points, ny_points, nz_points: 格子点数
            prefix: 接頭辞
            save: 保存するかどうか
            show: 表示するかどうか
            dpi: 画像のDPI値
        """
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
        ax.set_xticklabels(solution_names, rotation=45, ha="right")
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
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 10))
        error_types = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
        
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
        fig, axes = plt.subplots(2, 5, figsize=(18, 10))
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
        
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
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence_3d.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence_3d.png"
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
            "3D"
        """
        return "3D"
    
    def get_error_types(self) -> List[str]:
        """
        エラータイプのリストを返す
        
        Returns:
            3D用のエラータイプリスト
        """
        return ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]