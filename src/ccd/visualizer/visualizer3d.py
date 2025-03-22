"""
高精度コンパクト差分法 (CCD) の3次元結果可視化

このモジュールは、3次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple, Dict

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
        # カラーマップ設定
        self.cmap_solution = 'viridis'
        self.cmap_error = 'hot'
        # 解コンポーネント名
        self.solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
    
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
        3Dソリューションを可視化（統合ビュー）
        
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
        
        # 主要コンポーネントと各平面の統合ビュー
        for plane_name, mid_idx in [('x', mid_x), ('y', mid_y), ('z', mid_z)]:
            # 主要コンポーネントのみ可視化 (4つのプライマリコンポーネント)
            primary_indices = [0, 1, 4, 7]  # ψ, ψ_x, ψ_xx, ψ_xxx
            component_names = [self.solution_names[i] for i in primary_indices]
            
            # 2x4のレイアウト (4つのプライマリコンポーネント × (解+誤差))
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
                
                # 指定平面の断面データを抽出
                if plane_name == 'x':
                    num_slice = num_np[mid_idx, :, :]
                    ex_slice = ex_np[mid_idx, :, :]
                    error_slice = error_np[mid_idx, :, :]
                    x_data, y_data = Y_np[mid_idx, :, 0], Z_np[mid_idx, 0, :]
                    axis_labels = ['Y', 'Z']
                elif plane_name == 'y':
                    num_slice = num_np[:, mid_idx, :]
                    ex_slice = ex_np[:, mid_idx, :]
                    error_slice = error_np[:, mid_idx, :]
                    x_data, y_data = X_np[:, mid_idx, 0], Z_np[0, mid_idx, :]
                    axis_labels = ['X', 'Z']
                else:  # plane_name == 'z'
                    num_slice = num_np[:, :, mid_idx]
                    ex_slice = ex_np[:, :, mid_idx]
                    error_slice = error_np[:, :, mid_idx]
                    x_data, y_data = X_np[:, 0, mid_idx], Y_np[0, :, mid_idx]
                    axis_labels = ['X', 'Y']
                
                # メッシュグリッド作成
                xx, yy = np.meshgrid(x_data, y_data, indexing='ij')
                
                # 左側：数値解と等高線
                vmin = min(np.min(num_slice), np.min(ex_slice))
                vmax = max(np.max(num_slice), np.max(ex_slice))
                im1 = axes[i, 0].contourf(xx, yy, num_slice, 50, 
                                        cmap=self.cmap_solution, vmin=vmin, vmax=vmax)
                # 厳密解の等高線をオーバーレイ
                ct = axes[i, 0].contour(xx, yy, ex_slice, 10, colors='black', linewidths=0.5, alpha=0.7)
                
                axes[i, 0].set_title(f"{name} (Numerical)")
                axes[i, 0].set_xlabel(axis_labels[0])
                axes[i, 0].set_ylabel(axis_labels[1])
                plt.colorbar(im1, ax=axes[i, 0])
                
                # 右側：誤差
                # 誤差が非常に小さい場合はログスケールを使用
                if np.max(error_slice) < 1e-5 and np.max(error_slice) > 0:
                    norm = LogNorm(vmin=max(np.min(error_slice[error_slice > 0]), 1e-15), 
                                 vmax=np.max(error_slice))
                else:
                    norm = Normalize(vmin=0, vmax=np.max(error_slice))
                
                im2 = axes[i, 1].contourf(xx, yy, error_slice, 50, cmap=self.cmap_error, norm=norm)
                axes[i, 1].set_title(f"{name} Error (Max: {err:.2e})")
                axes[i, 1].set_xlabel(axis_labels[0])
                axes[i, 1].set_ylabel(axis_labels[1])
                plt.colorbar(im2, ax=axes[i, 1])
                
                # 誤差の等高線をオーバーレイ
                if np.max(error_slice) > 0:
                    err_levels = np.logspace(
                        np.log10(max(np.min(error_slice[error_slice > 0]), 1e-15)), 
                        np.log10(np.max(error_slice)), 
                        5
                    )
                    axes[i, 1].contour(xx, yy, error_slice, levels=err_levels, 
                                     colors='black', linewidths=0.5, alpha=0.7)
            
            plt.suptitle(f"{function_name}: {plane_name}={mid_idx} Plane ({nx_points}x{ny_points}x{nz_points} points)")
            plt.tight_layout()
            
            # 保存ファイル名
            if save:
                plane_suffix = f"_plane_{plane_name}{mid_idx}"
                filepath = self.generate_filename(function_name, nx_points, ny_points, nz_points, 
                                               f"{prefix}{plane_suffix}")
                plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        # 3Dボリュームビューの生成
        self._create_integrated_3d_visualization(grid, function_name, numerical, exact, errors, 
                                              prefix, save, show, dpi)
        
        # 誤差サマリーの生成
        self._create_error_summary(function_name, self.solution_names, errors, 
                                nx_points, ny_points, nz_points, prefix, save, show, dpi)
        
        return True
    
    def _create_integrated_3d_visualization(self, grid, function_name, numerical, exact, errors, 
                                         prefix="", save=True, show=False, dpi=150):
        """
        3D統合可視化（等値面のみ - 断面なし）
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解リスト
            exact: 厳密解リスト
            errors: 誤差リスト
            prefix: 接頭辞
            save: 保存するかどうか
            show: 表示するかどうか
            dpi: 画像のDPI値
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # 中心断面のインデックス (スライス図用)
        mid_x, mid_y, mid_z = nx_points//2, ny_points//2, nz_points//2
        
        # NumPy配列に変換
        num_psi = self._to_numpy(numerical[0])  # ψ成分のみ
        ex_psi = self._to_numpy(exact[0])
        error_psi = self._to_numpy(np.abs(num_psi - ex_psi))
        
        # グリッドポイント取得
        X, Y, Z = grid.get_points()
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        Z_np = self._to_numpy(Z)
        
        # 2x2のサブプロットレイアウト
        fig = plt.figure(figsize=(14, 12))
        
        # 1. 3Dボリュームレンダリング (等値面のみ)
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 等値面の値を決定（複数のレベルでより詳細な表示）
        val_max = np.max(num_psi)
        val_min = np.min(num_psi)
        if val_max > val_min:
            # より多くの等値面を使用して詳細化
            levels = np.linspace(0.3, 0.8, 4)  # 30%, 50%, 70%, 90%の4レベル
            iso_values = [val_min + level * (val_max - val_min) for level in levels]
            
            # 各等値面の色とアルファ値を設定
            alphas = [0.3, 0.4, 0.5, 0.6]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(iso_values)))
            
            # 各等値面の表示
            for idx, (iso_val, alpha, color) in enumerate(zip(iso_values, alphas, colors)):
                # 等値面の検出（単純化のため）
                verts = []
                step = max(1, min(nx_points, ny_points, nz_points) // 50)  # データ削減
                for i in range(1, nx_points-1, step):
                    for j in range(1, ny_points-1, step):
                        for k in range(1, nz_points-1, step):
                            if (num_psi[i, j, k] > iso_val and \
                                (num_psi[i-1, j, k] < iso_val or num_psi[i+1, j, k] < iso_val or
                                 num_psi[i, j-1, k] < iso_val or num_psi[i, j+1, k] < iso_val or
                                 num_psi[i, j, k-1] < iso_val or num_psi[i, j, k+1] < iso_val)):
                                verts.append((X_np[i, j, k], Y_np[i, j, k], Z_np[i, j, k]))
                
                # 代表点をプロットして等値面を近似
                if verts:
                    verts = np.array(verts)
                    ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
                              c=[color], alpha=alpha, s=15, edgecolors='none',
                              label=f'Level: {iso_val:.3f}')
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                 norm=plt.Normalize(vmin=val_min, vmax=val_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, shrink=0.6)
        cbar.set_label('ψ Value')
        
        ax1.set_title(f"3D Volume Visualization\nψ Component (Error: {errors[0]:.2e})")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 軸の設定
        ax1.set_xlim(-1.0, 1.0)
        ax1.set_ylim(-1.0, 1.0)
        ax1.set_zlim(-1.0, 1.0)
        
        # 2. 誤差分布の3D可視化（等値面のみ）
        ax2 = fig.add_subplot(222, projection='3d')
        
        # エラー値の範囲を特定
        if np.max(error_psi) > 0:
            # 対数スケールでの等値面レベル
            err_max = np.max(error_psi)
            err_min = max(np.min(error_psi[error_psi > 0]), 1e-15)
            
            # 対数スケールでの等値面レベル
            error_levels = np.logspace(np.log10(err_min), np.log10(err_max), 4)
            error_colors = plt.cm.hot(np.linspace(0.2, 0.8, len(error_levels)))
            error_alphas = [0.3, 0.4, 0.5, 0.6]
            
            # 各誤差レベルの等値面
            for idx, (err_val, alpha, color) in enumerate(zip(error_levels, error_alphas, error_colors)):
                # 誤差等値面の検出
                err_verts = []
                step = max(1, min(nx_points, ny_points, nz_points) // 50)  # データ削減
                for i in range(1, nx_points-1, step):
                    for j in range(1, ny_points-1, step):
                        for k in range(1, nz_points-1, step):
                            if (error_psi[i, j, k] > err_val and \
                                (error_psi[i-1, j, k] < err_val or error_psi[i+1, j, k] < err_val or
                                 error_psi[i, j-1, k] < err_val or error_psi[i, j+1, k] < err_val or
                                 error_psi[i, j, k-1] < err_val or error_psi[i, j, k+1] < err_val)):
                                err_verts.append((X_np[i, j, k], Y_np[i, j, k], Z_np[i, j, k]))
                
                # 誤差等値面のプロット
                if err_verts:
                    err_verts = np.array(err_verts)
                    ax2.scatter(err_verts[:, 0], err_verts[:, 1], err_verts[:, 2], 
                              c=[color], alpha=alpha, s=15, edgecolors='none',
                              label=f'Error: {err_val:.2e}')
        
        # 誤差用のカラーバー
        if np.max(error_psi) > 0:
            error_norm = LogNorm(vmin=max(np.min(error_psi[error_psi > 0]), 1e-15), 
                               vmax=np.max(error_psi))
            sm2 = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=error_norm)
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.6)
            cbar2.set_label('Error (Log Scale)')
            
            # レジェンド表示（誤差レベル）
            ax2.legend(loc='upper right', fontsize=8)
        
        ax2.set_title(f"Error Distribution\nψ Component (Max Error: {errors[0]:.2e})")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 軸の設定
        ax2.set_xlim(-1.0, 1.0)
        ax2.set_ylim(-1.0, 1.0)
        ax2.set_zlim(-1.0, 1.0)
        
        # 3. XY平面の断面比較（z=mid_z）
        ax3 = fig.add_subplot(223)
        
        # 数値解
        im3 = ax3.contourf(X_np[:, :, mid_z], Y_np[:, :, mid_z], num_psi[:, :, mid_z], 
                         50, cmap=self.cmap_solution, alpha=0.8)
        # 厳密解の等高線
        ct3 = ax3.contour(X_np[:, :, mid_z], Y_np[:, :, mid_z], ex_psi[:, :, mid_z], 
                        10, colors='black', linewidths=0.8)
        
        ax3.set_title(f"Z={mid_z} Plane: Numerical Solution with Exact Contours")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im3, ax=ax3)
        
        # 4. XY平面の誤差分布（z=mid_z）
        ax4 = fig.add_subplot(224)
        
        # 誤差値
        error_slice = error_psi[:, :, mid_z]
        
        # 誤差が非常に小さい場合はログスケールを使用
        if np.max(error_slice) < 1e-5 and np.max(error_slice) > 0:
            norm = LogNorm(vmin=max(np.min(error_slice[error_slice > 0]), 1e-15), 
                         vmax=np.max(error_slice))
        else:
            norm = Normalize(vmin=0, vmax=np.max(error_slice))
            
        im4 = ax4.contourf(X_np[:, :, mid_z], Y_np[:, :, mid_z], error_slice, 
                         50, cmap=self.cmap_error, norm=norm)
        
        # 誤差の等高線
        if np.max(error_slice) > 0:
            err_levels = np.logspace(
                np.log10(max(np.min(error_slice[error_slice > 0]), 1e-15)), 
                np.log10(np.max(error_slice)), 
                5
            )
            ax4.contour(X_np[:, :, mid_z], Y_np[:, :, mid_z], error_slice, 
                      levels=err_levels, colors='black', linewidths=0.5)
            
        ax4.set_title(f"Z={mid_z} Plane: Error Distribution")
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        plt.colorbar(im4, ax=ax4)
        
        plt.suptitle(f"{function_name} 3D Analysis ({nx_points}x{ny_points}x{nz_points} points)", 
                   fontsize=16)
        plt.tight_layout()
        
        # 保存
        if save:
            volume_filepath = self.generate_filename(function_name, nx_points, ny_points, nz_points, 
                                                  f"{prefix}_3d_integrated")
            plt.savefig(volume_filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _create_error_summary(self, function_name, solution_names, errors, 
                           nx_points, ny_points, nz_points, prefix="", save=True, show=False, dpi=150):
        """
        誤差のサマリーグラフを作成（棒グラフと分布）
        
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
        # 1x2のレイアウト
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x_pos = np.arange(len(solution_names))
        
        # 1. 標準バープロット
        bars = ax1.bar(x_pos, errors, color='orange', alpha=0.7)
        
        # バーラベル
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
        # ゼロを適切に置き換え
        log_errors = []
        for err in errors:
            if err == 0 or err < self.min_log_value:
                log_errors.append(self.min_log_value)
            else:
                log_errors.append(err)
                
        bars2 = ax2.bar(x_pos, log_errors, color='red', alpha=0.7)
        ax2.set_yscale('log')
        
        # バーラベル
        for bar, err in zip(bars2, errors):
            y_pos = bar.get_height() * 1.1
            if y_pos <= 0:
                y_pos = self.min_log_value
            ax2.annotate(f'{err:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
        
        # x, y, z方向のコンポーネントを色で区別
        for i, name in enumerate(solution_names):
            if '_x' in name:
                bars[i].set_color('blue')
                bars2[i].set_color('blue')
            elif '_y' in name:
                bars[i].set_color('green')
                bars2[i].set_color('green')
            elif '_z' in name:
                bars[i].set_color('purple')
                bars2[i].set_color('purple')
        
        ax2.set_title(f"Component Errors (Log Scale)")
        ax2.set_ylabel('Maximum Error (Log Scale)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(solution_names, rotation=45, ha="right")
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.suptitle(f"{function_name} Function: Error Summary ({nx_points}x{ny_points}x{nz_points} points)")
        plt.tight_layout()
        
        if save:
            summary_filepath = self.generate_filename(
                f"{function_name}_error_summary", 
                nx_points, 
                ny_points,
                nz_points,
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
        
        # 主要コンポーネントの選択
        primary_indices = [0, 1, 2, 3, 4, 5, 6]  # ψ, ψ_x, ψ_y, ψ_z, ψ_xx, ψ_yy, ψ_zz
        primary_names = [self.solution_names[i] for i in primary_indices]
        
        # 3x3のグリッドレイアウト (8コンポーネント + ヒートマップ)
        fig = plt.figure(figsize=(16, 14))
        grid = plt.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 各コンポーネントのエラーバープロット
        axes = []
        for i in range(len(primary_indices)):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(grid[row, col])
            axes.append(ax)
            
            comp_idx = primary_indices[i]
            comp_name = primary_names[i]
            
            # 関数ごとのエラー値
            error_values = []
            for name in func_names:
                err = results_summary[name][comp_idx]
                if err == 0.0 or err < self.min_log_value:
                    error_values.append(self.min_log_value)
                else:
                    error_values.append(err)
            
            # バープロット
            x_positions = np.arange(len(func_names))
            bars = ax.bar(x_positions, error_values)
            ax.set_yscale("log")
            ax.set_title(f"{comp_name} Error")
            ax.set_xlabel("Test Function")
            ax.set_ylabel("Error (log scale)")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(func_names, rotation=45, ha="right", fontsize=8)
            
            # バーに値ラベルを追加
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
        
        # ヒートマップ（すべてのコンポーネント・すべての関数）
        ax_heat = fig.add_subplot(grid[2, :])
        
        # ヒートマップデータ
        heat_data = np.zeros((len(func_names), len(self.solution_names)))
        for i, name in enumerate(func_names):
            for j in range(len(self.solution_names)):
                heat_data[i, j] = results_summary[name][j]
        
        # ゼロ値の処理
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
        
        # ヒートマップに値表示
        for i in range(len(func_names)):
            for j in range(len(self.solution_names)):
                val = heat_data[i, j]
                text = ax_heat.text(j, i, f"{val:.2e}",
                                  ha="center", va="center", color="w" if val > 1e-8 else "black",
                                  fontsize=7)
        
        # カラーバー
        cbar = fig.colorbar(im, ax=ax_heat)
        cbar.set_label('Error (Log Scale)')
        
        plt.suptitle(f"Error Comparison for All Functions ({grid_size}x{grid_size}x{grid_size} points)", 
                   fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 上部にタイトル用の余白
        
        # 保存
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison_{grid_size}x{grid_size}x{grid_size}.png"
        if not prefix:
            filename = f"{self.output_dir}/all_functions_comparison_{grid_size}x{grid_size}x{grid_size}.png"
            
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
        # 3x4レイアウト (軸方向でグループ化)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # 重要なコンポーネントをグループ化
        component_groups = [
            # ψと各方向の1階導関数
            [0, 1, 2, 3],  # ψ, ψ_x, ψ_y, ψ_z
            # 各方向の2階導関数
            [4, 5, 6],     # ψ_xx, ψ_yy, ψ_zz
            # 各方向の3階導関数
            [7, 8, 9]      # ψ_xxx, ψ_yyy, ψ_zzz
        ]
        
        group_titles = [
            "Function & First Derivatives",
            "Second Derivatives",
            "Third Derivatives"
        ]
        
        # グリッド間隔計算
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        # 収束次数の色とスタイル
        order_colors = ['r', 'g', 'b', 'purple']
        order_styles = ['--', '-.', ':', '-']
        order_values = [2, 4, 6, 8]
        
        # 各行＝各グループを処理
        for row, (group, title) in enumerate(zip(component_groups, group_titles)):
            # 各グループのコンポーネント処理
            for i, comp_idx in enumerate(group):
                if i < len(axes[row]):  # 行にある列数分までのインデックスを処理
                    ax = axes[row, i]
                    name = self.solution_names[comp_idx]
                    
                    # グリッドサイズごとの誤差収集
                    errors = [results[n][comp_idx] for n in grid_sizes]
                    
                    # ゼロ値の処理
                    plot_errors = []
                    for err in errors:
                        if err == 0 or err < self.min_log_value:
                            plot_errors.append(self.min_log_value)
                        else:
                            plot_errors.append(err)
                    
                    # 対数プロット
                    ax.loglog(grid_spacings, plot_errors, 'o-', color='black', linewidth=2, 
                            markersize=8, label=name)
                    
                    # 収束次数の参照線
                    if min(plot_errors) > 0:
                        x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                        for j, order in enumerate(order_values):
                            # 最後のデータポイントに合わせて参照線を配置
                            if len(plot_errors) > 1:
                                # 実際の収束次数を推定
                                if plot_errors[-1] > 0 and plot_errors[-2] > 0:
                                    est_order = np.log(plot_errors[-2]/plot_errors[-1]) / np.log(grid_spacings[-2]/grid_spacings[-1])
                                    order_label = f'O(h^{est_order:.1f})' if j == 0 else f'O(h^{order})'
                                    
                                    scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                                    y_ref = scale * x_ref ** order
                                    ax.loglog(x_ref, y_ref, order_styles[j], color=order_colors[j], 
                                            linewidth=1.5, label=order_label if j == 0 else f'O(h^{order})')
                    
                    # 実際の収束次数を計算して表示
                    if len(grid_spacings) >= 2:
                        orders = []
                        for k in range(1, len(grid_spacings)):
                            if plot_errors[k-1] > 0 and plot_errors[k] > 0:
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
            
            # 使用しない軸を非表示
            for i in range(len(group), 4):
                axes[row, i].axis('off')
        
        plt.suptitle(f"Grid Convergence Analysis for {function_name} Function (3D)", fontsize=16)
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence_3d.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence_3d.png"
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
            "3D"
        """
        return "3D"
    
    def get_error_types(self) -> List[str]:
        """
        エラータイプのリストを返す
        
        Returns:
            3D用のエラータイプリスト
        """
        return self.solution_names