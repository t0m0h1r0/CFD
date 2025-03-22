"""
高精度コンパクト差分法 (CCD) の3次元結果可視化

このモジュールは、3次元CCDソルバーの計算結果を効率的かつ包括的に可視化するための
クラスと機能を提供します。多面的な結果分析と直感的な視覚化に重点を置いています。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from skimage import measure
from typing import List, Optional, Tuple, Dict, Union, Any

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer3D(BaseVisualizer):
    """CCDソルバーの3D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_3d"):
        """初期化"""
        super().__init__(output_dir)
        self.default_cmap = 'viridis'
        self.error_cmap = 'hot'
        
    def generate_filename(self, func_name, nx_points, ny_points=None, nz_points=None, prefix="", suffix=""):
        """ファイル名を生成"""
        if ny_points is None:
            ny_points = nx_points
        if nz_points is None:
            nz_points = nx_points
            
        filename = f"{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points"
        if prefix:
            filename = f"{prefix}_{filename}"
        if suffix:
            filename = f"{filename}_{suffix}"
        
        return f"{self.output_dir}/{filename}.png"
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, 
                          prefix="", save=True, show=False, dpi=180):
        """
        3Dソリューションを統合ダッシュボード形式で可視化
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト
            exact: 厳密解のリスト
            errors: 誤差リスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
        """
        # グリッドサイズと中心インデックス
        nx, ny, nz = grid.nx_points, grid.ny_points, grid.nz_points
        mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
        
        # NumPy配列に変換
        X, Y, Z = self._to_numpy(grid.get_points())
        
        # 成分名のリスト
        component_names = self.get_error_types()
        
        # 基本データ
        psi = self._to_numpy(numerical[0])
        psi_ex = self._to_numpy(exact[0])
        psi_error = self._to_numpy(np.abs(psi - psi_ex))
        
        # メインダッシュボード作成
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.7])
        
        # メインタイトル
        fig.suptitle(f"{function_name} Function Analysis ({nx}x{ny}x{nz} points)", 
                    fontsize=18, y=0.98)
        
        # 1. 3D等値面と断面 (左上)
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_3d_isosurfaces(ax_3d, grid, psi, psi_error)
        
        # 2. 直交3断面 (中央上)
        ax_slices = fig.add_subplot(gs[0, 1])
        self._plot_orthogonal_slices(ax_slices, grid, psi, mid_x, mid_y, mid_z)
        
        # 3. 誤差分布 (右上)
        ax_error_dist = fig.add_subplot(gs[0, 2])
        self._plot_error_distribution(ax_error_dist, grid, psi_error, mid_z)
        
        # 4. 主成分プロファイル (左中)
        ax_profile = fig.add_subplot(gs[1, 0])
        self._plot_profiles(ax_profile, grid, numerical, exact, errors, mid_x, mid_y, mid_z)
        
        # 5. 誤差の空間分布可視化 (中央中)
        ax_error_3d = fig.add_subplot(gs[1, 1], projection='3d')
        self._plot_3d_error_distribution(ax_error_3d, grid, psi_error, errors[0])
        
        # 6. 比較スライダー (右中)
        ax_compare = fig.add_subplot(gs[1, 2])
        self._plot_exact_numerical_comparison(ax_compare, grid, psi, psi_ex, mid_z)
        
        # 7. 誤差サマリーと統計情報 (下段全体)
        ax_summary = fig.add_subplot(gs[2, :])
        self._plot_error_summary_extended(ax_summary, component_names, numerical, exact, errors)
        
        # レイアウト調整
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存/表示処理
        if save:
            filepath = self.generate_filename(function_name, nx, ny, nz, prefix, "dashboard")
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Dashboard visualization saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # 断面アニメーション風のマルチスライス可視化（別ファイル）
        if save:
            self._create_multi_slice_visualization(grid, function_name, 
                                                  numerical, exact, errors, 
                                                  prefix, dpi)
        
        # 成分別詳細可視化（別ファイル）
        if save:
            self._create_component_detail_visualization(grid, function_name, 
                                                       numerical, exact, errors, 
                                                       prefix, dpi)
        
        return True
    
    def _plot_orthogonal_slices(self, ax, grid, data, mid_x, mid_y, mid_z):
        """3つの直交する断面を1つの図に表示"""
        # 各方向の断面を抽出
        xy_slice = data[:, :, mid_z]
        xz_slice = data[:, mid_y, :]
        yz_slice = data[mid_x, :, :]
        
        # グリッド座標
        X, Y, Z = grid.get_points()
        x = self._to_numpy(X[:, 0, 0])
        y = self._to_numpy(Y[0, :, 0])
        z = self._to_numpy(Z[0, 0, :])
        
        # 2x2のグリッドを作成
        n_grid = max(len(x), len(y), len(z))
        grid_img = np.zeros((n_grid*2, n_grid*2))
        
        # 各断面データを適切な位置に配置
        pad_x = (n_grid*2 - len(x)) // 2
        pad_y = (n_grid*2 - len(y)) // 2
        pad_z = (n_grid*2 - len(z)) // 2
        
        # XY断面（中央）
        grid_img[pad_y:pad_y+len(y), pad_x:pad_x+len(x)] = xy_slice.T
        
        # XZ断面（上部）
        grid_img[:len(z), pad_x:pad_x+len(x)] = xz_slice.T
        
        # YZ断面（右部）
        grid_img[pad_y:pad_y+len(y), -len(z):] = yz_slice
        
        # 表示
        im = ax.imshow(grid_img, cmap=self.default_cmap, origin='lower')
        plt.colorbar(im, ax=ax, label='Value')
        
        # 断面の位置を示す線
        ax.axhline(pad_y+len(y)//2, color='r', linestyle='--', alpha=0.5)
        ax.axvline(pad_x+len(x)//2, color='r', linestyle='--', alpha=0.5)
        
        # 軸ラベルと枠線
        ax.text(pad_x+len(x)//2, -5, "X", ha='center')
        ax.text(-5, pad_y+len(y)//2, "Y", va='center')
        ax.text(n_grid*2-len(z)//2, pad_y+len(y)//2, "Z", ha='center')
        
        # 各断面の領域を枠線で囲む
        # XY断面
        ax.plot([pad_x, pad_x+len(x), pad_x+len(x), pad_x, pad_x],
                [pad_y, pad_y, pad_y+len(y), pad_y+len(y), pad_y], 
                'w-', alpha=0.8)
        # XZ断面
        ax.plot([pad_x, pad_x+len(x), pad_x+len(x), pad_x, pad_x],
                [0, 0, len(z), len(z), 0], 
                'w-', alpha=0.8)
        # YZ断面
        ax.plot([n_grid*2-len(z), n_grid*2, n_grid*2, n_grid*2-len(z), n_grid*2-len(z)],
                [pad_y, pad_y, pad_y+len(y), pad_y+len(y), pad_y], 
                'w-', alpha=0.8)
        
        ax.set_title("Orthogonal Slices Visualization")
        ax.axis('off')
    
    def _plot_3d_isosurfaces(self, ax, grid, data, error_data, levels=3):
        """等値面と断面を組み合わせた3D可視化"""
        X, Y, Z = grid.get_points()
        x = self._to_numpy(X[:, 0, 0])
        y = self._to_numpy(Y[0, :, 0])
        z = self._to_numpy(Z[0, 0, :])
        
        # グリッドの中心インデックス
        mid_x, mid_y, mid_z = len(x)//2, len(y)//2, len(z)//2
        
        # データの最小・最大値
        vmin, vmax = np.min(data), np.max(data)
        
        # 等値面レベルの計算（値の範囲内でlevels個）
        iso_levels = np.linspace(vmin + (vmax - vmin) * 0.2, 
                                vmin + (vmax - vmin) * 0.8, 
                                levels)
        
        # 等値面を描画
        for level_idx, level in enumerate(iso_levels):
            try:
                verts, faces, _, _ = measure.marching_cubes(data, level)
                # 座標変換
                verts_scaled = np.zeros_like(verts)
                verts_scaled[:, 0] = x[0] + verts[:, 0] * (x[-1] - x[0]) / (len(x) - 1)
                verts_scaled[:, 1] = y[0] + verts[:, 1] * (y[-1] - y[0]) / (len(y) - 1)
                verts_scaled[:, 2] = z[0] + verts[:, 2] * (z[-1] - z[0]) / (len(z) - 1)
                
                # 透明度を調整、深いレベルほど不透明に
                alpha = 0.2 + 0.1 * level_idx
                
                # 等値面描画
                ax.plot_trisurf(
                    verts_scaled[:, 0], verts_scaled[:, 1], verts_scaled[:, 2],
                    triangles=faces,
                    color=plt.cm.viridis((level - vmin) / (vmax - vmin)),
                    alpha=alpha
                )
            except Exception as e:
                continue
        
        # メッシュグリッド作成し、断面を表示
        # xy平面
        X_xy, Y_xy = np.meshgrid(x, y)
        Z_xy = np.full_like(X_xy, z[mid_z])
        ax.plot_surface(
            X_xy, Y_xy, Z_xy,
            facecolors=plt.cm.viridis((data[:, :, mid_z].T - vmin) / (vmax - vmin)),
            alpha=0.7, shade=False
        )
        
        # 軸設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Isosurfaces")
        
        # 表示角度の設定
        ax.view_init(elev=30, azim=45)
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6)
    
    def _plot_error_distribution(self, ax, grid, error_data, mid_z):
        """誤差分布ヒートマップを表示"""
        # Z方向中央の誤差断面を表示
        error_slice = error_data[:, :, mid_z]
        
        # 対数スケールで表示（小さな値を持つ場合は対応）
        min_error = np.min(error_data[error_data > 0]) if np.any(error_data > 0) else 1e-10
        
        im = ax.imshow(
            error_slice.T, 
            norm=LogNorm(vmin=min_error, vmax=np.max(error_data)),
            cmap=self.error_cmap, origin='lower'
        )
        
        plt.colorbar(im, ax=ax, label='Error (log scale)')
        ax.set_title(f'Error Distribution at z-slice')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 等高線を追加して誤差レベルを強調
        try:
            levels = np.logspace(np.log10(min_error), np.log10(np.max(error_data)), 5)
            cs = ax.contour(error_slice.T, levels=levels, colors='w', alpha=0.5, linewidths=0.5)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.1e')
        except:
            pass
        
        # グリッド追加
        ax.grid(False)
    
    def _plot_profiles(self, ax, grid, numerical, exact, errors, mid_x, mid_y, mid_z):
        """主要成分の断面プロファイルを表示"""
        # グリッド座標
        X, Y, Z = grid.get_points()
        x = self._to_numpy(X[:, 0, 0])
        
        # 主要成分を選択
        components = [0, 1, 4]  # ψ, ψ_x, ψ_xx
        labels = ["ψ", "ψ_x", "ψ_xx"]
        
        # X方向のプロファイルを表示
        for i, (comp_idx, label) in enumerate(zip(components, labels)):
            num = self._to_numpy(numerical[comp_idx])
            ex = self._to_numpy(exact[comp_idx])
            
            # X方向プロファイル
            profile_num = num[:, mid_y, mid_z]
            profile_ex = ex[:, mid_y, mid_z]
            
            # オフセットを加えて表示（見やすくするため）
            offset = i * np.max(np.abs(profile_ex)) * 1.5
            ax.plot(x, profile_num + offset, 'r-', linewidth=2, alpha=0.7, 
                    label=f"{label} (numerical)" if i == 0 else None)
            ax.plot(x, profile_ex + offset, 'b--', linewidth=1.5, 
                    label=f"{label} (exact)" if i == 0 else None)
            
            # 成分名ラベル
            ax.text(x[0], offset, label, fontsize=10, va='center')
            
            # 誤差値を表示
            ax.text(x[-1], offset, f"Error: {errors[comp_idx]:.2e}", 
                    fontsize=8, ha='right', va='center')
        
        ax.set_title("Component Profiles along X-axis")
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Value (with offset)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_3d_error_distribution(self, ax, grid, error_data, max_error):
        """3D空間内の誤差分布を可視化"""
        X, Y, Z = grid.get_points()
        x = self._to_numpy(X[:, 0, 0])
        y = self._to_numpy(Y[0, :, 0])
        z = self._to_numpy(Z[0, 0, :])
        
        # 誤差閾値を設定（最大誤差の一定割合）
        error_threshold = max_error * 0.1
        
        # 閾値以上の誤差を持つ点を抽出
        high_error_indices = np.where(error_data > error_threshold)
        
        if len(high_error_indices[0]) > 0:
            # 各点の座標と誤差値
            xe = x[high_error_indices[0]]
            ye = y[high_error_indices[1]]
            ze = z[high_error_indices[2]]
            e_values = error_data[high_error_indices]
            
            # 誤差値に基づいてサイズとカラーを決定
            sizes = 10 + 100 * (e_values / max_error)
            colors = e_values / max_error
            
            # 散布図として表示
            scatter = ax.scatter(xe, ye, ze, s=sizes, c=colors, 
                                cmap=self.error_cmap, alpha=0.6)
            
            # カラーバー
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Error Magnitude')
        else:
            ax.text(0.5, 0.5, 0.5, "No significant errors", 
                    ha='center', va='center', transform=ax.transAxes)
        
        # ドメイン境界を表示
        ax.plot([x[0], x[-1], x[-1], x[0], x[0]],
                [y[0], y[0], y[-1], y[-1], y[0]],
                [z[0], z[0], z[0], z[0], z[0]], 'k-', alpha=0.3)
        ax.plot([x[0], x[-1], x[-1], x[0], x[0]],
                [y[0], y[0], y[-1], y[-1], y[0]],
                [z[-1], z[-1], z[-1], z[-1], z[-1]], 'k-', alpha=0.3)
        for i in range(2):
            for j in range(2):
                ax.plot([x[i*-1], x[i*-1]], [y[j*-1], y[j*-1]], [z[0], z[-1]], 'k-', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Error Distribution")
        ax.view_init(elev=25, azim=30)
    
    def _plot_exact_numerical_comparison(self, ax, grid, numerical, exact, slice_idx):
        """比較スライダービュー（数値解と厳密解の差分を強調）"""
        # Z方向のスライスを使用
        num_slice = numerical[:, :, slice_idx]
        ex_slice = exact[:, :, slice_idx]
        
        # 差分計算
        diff = num_slice - ex_slice
        
        # データ範囲
        vmax = max(abs(np.min(diff)), abs(np.max(diff)))
        vmin = -vmax
        
        # 差分の表示（赤青の発散カラーマップ）
        im = ax.imshow(diff.T, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        
        plt.colorbar(im, ax=ax, label='Numerical - Exact')
        ax.set_title('Solution Difference at z-slice')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 枠線とグリッド
        ax.grid(False)
        
        # ゼロレベルの等高線を強調表示
        try:
            cs = ax.contour(diff.T, levels=[0], colors='k', linewidths=0.5)
            ax.clabel(cs, inline=True, fontsize=8)
        except:
            pass
    
    def _plot_error_summary_extended(self, ax, component_names, numerical, exact, errors):
        """拡張誤差サマリー（統計情報付き）"""
        # 水平バーチャートとして誤差を表示
        y_pos = np.arange(len(component_names))
        
        # 0値対応のために小さな値を設定
        plot_errors = np.array(errors).copy()
        plot_errors[plot_errors == 0] = self.min_log_value
        
        # グリッドとフレーム調整
        ax.set_axisbelow(True)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
        # 横向きバーチャート
        bars = ax.barh(y_pos, plot_errors, height=0.6)
        
        # バーの色を設定（大きな誤差は赤く）
        normalized_errors = plot_errors / max(plot_errors)
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.YlOrRd(normalized_errors[i]))
        
        ax.set_xscale('log')
        ax.set_title("Error Analysis by Component")
        ax.set_xlabel('Maximum Error (log scale)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(component_names)
        
        # バーにラベルを追加
        for i, (bar, err) in enumerate(zip(bars, errors)):
            ax.text(
                max(err, self.min_log_value) * 1.1, 
                i, 
                f'{err:.2e}',
                va='center'
            )
        
        # 追加の統計情報を右側に表示
        # 各成分の統計情報を計算
        stats_data = []
        for i, comp_name in enumerate(component_names):
            num = self._to_numpy(numerical[i])
            ex = self._to_numpy(exact[i])
            err_array = np.abs(num - ex)
            
            stats_data.append({
                'component': comp_name,
                'max_err': errors[i],
                'mean_err': np.mean(err_array),
                'median_err': np.median(err_array),
                'std_err': np.std(err_array),
                'l2_norm': np.sqrt(np.sum(err_array**2)),
                'rms': np.sqrt(np.mean(err_array**2))
            })
        
        # 主要成分のみの詳細統計を右側にテキスト表示
        for i, stats in enumerate(stats_data[:3]):  # 最初の3成分のみ
            ax.text(
                0.75, 0.85 - i*0.25, 
                f"{stats['component']} Statistics:\n"
                f"  Mean Error: {stats['mean_err']:.2e}\n"
                f"  Median Error: {stats['median_err']:.2e}\n"
                f"  RMS Error: {stats['rms']:.2e}\n"
                f"  L2 Norm: {stats['l2_norm']:.2e}",
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
            )
    
    def _create_multi_slice_visualization(self, grid, function_name, numerical, exact, errors, prefix, dpi=150):
        """複数スライスを組み合わせた可視化（アニメーション的効果）"""
        # グリッドサイズと座標
        nx, ny, nz = grid.nx_points, grid.ny_points, grid.nz_points
        X, Y, Z = grid.get_points()
        z = self._to_numpy(Z[0, 0, :])
        
        # 主要成分のみ使用
        psi = self._to_numpy(numerical[0])
        psi_ex = self._to_numpy(exact[0])
        
        # 表示するスライス位置
        n_slices = min(5, nz)
        z_indices = np.linspace(0, nz-1, n_slices).astype(int)
        
        # 図の作成
        fig, axes = plt.subplots(2, n_slices, figsize=(4*n_slices, 8))
        plt.suptitle(f"{function_name}: Multiple Z-Slices Visualization", fontsize=16)
        
        # 全スライスの値範囲を統一
        vmin, vmax = np.min(psi), np.max(psi)
        err_max = np.max(np.abs(psi - psi_ex))
        
        for i, z_idx in enumerate(z_indices):
            # 数値解
            ax_num = axes[0, i]
            im_num = ax_num.imshow(psi[:, :, z_idx].T, origin='lower', 
                                  vmin=vmin, vmax=vmax, cmap=self.default_cmap)
            ax_num.set_title(f"Z={z[z_idx]:.2f}")
            ax_num.set_xlabel('X')
            if i == 0:
                ax_num.set_ylabel('Y - Numerical')
            else:
                ax_num.set_yticklabels([])
            
            # 誤差
            ax_err = axes[1, i]
            err_data = np.abs(psi[:, :, z_idx] - psi_ex[:, :, z_idx])
            im_err = ax_err.imshow(err_data.T, origin='lower', 
                                  vmin=0, vmax=err_max, cmap=self.error_cmap)
            ax_err.set_xlabel('X')
            if i == 0:
                ax_err.set_ylabel('Y - Error')
            else:
                ax_err.set_yticklabels([])
        
        # カラーバーを追加
        cbar_num = plt.colorbar(im_num, ax=axes[0, :], shrink=0.8)
        cbar_num.set_label('Value')
        cbar_err = plt.colorbar(im_err, ax=axes[1, :], shrink=0.8)
        cbar_err.set_label('Error')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存
        filepath = self.generate_filename(function_name, nx, ny, nz, prefix, "multi_slice")
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Multi-slice visualization saved to: {filepath}")
    
    def _create_component_detail_visualization(self, grid, function_name, numerical, exact, errors, prefix, dpi=150):
        """主要成分の詳細可視化"""
        # グリッドサイズと座標
        nx, ny, nz = grid.nx_points, grid.ny_points, grid.nz_points
        mid_z = nz // 2
        
        # 主成分のみを詳細表示
        components = [0, 1, 4]  # ψ, ψ_x, ψ_xx
        component_names = ["ψ", "ψ_x", "ψ_xx"]
        
        # 図の作成
        fig, axes = plt.subplots(len(components), 3, figsize=(15, 5*len(components)))
        plt.suptitle(f"{function_name}: Component Details (z={mid_z})", fontsize=16)
        
        for i, (comp_idx, name) in enumerate(zip(components, component_names)):
            num = self._to_numpy(numerical[comp_idx])
            ex = self._to_numpy(exact[comp_idx])
            err = errors[comp_idx]
            
            # 数値解断面
            ax_num = axes[i, 0]
            im_num = ax_num.imshow(num[:, :, mid_z].T, origin='lower', cmap=self.default_cmap)
            ax_num.set_title(f"{name} (Numerical)")
            plt.colorbar(im_num, ax=ax_num)
            
            # 厳密解断面
            ax_ex = axes[i, 1]
            im_ex = ax_ex.imshow(ex[:, :, mid_z].T, origin='lower', cmap=self.default_cmap)
            ax_ex.set_title(f"{name} (Exact)")
            plt.colorbar(im_ex, ax=ax_ex)
            
            # 誤差断面
            ax_err = axes[i, 2]
            err_data = np.abs(num[:, :, mid_z] - ex[:, :, mid_z])
            # 対数スケールで表示
            min_err = max(np.min(err_data[err_data > 0]), 1e-10) if np.any(err_data > 0) else 1e-10
            norm = LogNorm(vmin=min_err, vmax=max(np.max(err_data), min_err*10))
            im_err = ax_err.imshow(err_data.T, origin='lower', norm=norm, cmap=self.error_cmap)
            ax_err.set_title(f"{name} (Error: {err:.2e})")
            plt.colorbar(im_err, ax=ax_err)
            
            # グリッド線とラベル
            for ax in [ax_num, ax_ex, ax_err]:
                ax.grid(False)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存
        filepath = self.generate_filename(function_name, nx, ny, nz, prefix, "components")
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Component detail visualization saved to: {filepath}")
    
    def compare_all_functions_errors(self, results_summary, grid_size, prefix="", save=True, dpi=150, show=False):
        """すべてのテスト関数の誤差を比較するグラフを生成（拡張版）"""
        func_names = list(results_summary.keys())
        error_types = self.get_error_types()
        
        # 図の作成とレイアウト設定
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1])
        
        # 1. バーチャート（2行5列のサブプロット）
        gs_bars = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0])
        axes_bars = [fig.add_subplot(gs_bars[i//5, i%5]) for i in range(10)]
        
        for i, (error_type, ax) in enumerate(zip(error_types, axes_bars)):
            # 各関数の対応する誤差を取得
            errors = [results_summary[name][i] for name in func_names]
            
            # 0値を最小値に置き換え（対数スケール用）
            plot_errors = [max(err, self.min_log_value) for err in errors]
            
            # バーチャート生成
            bars = ax.bar(np.arange(len(func_names)), plot_errors)
            ax.set_yscale("log")
            ax.set_title(f"{error_type}")
            ax.set_xticks(np.arange(len(func_names)))
            ax.set_xticklabels(func_names, rotation=45, ha="right", fontsize=8)
            
            # グリッド追加
            ax.grid(True, which="major", linestyle="--", alpha=0.5)
            
            # 値のアノテーション
            for j, (bar, err) in enumerate(zip(bars, errors)):
                label = "0.0" if err == 0.0 else f"{err:.1e}"
                ax.annotate(
                    label, xy=(j, plot_errors[j] * 1.1), 
                    ha="center", va="bottom", fontsize=7, rotation=90
                )
        
        # 2. レーダーチャート（各成分ごとの相対誤差）
        ax_radar = fig.add_subplot(gs[1], polar=True)
        self._plot_error_radar_chart(ax_radar, results_summary, error_types[:5])  # 最初の5成分のみ使用
        
        # 3. ヒートマップ（全成分の相対誤差を一覧表示）
        ax_heatmap = fig.add_subplot(gs[2])
        self._plot_error_heatmap(ax_heatmap, results_summary, error_types, func_names)
        
        # 全体のタイトルとレイアウト調整
        plt.suptitle(f"Error Comparison ({grid_size}x{grid_size}x{grid_size} points)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存と表示
        if save:
            filename = f"{self.output_dir}/{prefix}_function_comparison_{grid_size}x{grid_size}x{grid_size}.png"
            if not prefix:
                filename = f"{self.output_dir}/function_comparison_{grid_size}x{grid_size}x{grid_size}.png"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Function comparison saved to: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def _plot_error_radar_chart(self, ax, results_summary, error_types):
        """レーダーチャートで各関数の誤差パターンを比較"""
        func_names = list(results_summary.keys())
        
        # 各関数ごとに各成分の誤差を抽出
        data = []
        for func in func_names:
            # 対応する誤差値を取得
            values = [results_summary[func][error_types.index(t)] for t in error_types]
            data.append(values)
        
        # 0値を置換（対数スケールでの表示用）
        data = np.array(data)
        data[data == 0] = self.min_log_value
        
        # 各成分ごとの最大値で正規化
        max_values = np.max(data, axis=0)
        norm_data = data / max_values
        
        # 角度設定
        angles = np.linspace(0, 2*np.pi, len(error_types), endpoint=False)
        # 閉じたポリゴンにするため最初の角度を最後に追加
        angles = np.append(angles, angles[0])
        
        # レーダーチャートの作成
        for i, func in enumerate(func_names):
            values = norm_data[i]
            # 閉じたポリゴンにするため最初の値を最後に追加
            values = np.append(values, values[0])
            ax.plot(angles, values, 'o-', linewidth=2, label=func)
            ax.fill(angles, values, alpha=0.1)
        
        # 軸ラベルの設定
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(error_types)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
        ax.set_title("Relative Error Pattern by Component")
        
        # グリッド設定
        ax.grid(True)
        
        # 凡例の配置
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def _plot_error_heatmap(self, ax, results_summary, error_types, func_names):
        """ヒートマップで全成分の相対誤差を比較"""
        # 誤差データ収集
        data = np.zeros((len(func_names), len(error_types)))
        for i, func in enumerate(func_names):
            for j, err_type in enumerate(error_types):
                data[i, j] = results_summary[func][j]
        
        # 0値を置換
        data[data == 0] = self.min_log_value
        
        # 各成分ごとの最大値で正規化
        max_values = np.max(data, axis=0)
        norm_data = data / max_values
        
        # ヒートマップ表示
        im = ax.imshow(norm_data, cmap='YlOrRd', aspect='auto')
        
        # 軸ラベルの設定
        ax.set_yticks(np.arange(len(func_names)))
        ax.set_yticklabels(func_names)
        ax.set_xticks(np.arange(len(error_types)))
        ax.set_xticklabels(error_types, rotation=45, ha='right')
        
        # タイトルとカラーバー
        ax.set_title("Relative Error Heatmap (normalized by component)")
        plt.colorbar(im, ax=ax, label='Relative Error')
        
        # セル内に誤差値を表示
        for i in range(len(func_names)):
            for j in range(len(error_types)):
                text = ax.text(j, i, f"{data[i, j]:.1e}",
                              ha="center", va="center", 
                              color="black" if norm_data[i, j] < 0.5 else "white",
                              fontsize=7)
    
    def visualize_grid_convergence(self, function_name, grid_sizes, results, prefix="", save=True, show=False, dpi=150):
        """拡張グリッド収束性の可視化"""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1])
        
        # 成分名
        solution_names = self.get_error_types()
        
        # グリッド間隔
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        # 1. 成分別収束プロット（2行5列のサブプロット）
        gs_conv = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0])
        axes_conv = [fig.add_subplot(gs_conv[i//5, i%5]) for i in range(10)]
        
        # 収束次数の推定結果格納用
        convergence_orders = []
        
        for i, (ax, name) in enumerate(zip(axes_conv, solution_names)):
            # 各グリッドサイズでの誤差を取得
            errors = [results[n][i] for n in grid_sizes]
            
            # 対数プロット
            if all(err == 0 for err in errors):
                ax.text(0.5, 0.5, "All errors are zero", 
                       ha='center', va='center', transform=ax.transAxes)
                convergence_orders.append(float('inf'))
                continue
            
            # 0値を置き換え
            plot_errors = [max(err, self.min_log_value) for err in errors]
            
            # 対数-対数プロット
            ax.loglog(grid_spacings, plot_errors, 'o-', linewidth=2, label=name)
            
            # 収束次数を推定（最小二乗法）
            if len(grid_spacings) >= 2 and not all(e == 0 for e in errors):
                # 対数変換して線形回帰
                log_h = np.log(grid_spacings)
                log_e = np.log(plot_errors)
                A = np.vstack([log_h, np.ones(len(log_h))]).T
                order, _ = np.linalg.lstsq(A, log_e, rcond=None)[0]
                
                # 収束次数を保存
                convergence_orders.append(order)
                
                # 推定した次数の線を描画
                x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                y_ref = scale * x_ref ** order
                ax.loglog(x_ref, y_ref, '--', color='red', 
                         label=f'O(h^{order:.2f})')
                
                # 理論的な収束次数の参照線（単純化）
                for ref_order, style in zip([2, 4], ['-.', ':']):
                    scale = plot_errors[-1] / (grid_spacings[-1] ** ref_order)
                    y_ref = scale * x_ref ** ref_order
                    ax.loglog(x_ref, y_ref, style, color='gray', alpha=0.7,
                             label=f'O(h^{ref_order})')
            else:
                convergence_orders.append(None)
            
            ax.set_title(f"{name}")
            ax.set_xlabel('h')
            ax.set_ylabel('Error')
            ax.grid(True)
            ax.legend(fontsize=8)
        
        # 2. 収束次数バーチャート
        ax_order = fig.add_subplot(gs[1])
        self._plot_convergence_order_chart(ax_order, solution_names, convergence_orders)
        
        # 3. 格子サイズごとの誤差ヒートマップ
        ax_heatmap = fig.add_subplot(gs[2])
        self._plot_convergence_heatmap(ax_heatmap, solution_names, grid_sizes, results)
        
        # タイトルとレイアウト調整
        plt.suptitle(f"Grid Convergence: {function_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存と表示
        if save:
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Grid convergence visualization saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def _plot_convergence_order_chart(self, ax, solution_names, convergence_orders):
        """収束次数の棒グラフ"""
        # 有限の収束次数のみ表示（無限大や未定義は除外）
        valid_indices = [i for i, order in enumerate(convergence_orders) 
                         if order is not None and not np.isinf(order) and not np.isnan(order)]
        
        if not valid_indices:
            ax.text(0.5, 0.5, "No valid convergence orders", 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        valid_names = [solution_names[i] for i in valid_indices]
        valid_orders = [convergence_orders[i] for i in valid_indices]
        
        # 棒グラフ
        bars = ax.bar(np.arange(len(valid_names)), valid_orders)
        
        # 理論的収束次数の参照線
        ax.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='2nd order')
        ax.axhline(y=4, color='g', linestyle='--', alpha=0.7, label='4th order')
        
        # バーに値を表示
        for i, (bar, order) in enumerate(zip(bars, valid_orders)):
            height = order
            ax.text(i, height + 0.1, f'{order:.2f}', 
                   ha='center', va='bottom')
        
        # 軸設定
        ax.set_xticks(np.arange(len(valid_names)))
        ax.set_xticklabels(valid_names)
        ax.set_ylabel('Convergence Order')
        ax.set_title('Estimated Convergence Orders by Component')
        ax.grid(True, axis='y')
        ax.legend()
    
    def _plot_convergence_heatmap(self, ax, solution_names, grid_sizes, results):
        """格子サイズと成分ごとの誤差ヒートマップ"""
        # 誤差データ収集
        data = np.zeros((len(grid_sizes), len(solution_names)))
        for i, size in enumerate(grid_sizes):
            for j in range(len(solution_names)):
                data[i, j] = results[size][j]
        
        # 0値を置換
        data[data == 0] = self.min_log_value
        
        # 対数スケールで表示
        log_data = np.log10(data)
        
        # ヒートマップ表示
        im = ax.imshow(log_data, cmap='coolwarm_r', aspect='auto')
        
        # 軸ラベルの設定
        ax.set_yticks(np.arange(len(grid_sizes)))
        ax.set_yticklabels([f'Size {n}' for n in grid_sizes])
        ax.set_xticks(np.arange(len(solution_names)))
        ax.set_xticklabels(solution_names, rotation=45, ha='right')
        
        # タイトルとカラーバー
        ax.set_title("Error Log10 Values by Grid Size and Component")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Log10(Error)')
        
        # セル内に値を表示
        for i in range(len(grid_sizes)):
            for j in range(len(solution_names)):
                text = ax.text(j, i, f"{data[i, j]:.1e}",
                              ha="center", va="center", 
                              color="white" if log_data[i, j] < -5 else "black",
                              fontsize=7)
    
    def get_dimension_label(self) -> str:
        """次元ラベルを返す"""
        return "3D"
    
    def get_error_types(self) -> List[str]:
        """エラータイプのリストを返す"""
        return ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]