"""
高精度コンパクト差分法 (CCD) の3次元結果可視化

このモジュールは、3次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。効率的な可視化と情報量の充実に重点を置いています。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from skimage import measure
from typing import List, Optional, Tuple, Dict

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer3D(BaseVisualizer):
    """CCDソルバーの3D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_3d"):
        """初期化"""
        super().__init__(output_dir)
        # カラーマップ設定
        self.cmap_solution = 'viridis'  # 解用
        self.cmap_error = 'hot'         # 誤差用
        self.cmap_gradient = 'coolwarm' # 勾配用
    
    def generate_filename(self, func_name, nx_points, ny_points=None, nz_points=None, prefix=""):
        """ファイル名を生成"""
        if ny_points is None:
            ny_points = nx_points
        if nz_points is None:
            nz_points = nx_points
            
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=180):
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
        
        # 主要な解成分を抽出
        psi = self._to_numpy(numerical[0])       # 解
        psi_x = self._to_numpy(numerical[1])     # x方向偏導関数
        psi_y = self._to_numpy(numerical[2])     # y方向偏導関数
        psi_z = self._to_numpy(numerical[3])     # z方向偏導関数
        psi_xx = self._to_numpy(numerical[4])    # x方向2階偏導関数
        psi_yy = self._to_numpy(numerical[5])    # y方向2階偏導関数
        psi_zz = self._to_numpy(numerical[6])    # z方向2階偏導関数
        
        # 対応する厳密解
        psi_ex = self._to_numpy(exact[0])
        psi_x_ex = self._to_numpy(exact[1])
        psi_y_ex = self._to_numpy(exact[2])
        psi_z_ex = self._to_numpy(exact[3])
        
        # 誤差計算
        error = self._to_numpy(np.abs(psi - psi_ex))
        error_x = self._to_numpy(np.abs(psi_x - psi_x_ex))
        error_y = self._to_numpy(np.abs(psi_y - psi_y_ex))
        error_z = self._to_numpy(np.abs(psi_z - psi_z_ex))
        
        # 成分名のリスト
        component_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
        
        # 拡張ダッシュボード作成 (3x3グリッド)
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1, 0.8])
        
        # メインタイトル
        fig.suptitle(f"{function_name} Function Analysis ({nx}x{ny}x{nz} points)", 
                    fontsize=18, y=0.98)
        
        # 1. 3D可視化（左上、大きめのスペース）
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_3d_visualization(ax_3d, grid, psi, error, errors[0])
        
        # 2. 3面断面図（中央上）
        ax_slices = fig.add_subplot(gs[0, 1])
        self._plot_orthogonal_slices(ax_slices, grid, psi, mid_x, mid_y, mid_z)
        
        # 3. 勾配可視化（右上）
        ax_gradient = fig.add_subplot(gs[0, 2], projection='3d')
        self._plot_gradient_visualization(ax_gradient, grid, psi_x, psi_y, psi_z)
        
        # 4. 誤差分布ヒートマップ（中段左）
        ax_error = fig.add_subplot(gs[1, 0])
        self._plot_error_distribution(ax_error, grid, error, mid_z)
        
        # 5. ラプラシアン分布（中段中央）
        ax_laplacian = fig.add_subplot(gs[1, 1])
        laplacian = psi_xx + psi_yy + psi_zz
        self._plot_laplacian_distribution(ax_laplacian, grid, laplacian, mid_z)
        
        # 6. 誤差ヒストグラム（中段右）
        ax_error_hist = fig.add_subplot(gs[1, 2])
        self._plot_error_histogram(ax_error_hist, error, error_x, error_y, error_z)
        
        # 7. 誤差サマリー（左下）
        ax_summary = fig.add_subplot(gs[2, 0])
        self._plot_error_summary(ax_summary, component_names, errors, function_name)
        
        # 8. 統計情報（中央下）
        ax_stats = fig.add_subplot(gs[2, 1])
        self._plot_statistics_table(ax_stats, numerical, exact, errors)
        
        # 9. 等値面情報（右下）
        ax_isosurfaces = fig.add_subplot(gs[2, 2])
        self._plot_isosurface_analysis(ax_isosurfaces, psi, psi_ex)
        
        # レイアウト調整
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存/表示処理
        if save:
            filepath = self.generate_filename(function_name, nx, ny, nz, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Visualization saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def _plot_orthogonal_slices(self, ax, grid, data, mid_x, mid_y, mid_z):
        """3つの直交する断面を1つの図に統合表示"""
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
        im = ax.imshow(grid_img, cmap=self.cmap_solution, origin='lower')
        plt.colorbar(im, ax=ax, label='Value')
        
        # ラベル
        ax.set_title(f"Orthogonal Slices at x={mid_x}, y={mid_y}, z={mid_z}")
        ax.axis('off')
        
        # 断面を示す線を追加
        ax.axhline(pad_y+len(y)//2, color='r', linestyle='--', alpha=0.5)
        ax.axvline(pad_x+len(x)//2, color='r', linestyle='--', alpha=0.5)
        
        # 軸ラベルを表示
        ax.text(pad_x+len(x)//2, -5, "X", ha='center')
        ax.text(-5, pad_y+len(y)//2, "Y", va='center')
        ax.text(n_grid*2-len(z)//2, pad_y+len(y)//2, "Z", ha='center')
    
    def _plot_3d_visualization(self, ax, grid, data, error_data, max_error):
        """3D可視化（等値面とスライス表示）"""
        X, Y, Z = grid.get_points()
        x = self._to_numpy(X[:, 0, 0])
        y = self._to_numpy(Y[0, :, 0])
        z = self._to_numpy(Z[0, 0, :])
        
        # グリッドの中心インデックス
        mid_x, mid_y, mid_z = len(x)//2, len(y)//2, len(z)//2
        
        # データの最小・最大値
        vmin, vmax = np.min(data), np.max(data)
        
        # 等値面レベルの計算（値の範囲内で3つ）
        levels = [
            vmin + (vmax - vmin) * 0.25,
            vmin + (vmax - vmin) * 0.5,
            vmin + (vmax - vmin) * 0.75
        ]
        
        # 中央断面をプロット
        x_mesh, z_mesh = np.meshgrid(x, z)
        y_plane = np.ones_like(x_mesh) * y[mid_y]
        ax.plot_surface(
            x_mesh, y_plane, z_mesh, 
            facecolors=plt.cm.viridis((data[:, mid_y, :].T - vmin) / (vmax - vmin)),
            alpha=0.6, shade=False
        )
        
        # 等値面を描画
        for i, level in enumerate(levels):
            try:
                verts, faces, _, _ = measure.marching_cubes(data, level)
                # 座標変換
                verts_scaled = np.zeros_like(verts)
                verts_scaled[:, 0] = x[0] + verts[:, 0] * (x[-1] - x[0]) / (len(x) - 1)
                verts_scaled[:, 1] = y[0] + verts[:, 1] * (y[-1] - y[0]) / (len(y) - 1)
                verts_scaled[:, 2] = z[0] + verts[:, 2] * (z[-1] - z[0]) / (len(z) - 1)
                
                # 透明度を調整して等値面描画
                ax.plot_trisurf(
                    verts_scaled[:, 0], verts_scaled[:, 1], verts_scaled[:, 2],
                    triangles=faces,
                    color=plt.cm.viridis((level - vmin) / (vmax - vmin)),
                    alpha=0.3
                )
                
                # レベル表示を追加
                ax.text(
                    x[0], y[0], z[0] + (i + 1) * (z[-1] - z[0]) / 5,
                    f"Level: {level:.2f}",
                    color=plt.cm.viridis((level - vmin) / (vmax - vmin))
                )
            except:
                continue
        
        # 軸設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Visualization (Max Error: {max_error:.2e})")
        
        # 表示角度の設定
        ax.view_init(elev=15, azim=60)
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7)
    
    def _plot_gradient_visualization(self, ax, grid, grad_x, grad_y, grad_z):
        """勾配ベクトルの可視化"""
        X, Y, Z = grid.get_points()
        x = self._to_numpy(X[:, 0, 0])
        y = self._to_numpy(Y[0, :, 0])
        z = self._to_numpy(Z[0, 0, :])
        
        # サンプリング間隔（視認性向上のため間引き）
        stride = max(1, len(x) // 5)
        
        # グリッドの中心インデックス
        mid_x, mid_y, mid_z = len(x)//2, len(y)//2, len(z)//2
        
        # 勾配ベクトルの大きさ
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        max_mag = np.max(gradient_magnitude)
        
        # 勾配ベクトルの正規化（視覚化のため）
        scale_factor = 0.1 * (x[-1] - x[0]) / max_mag
        u = grad_x * scale_factor
        v = grad_y * scale_factor
        w = grad_z * scale_factor
        
        # XY平面の勾配ベクトルをプロット
        for i in range(0, len(x), stride):
            for j in range(0, len(y), stride):
                ax.quiver(
                    x[i], y[j], z[mid_z],
                    u[i, j, mid_z], v[i, j, mid_z], w[i, j, mid_z],
                    color=plt.cm.coolwarm(gradient_magnitude[i, j, mid_z] / max_mag),
                    alpha=0.7
                )
        
        # XZ平面の勾配ベクトルをプロット
        for i in range(0, len(x), stride):
            for k in range(0, len(z), stride):
                ax.quiver(
                    x[i], y[mid_y], z[k],
                    u[i, mid_y, k], v[i, mid_y, k], w[i, mid_y, k],
                    color=plt.cm.coolwarm(gradient_magnitude[i, mid_y, k] / max_mag),
                    alpha=0.7
                )
        
        # 軸設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Gradient Vectors at Center Planes\nMax Magnitude: {max_mag:.2e}")
        
        # 表示角度の設定
        ax.view_init(elev=15, azim=60)
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=max_mag))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7, label='Gradient Magnitude')
    
    def _plot_error_distribution(self, ax, grid, error_data, mid_z):
        """誤差分布ヒートマップを表示"""
        # Z方向中央の誤差断面を表示
        error_slice = error_data[:, :, mid_z]
        
        # 対数スケールで表示（小さな値を持つ場合は対応）
        min_error = np.min(error_data[error_data > 0]) if np.any(error_data > 0) else 1e-10
        
        im = ax.imshow(
            error_slice.T, 
            norm=LogNorm(vmin=min_error, vmax=np.max(error_data)),
            cmap=self.cmap_error, origin='lower'
        )
        
        plt.colorbar(im, ax=ax, label='Error (log scale)')
        ax.set_title(f'Error Distribution at z={mid_z}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 最大誤差位置をマーク
        max_pos = np.unravel_index(np.argmax(error_slice), error_slice.shape)
        ax.plot(max_pos[0], max_pos[1], 'rx', markersize=10)
        ax.text(max_pos[0], max_pos[1], f" Max: {np.max(error_slice):.2e}", color='r')
    
    def _plot_laplacian_distribution(self, ax, grid, laplacian_data, mid_z):
        """ラプラシアン分布をヒートマップで表示"""
        # Z方向中央の断面を表示
        laplacian_slice = laplacian_data[:, :, mid_z]
        
        # 対称的なカラーマップを使用するための正規化
        abs_max = max(abs(np.min(laplacian_slice)), abs(np.max(laplacian_slice)))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        
        im = ax.imshow(
            laplacian_slice.T, 
            norm=norm,
            cmap='seismic', origin='lower'
        )
        
        plt.colorbar(im, ax=ax, label='Laplacian Value')
        ax.set_title(f'Laplacian Distribution at z={mid_z}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # ゼロクロッシング（ラプラシアンがゼロになる部分）を強調
        ax.contour(
            laplacian_slice.T, levels=[0], 
            colors='g', linestyles='dashed', linewidths=1,
            alpha=0.8
        )
    
    def _plot_error_histogram(self, ax, error, error_x, error_y, error_z):
        """誤差分布のヒストグラムを表示"""
        # 非ゼロの値のみを考慮
        def filtered_hist(data, label, bins=30):
            nonzero_data = data[data > 0].flatten()
            if len(nonzero_data) > 0:
                ax.hist(nonzero_data, bins=bins, alpha=0.6, label=label, 
                       log=True, density=True)
        
        # 各成分の誤差ヒストグラムを描画
        filtered_hist(error, 'ψ')
        filtered_hist(error_x, 'ψ_x')
        filtered_hist(error_y, 'ψ_y')
        filtered_hist(error_z, 'ψ_z')
        
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency (log scale)')
        ax.set_title('Error Histogram')
        ax.legend()
        
        # 統計情報を表示
        ax.text(0.05, 0.95, f"ψ Max: {np.max(error):.2e}",
               transform=ax.transAxes, va='top')
        ax.text(0.05, 0.90, f"ψ Mean: {np.mean(error):.2e}",
               transform=ax.transAxes, va='top')
        ax.text(0.05, 0.85, f"Grad Max: {max(np.max(error_x), np.max(error_y), np.max(error_z)):.2e}",
               transform=ax.transAxes, va='top')
        
        # x軸を対数スケールに設定
        ax.set_xscale('log')
    
    def _plot_error_summary(self, ax, component_names, errors, function_name):
        """誤差サマリープロットを描画"""
        # 水平バーチャートとして表示（読みやすさ向上）
        y_pos = np.arange(len(component_names))
        
        # 0値対応のために小さな値を設定
        plot_errors = np.array(errors).copy()
        plot_errors[plot_errors == 0] = self.min_log_value
        
        # 横向きバーチャート
        bars = ax.barh(y_pos, plot_errors, color='skyblue')
        ax.set_xscale('log')
        ax.set_title(f"Error Summary")
        ax.set_xlabel('Maximum Error (log scale)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(component_names)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # バーにラベルを追加
        for i, (bar, err) in enumerate(zip(bars, errors)):
            ax.text(
                max(err, self.min_log_value) * 1.1, 
                i, 
                f'{err:.2e}',
                va='center'
            )
        
        # 許容誤差レベルのマーク（例: 1e-6）
        ax.axvline(1e-6, color='r', linestyle='--', label='Tolerance (1e-6)')
        ax.legend()
    
    def _plot_statistics_table(self, ax, numerical, exact, errors):
        """統計情報をテーブル形式で表示"""
        ax.axis('off')
        ax.set_title('Solution Statistics', fontweight='bold')
        
        # メインの解成分について統計情報を計算
        stats = []
        
        # 主要成分のみ統計情報を表示
        for i, name in enumerate(["ψ", "ψ_x", "ψ_y", "ψ_z"]):
            num = self._to_numpy(numerical[i])
            ex = self._to_numpy(exact[i])
            
            # 統計値を計算
            max_abs = np.max(np.abs(num))
            mean_abs = np.mean(np.abs(num))
            max_err = errors[i]
            mean_err = np.mean(np.abs(num - ex))
            rel_err = max_err / max_abs if max_abs > 0 else 0
            
            stats.append([name, f"{max_abs:.2e}", f"{mean_abs:.2e}", 
                         f"{max_err:.2e}", f"{mean_err:.2e}", f"{rel_err:.2%}"])
        
        # テーブルを作成
        columns = ['Component', 'Max Abs', 'Mean Abs', 'Max Error', 'Mean Error', 'Rel. Error']
        ax.table(
            cellText=stats,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colWidths=[0.15, 0.16, 0.16, 0.16, 0.16, 0.16]
        )
        
        # グリッド収束性の見積もり情報を追加
        expected_order = self._estimate_convergence_order(errors)
        ax.text(0.5, 0.1, f"Estimated Convergence Order: {expected_order:.1f}",
               ha='center', transform=ax.transAxes, color='blue', fontweight='bold')
    
    def _plot_isosurface_analysis(self, ax, numerical, exact):
        """等値面分析情報を表示"""
        ax.axis('off')
        ax.set_title('Isosurface Analysis', fontweight='bold')
        
        # 値の範囲
        num_min, num_max = np.min(numerical), np.max(numerical)
        ex_min, ex_max = np.min(exact), np.max(exact)
        
        # 特徴的な等値面レベルを計算
        num_levels = [
            num_min,
            num_min + 0.25 * (num_max - num_min),
            num_min + 0.5 * (num_max - num_min),
            num_min + 0.75 * (num_max - num_min),
            num_max
        ]
        
        # 数値データに対する等値面の体積を計算
        volumes = []
        for level in num_levels:
            try:
                verts, faces, _, _ = measure.marching_cubes(numerical, level)
                # 単純な推定: 三角形の数に比例
                volume = len(faces)
                volumes.append(volume)
            except:
                volumes.append(0)
        
        # テーブルデータを作成
        table_data = []
        for i, level in enumerate(num_levels):
            level_pct = (level - num_min) / (num_max - num_min) * 100 if num_max > num_min else 0
            table_data.append([
                f"{level:.2e}", 
                f"{level_pct:.1f}%", 
                f"{volumes[i]}" if volumes[i] > 0 else "N/A"
            ])
        
        # テーブルを作成
        columns = ['Level', '% of Range', 'Est. Volume']
        ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colWidths=[0.33, 0.33, 0.33]
        )
        
        # 値の範囲比較情報を追加
        value_range_ratio = (num_max - num_min) / (ex_max - ex_min) if (ex_max - ex_min) > 0 else 0
        ax.text(0.5, 0.1, f"Value Range Ratio (Num/Exact): {value_range_ratio:.2f}",
               ha='center', transform=ax.transAxes, color='blue')
    
    def _estimate_convergence_order(self, errors):
        """誤差から収束次数を推定"""
        # 標準的な次数の推定値
        if max(errors) < 1e-10:
            return 6  # 非常に高精度
        elif max(errors) < 1e-8:
            return 4  # 高精度
        elif max(errors) < 1e-6:
            return 2  # 中程度の精度
        else:
            return 1  # 低精度
    
    def compare_all_functions_errors(self, results_summary, grid_size, prefix="", save=True, dpi=150, show=False):
        """すべてのテスト関数の誤差を比較するグラフを生成"""
        func_names = list(results_summary.keys())
        error_types = self.get_error_types()
        
        # より効率的なレイアウト（2行5列のサブプロットを使用）
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.flatten()
        
        for i, (error_type, ax) in enumerate(zip(error_types, axes)):
            if i < len(error_types):
                # 各関数の対応する誤差を取得
                original_errors = [results_summary[name][i] for name in func_names]
                
                # 対数スケール用に0値を置き換え
                plot_errors = []
                for err in original_errors:
                    if err == 0.0:
                        plot_errors.append(self.min_log_value)
                    else:
                        plot_errors.append(err)
                
                # バーチャート生成
                bars = ax.bar(np.arange(len(func_names)), plot_errors)
                ax.set_yscale("log")
                ax.set_title(f"{error_type}")
                ax.set_xticks(np.arange(len(func_names)))
                ax.set_xticklabels(func_names, rotation=45, ha="right", fontsize=8)
                
                # グリッド追加
                ax.grid(True, which="major", linestyle="--", alpha=0.5)
                
                # 値のアノテーション
                for j, (bar, err) in enumerate(zip(bars, original_errors)):
                    label = "0.0" if err == 0.0 else f"{err:.1e}"
                    ax.annotate(
                        label, xy=(j, plot_errors[j] * 1.1), 
                        ha="center", va="bottom", fontsize=7, rotation=90
                    )
        
        # 全体のタイトルとレイアウト調整
        plt.suptitle(f"Error Comparison ({grid_size}x{grid_size}x{grid_size} points)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存と表示
        if save:
            filename = f"{self.output_dir}/{prefix}_function_comparison_{grid_size}x{grid_size}x{grid_size}.png"
            if not prefix:
                filename = f"{self.output_dir}/function_comparison_{grid_size}x{grid_size}x{grid_size}.png"
            plt.savefig(filename, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filename
    
    def visualize_grid_convergence(self, function_name, grid_sizes, results, prefix="", save=True, show=False, dpi=150):
        """グリッド収束性のグラフを生成（効率化）"""
        # 2行5列のレイアウトを使用
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.flatten()
        solution_names = self.get_error_types()
        
        # グリッド間隔
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        # 収束次数の推定値を保存
        convergence_orders = []
        
        for i, (ax, name) in enumerate(zip(axes, solution_names)):
            if i < len(solution_names):
                # 各グリッドサイズでの誤差を取得
                errors = [results[n][i] for n in grid_sizes]
                
                # 誤差がすべて0の場合は特別処理
                if all(err == 0 for err in errors):
                    ax.text(0.5, 0.5, "All errors are zero", 
                            ha='center', va='center', transform=ax.transAxes)
                    convergence_orders.append(float('inf'))
                    continue
                
                # 0値を置き換え
                plot_errors = [max(err, self.min_log_value) for err in errors]
                
                # 対数-対数プロット
                ax.loglog(grid_spacings, plot_errors, 'o-', label=name)
                
                # 収束次数を推定
                if len(grid_spacings) >= 2 and all(err > 0 for err in plot_errors):
                    # 単純な傾き計算（最後の2点を使用）
                    order = (np.log(plot_errors[-2]) - np.log(plot_errors[-1])) / \
                           (np.log(grid_spacings[-2]) - np.log(grid_spacings[-1]))
                    convergence_orders.append(order)
                    
                    # 推定した収束次数を表示
                    ax.text(0.05, 0.05, f"Order: {order:.2f}", 
                           transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
                else:
                    convergence_orders.append(0)
                
                # 収束次数の参照線
                if min(plot_errors) > 0:
                    x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                    # 2次および4次の収束線
                    for order, style in zip([2, 4], ['--', '-.']):
                        scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                        y_ref = scale * x_ref ** order
                        ax.loglog(x_ref, y_ref, style, label=f'O(h^{order})')
                
                ax.set_title(f"{name}")
                ax.set_xlabel('h')
                ax.set_ylabel('Error')
                ax.grid(True)
                ax.legend(fontsize=8)
        
        # 平均収束次数の計算
        valid_orders = [o for o in convergence_orders if o != float('inf') and o > 0]
        avg_order = np.mean(valid_orders) if valid_orders else 0
        
        # タイトルとレイアウト調整
        plt.suptitle(f"Grid Convergence: {function_name} (Avg. Order: {avg_order:.2f})")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存と表示
        if save:
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_convergence.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_convergence.png"
            plt.savefig(filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def get_dimension_label(self) -> str:
        """次元ラベルを返す"""
        return "3D"
    
    def get_error_types(self) -> List[str]:
        """エラータイプのリストを返す"""
        return ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]