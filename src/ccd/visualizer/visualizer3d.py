"""
高精度コンパクト差分法 (CCD) の3次元結果可視化

このモジュールは、3次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。効率的な可視化と直感的な結果表示に重点を置いています。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from skimage import measure
from typing import List, Optional, Tuple, Dict

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer3D(BaseVisualizer):
    """CCDソルバーの3D結果を可視化するクラス"""
    
    def __init__(self, output_dir="results_3d"):
        """初期化"""
        super().__init__(output_dir)
    
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
        3Dソリューションを可視化（統合ダッシュボード形式）
        
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
        
        # NumPy配列に変換（メイン成分のみ）
        X, Y, Z = self._to_numpy(grid.get_points())
        psi = self._to_numpy(numerical[0])
        psi_ex = self._to_numpy(exact[0])
        error = self._to_numpy(np.abs(psi - psi_ex))
        
        # 成分名のリスト
        component_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
        
        # ダッシュボード作成 (2x2レイアウト)
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 2, figure=fig)
        
        # メインタイトル
        fig.suptitle(f"{function_name} Function Analysis ({nx}x{ny}x{nz} points)", 
                    fontsize=16, y=0.98)
        
        # 1. 3面断面図 (左上)
        ax_slices = fig.add_subplot(gs[0, 0])
        self._plot_orthogonal_slices(ax_slices, grid, psi, mid_x, mid_y, mid_z)
        
        # 2. 3D可視化 (右上)
        ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
        self._plot_3d_visualization(ax_3d, grid, psi, error, errors[0])
        
        # 3. 誤差分布 (左下)
        ax_error = fig.add_subplot(gs[1, 0])
        self._plot_error_distribution(ax_error, grid, error, mid_z)
        
        # 4. 誤差サマリー (右下)
        ax_summary = fig.add_subplot(gs[1, 1])
        self._plot_error_summary(ax_summary, component_names, errors, function_name)
        
        # レイアウト調整
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存/表示処理
        if save:
            filepath = self.generate_filename(function_name, nx, ny, nz, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
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
        im = ax.imshow(grid_img, cmap='viridis', origin='lower')
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
        """3D可視化（等値面とスライス）を効率的に生成"""
        X, Y, Z = grid.get_points()
        x = self._to_numpy(X[:, 0, 0])
        y = self._to_numpy(Y[0, :, 0])
        z = self._to_numpy(Z[0, 0, :])
        
        # グリッドの中心インデックス
        mid_x, mid_y, mid_z = len(x)//2, len(y)//2, len(z)//2
        
        # データの最小・最大値
        vmin, vmax = np.min(data), np.max(data)
        
        # 等値面レベルの計算（値の範囲内で2つだけ）
        levels = [
            vmin + (vmax - vmin) * 0.3,
            vmin + (vmax - vmin) * 0.7
        ]
        
        # メッシュグリッド作成
        xx, yy = np.meshgrid(x, y)
        
        # 中央断面をプロット
        x_mesh, z_mesh = np.meshgrid(x, z)
        y_plane = np.ones_like(x_mesh) * y[mid_y]
        ax.plot_surface(
            x_mesh, y_plane, z_mesh, 
            facecolors=plt.cm.viridis((data[:, mid_y, :].T - vmin) / (vmax - vmin)),
            alpha=0.6, shade=False
        )
        
        # 等値面を描画（最大2つに制限）
        for level in levels:
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
            except:
                continue
        
        # 軸設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Visualization (Max Error: {max_error:.2e})")
        
        # 表示角度の設定
        ax.view_init(elev=30, azim=45)
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7)
    
    def _plot_error_distribution(self, ax, grid, error_data, mid_z):
        """誤差分布ヒートマップを表示"""
        # Z方向中央の誤差断面を表示
        error_slice = error_data[:, :, mid_z]
        
        # 対数スケールで表示（小さな値を持つ場合は対応）
        min_error = np.min(error_data[error_data > 0]) if np.any(error_data > 0) else 1e-10
        
        im = ax.imshow(
            error_slice.T, 
            norm=LogNorm(vmin=min_error, vmax=np.max(error_data)),
            cmap='hot', origin='lower'
        )
        
        plt.colorbar(im, ax=ax, label='Error (log scale)')
        ax.set_title(f'Error Distribution at z={mid_z}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _plot_error_summary(self, ax, component_names, errors, function_name):
        """誤差サマリープロットを描画"""
        # 水平バーチャートとして表示（読みやすさ向上）
        y_pos = np.arange(len(component_names))
        
        # 0値対応のために小さな値を設定
        plot_errors = np.array(errors).copy()
        plot_errors[plot_errors == 0] = self.min_log_value
        
        # 横向きバーチャート
        bars = ax.barh(y_pos, plot_errors)
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
    
    def compare_all_functions_errors(self, results_summary, grid_size, prefix="", save=True, dpi=150, show=False):
        """すべてのテスト関数の誤差を比較するグラフを生成"""
        func_names = list(results_summary.keys())
        error_types = self.get_error_types()
        
        # より効率的なレイアウト（2行5列のサブプロットを使用）
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.flatten()
        
        for i, (error_type, ax) in enumerate(zip(error_types, axes)):
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
        # 同様に2行5列のレイアウトを使用
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.flatten()
        solution_names = self.get_error_types()
        
        # グリッド間隔
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        for i, (ax, name) in enumerate(zip(axes, solution_names)):
            if i < len(solution_names):
                # 各グリッドサイズでの誤差を取得
                errors = [results[n][i] for n in grid_sizes]
                
                # 対数プロット
                if all(err == 0 for err in errors):
                    ax.text(0.5, 0.5, "All errors are zero", 
                            ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # 0値を置き換え
                plot_errors = [max(err, self.min_log_value) for err in errors]
                
                # 対数-対数プロット
                ax.loglog(grid_spacings, plot_errors, 'o-', label=name)
                
                # 収束次数の参照線（単純化）
                if min(plot_errors) > 0:
                    x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                    # 2次および4次の収束線のみ表示
                    for order, style in zip([2, 4], ['--', '-.']):
                        scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                        y_ref = scale * x_ref ** order
                        ax.loglog(x_ref, y_ref, style, label=f'O(h^{order})')
                
                ax.set_title(f"{name}")
                ax.set_xlabel('h')
                ax.set_ylabel('Error')
                ax.grid(True)
                ax.legend(fontsize=8)
        
        # タイトルとレイアウト調整
        plt.suptitle(f"Grid Convergence: {function_name}")
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