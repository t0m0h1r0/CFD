"""
高精度コンパクト差分法 (CCD) の3次元結果可視化

このモジュールは、3次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。scikit-imageを用いた等値面表示機能を追加し、
すべての可視化結果を1枚の画像にまとめて出力します。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from skimage import measure
from typing import List, Optional, Tuple, Union

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
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=180):
        """
        3Dソリューションを可視化し、すべての結果を1枚の画像にまとめる
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト [psi, psi_x, psi_y, psi_z, psi_xx, psi_yy, psi_zz, psi_xxx, psi_yyy, psi_zzz]
            exact: 厳密解のリスト
            errors: 誤差リスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            保存に成功したかどうかのブール値
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # CuPy配列をNumPyに変換
        X, Y, Z = grid.get_points()
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        Z_np = self._to_numpy(Z)
        
        # 中心断面のインデックス
        mid_x = nx_points // 2
        mid_y = ny_points // 2
        mid_z = nz_points // 2
        
        # 可視化するソリューションのリスト
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
        
        # 主要解成分だけを可視化
        vis_components = [0, 1, 4, 7]  # ψ, ψ_x, ψ_xx, ψ_xxx
        
        # 1枚の大きな図を作成 (4 rows x 3 columns layout)
        fig = plt.figure(figsize=(24, 30))
        
        # GridSpecを使用して複雑なレイアウトを作成
        gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.7])
        
        # タイトル
        fig.suptitle(f"{function_name} Function Analysis ({nx_points}x{ny_points}x{nz_points} points)", 
                    fontsize=24, y=0.98)
        
        # NumPy配列に変換（主要成分のみ）
        psi_np = self._to_numpy(numerical[0])
        psi_ex_np = self._to_numpy(exact[0])
        error_np = self._to_numpy(np.abs(psi_np - psi_ex_np))
        
        # 1行目: 断面図 (x, y, z)
        plane_axes = []
        for col, (idx_name, idx) in enumerate([('x', mid_x), ('y', mid_y), ('z', mid_z)]):
            ax = fig.add_subplot(gs[0, col])
            plane_axes.append(ax)
            
            # 平面ごとの断面を抽出 (ψ成分のみ)
            if idx_name == 'x':
                num_slice = psi_np[idx, :, :]
                xlabel, ylabel = 'Y', 'Z'
                x_data, y_data = Y_np[idx, :, 0], Z_np[idx, 0, :]
                plane_title = f"X={idx} Plane (ψ)"
            elif idx_name == 'y':
                num_slice = psi_np[:, idx, :]
                xlabel, ylabel = 'X', 'Z'
                x_data, y_data = X_np[:, idx, 0], Z_np[0, idx, :]
                plane_title = f"Y={idx} Plane (ψ)"
            else:  # idx_name == 'z'
                num_slice = psi_np[:, :, idx]
                xlabel, ylabel = 'X', 'Y'
                x_data, y_data = X_np[:, 0, idx], Y_np[0, :, idx]
                plane_title = f"Z={idx} Plane (ψ)"
                
            xx, yy = np.meshgrid(x_data, y_data, indexing='ij')
            
            # コンター図として描画
            im = ax.contourf(xx, yy, num_slice, 50, cmap='viridis')
            ax.set_title(plane_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax)
        
        # 2行目: 誤差および異なる解成分の断面
        for col, comp_idx in enumerate(vis_components[1:]):  # ψ_x, ψ_xx, ψ_xxx
            ax = fig.add_subplot(gs[1, col])
            name = solution_names[comp_idx]
            num = numerical[comp_idx]
            ex = exact[comp_idx]
            err = errors[comp_idx]
            
            # NumPy配列に変換
            num_np = self._to_numpy(num)
            
            # Z平面の断面を抽出
            num_slice = num_np[:, :, mid_z]
            x_data, y_data = X_np[:, 0, mid_z], Y_np[0, :, mid_z]
            xx, yy = np.meshgrid(x_data, y_data, indexing='ij')
            
            # コンター図として描画
            im = ax.contourf(xx, yy, num_slice, 50, cmap='viridis')
            ax.set_title(f"{name} at Z={mid_z} (Error: {err:.2e})")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        # 3行目: 3D可視化
        # 3D等値面
        ax_iso = fig.add_subplot(gs[2, 0], projection='3d')
        self._plot_isosurface(ax_iso, grid, psi_np, function_name, errors[0])
        
        # 3D断面スライス
        ax_slice = fig.add_subplot(gs[2, 1], projection='3d')
        self._plot_3d_slices_improved(ax_slice, grid, psi_np, function_name, errors[0])
        
        # 誤差等値面
        ax_error = fig.add_subplot(gs[2, 2], projection='3d')
        self._plot_error_isosurface(ax_error, grid, psi_np, psi_ex_np, error_np, function_name, errors[0])
        
        # 4行目: 誤差サマリー (水平バーグラフ)
        ax_summary = fig.add_subplot(gs[3, :])
        self._plot_error_summary(ax_summary, solution_names, errors, function_name)
        
        # 余白を調整
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # タイトル用の余白を確保
        
        # 保存ファイル名
        if save:
            filepath = self.generate_filename(function_name, nx_points, ny_points, nz_points, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Combined visualization saved to: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def _plot_isosurface(self, ax, grid, data_np, function_name, error):
        """
        等値面を描画
        
        Args:
            ax: 描画するAxis
            grid: グリッドオブジェクト
            data_np: 描画データ (NumPy配列)
            function_name: 関数名
            error: 誤差値
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # メッシュグリッド
        X, Y, Z = grid.get_points()
        x_flat = self._to_numpy(X[:, 0, 0])
        y_flat = self._to_numpy(Y[0, :, 0])
        z_flat = self._to_numpy(Z[0, 0, :])
        
        # 値の範囲
        vmin, vmax = np.min(data_np), np.max(data_np)
        
        # 等値面の値を計算
        n_levels = 3
        iso_values = np.linspace(vmin + (vmax - vmin) * 0.3, vmax - (vmax - vmin) * 0.1, n_levels)
        
        # 各等値面レベルに対してマーチングキューブ法を適用
        for i, iso_val in enumerate(iso_values):
            try:
                # skimageのmarching_cubesを使用して等値面を抽出
                verts, faces, normals, values = measure.marching_cubes(data_np, iso_val)
                
                # 正規化された値に基づいて色を選択
                normalized_val = (iso_val - vmin) / (vmax - vmin)
                
                # 座標をスケーリング
                vert_coords = np.zeros_like(verts)
                vert_coords[:, 0] = x_flat[0] + verts[:, 0] * (x_flat[-1] - x_flat[0]) / (nx_points - 1)
                vert_coords[:, 1] = y_flat[0] + verts[:, 1] * (y_flat[-1] - y_flat[0]) / (ny_points - 1)
                vert_coords[:, 2] = z_flat[0] + verts[:, 2] * (z_flat[-1] - z_flat[0]) / (nz_points - 1)
                
                # 等値面の描画
                mesh = ax.plot_trisurf(
                    vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2],
                    triangles=faces,
                    color=plt.cm.viridis(normalized_val),
                    alpha=0.6
                )
            except:
                continue
        
        # 軸ラベル
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Isosurfaces (ψ) - Error: {error:.2e}")
        
        # 軸範囲を明示的に設定
        ax.set_xlim(x_flat[0], x_flat[-1])
        ax.set_ylim(y_flat[0], y_flat[-1])
        ax.set_zlim(z_flat[0], z_flat[-1])
        
        # 視点を設定
        ax.view_init(elev=30, azim=-60)
        
        # カラーバー用のスカラーマッパブル
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7)
    
    def _plot_3d_slices_improved(self, ax, grid, data_np, function_name, error):
        """
        3D断面スライスを描画（改良版）- 完全に直交するスライスを正確に描画
        
        Args:
            ax: 描画するAxis
            grid: グリッドオブジェクト
            data_np: 描画データ (NumPy配列)
            function_name: 関数名
            error: 誤差値
        """
        # 中心断面のインデックス
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        mid_x = nx_points // 2
        mid_y = ny_points // 2
        mid_z = nz_points // 2
        
        # グリッド座標の取得
        X, Y, Z = grid.get_points()
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        Z_np = self._to_numpy(Z)
        
        # 範囲を明示的に取得
        x_min, x_max = X_np[0, 0, 0], X_np[-1, 0, 0]
        y_min, y_max = Y_np[0, 0, 0], Y_np[0, -1, 0]
        z_min, z_max = Z_np[0, 0, 0], Z_np[0, 0, -1]
        
        # 中心位置の値
        mid_x_val = X_np[mid_x, 0, 0]
        mid_y_val = Y_np[0, mid_y, 0]
        mid_z_val = Z_np[0, 0, mid_z]
        
        # カラーマッピングのための正規化
        vmin, vmax = np.min(data_np), np.max(data_np)
        norm = plt.Normalize(vmin, vmax)
        
        # 平面の頂点座標
        # XY平面 (Z = mid_z_val)
        xy_vertices = np.array([
            [x_min, y_min, mid_z_val],
            [x_max, y_min, mid_z_val],
            [x_max, y_max, mid_z_val],
            [x_min, y_max, mid_z_val]
        ])
        
        # XZ平面 (Y = mid_y_val)
        xz_vertices = np.array([
            [x_min, mid_y_val, z_min],
            [x_max, mid_y_val, z_min],
            [x_max, mid_y_val, z_max],
            [x_min, mid_y_val, z_max]
        ])
        
        # YZ平面 (X = mid_x_val)
        yz_vertices = np.array([
            [mid_x_val, y_min, z_min],
            [mid_x_val, y_max, z_min],
            [mid_x_val, y_max, z_max],
            [mid_x_val, y_min, z_max]
        ])
        
        # 明示的なメッシュグリッド作成 - より精確な直交平面表示のため
        # XY平面のメッシュグリッド (Z = mid_z_val)
        xy_grid_x, xy_grid_y = np.meshgrid(X_np[:, 0, 0], Y_np[0, :, 0])
        xy_grid_z = np.ones_like(xy_grid_x) * mid_z_val
        xy_grid_values = data_np[:, :, mid_z].T
        
        # XZ平面のメッシュグリッド (Y = mid_y_val)
        xz_grid_x, xz_grid_z = np.meshgrid(X_np[:, 0, 0], Z_np[0, 0, :])
        xz_grid_y = np.ones_like(xz_grid_x) * mid_y_val
        xz_grid_values = data_np[:, mid_y, :].T
        
        # YZ平面のメッシュグリッド (X = mid_x_val)
        yz_grid_y, yz_grid_z = np.meshgrid(Y_np[0, :, 0], Z_np[0, 0, :])
        yz_grid_x = np.ones_like(yz_grid_y) * mid_x_val
        yz_grid_values = data_np[mid_x, :, :].T
        
        # 各スライスの描画
        # XY平面 (Z = mid_z_val)
        surf_xy = ax.plot_surface(
            xy_grid_x, xy_grid_y, xy_grid_z,
            facecolors=plt.cm.viridis(norm(xy_grid_values)),
            alpha=0.8, rstride=1, cstride=1, shade=False
        )
        
        # XZ平面 (Y = mid_y_val)
        surf_xz = ax.plot_surface(
            xz_grid_x, xz_grid_y, xz_grid_z,
            facecolors=plt.cm.viridis(norm(xz_grid_values)),
            alpha=0.8, rstride=1, cstride=1, shade=False
        )
        
        # YZ平面 (X = mid_x_val)
        surf_yz = ax.plot_surface(
            yz_grid_x, yz_grid_y, yz_grid_z,
            facecolors=plt.cm.viridis(norm(yz_grid_values)),
            alpha=0.8, rstride=1, cstride=1, shade=False
        )
        
        # 各平面の枠線を明示的に描画して直交性を強調
        # XY平面の枠線
        ax.plot(xy_vertices[[0, 1, 2, 3, 0], 0], xy_vertices[[0, 1, 2, 3, 0], 1], xy_vertices[[0, 1, 2, 3, 0], 2], 'k-', lw=1.5)
        
        # XZ平面の枠線
        ax.plot(xz_vertices[[0, 1, 2, 3, 0], 0], xz_vertices[[0, 1, 2, 3, 0], 1], xz_vertices[[0, 1, 2, 3, 0], 2], 'k-', lw=1.5)
        
        # YZ平面の枠線
        ax.plot(yz_vertices[[0, 1, 2, 3, 0], 0], yz_vertices[[0, 1, 2, 3, 0], 1], yz_vertices[[0, 1, 2, 3, 0], 2], 'k-', lw=1.5)
        
        # 軸ラベル
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Orthogonal Slices (ψ) - Error: {error:.2e}")
        
        # 軸範囲を明示的に設定
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # 正射影（アスペクト比1:1:1）で表示して直交性を視覚的に正確に
        ax.set_box_aspect([1, 1, 1])
        
        # 視点の設定 - 直交性がはっきりと見えるアングル
        ax.view_init(elev=30, azim=45)
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7)
    
    def _plot_error_isosurface(self, ax, grid, numerical_np, exact_np, error_np, function_name, max_error):
        """
        誤差等値面を描画
        
        Args:
            ax: 描画するAxis
            grid: グリッドオブジェクト
            numerical_np: 数値解 (NumPy配列)
            exact_np: 厳密解 (NumPy配列)
            error_np: 誤差データ (NumPy配列)
            function_name: 関数名
            max_error: 最大誤差値
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # メッシュグリッド
        X, Y, Z = grid.get_points()
        x_flat = self._to_numpy(X[:, 0, 0])
        y_flat = self._to_numpy(Y[0, :, 0])
        z_flat = self._to_numpy(Z[0, 0, :])
        
        # 誤差の範囲（非常に小さい値は無視）
        err_min = np.max([np.min(error_np[error_np > 0]), 1e-15])
        err_max = np.max(error_np)
        
        # 対数スケールで等値面の値を計算
        n_levels = 3
        if err_max / err_min > 100:
            iso_values = np.logspace(np.log10(err_min), np.log10(err_max * 0.7), n_levels)
        else:
            iso_values = np.linspace(err_min, err_max * 0.7, n_levels)
        
        # 各等値面レベルに対してマーチングキューブ法を適用
        for i, iso_val in enumerate(iso_values):
            try:
                # skimageのmarching_cubesを使用して等値面を抽出
                verts, faces, normals, values = measure.marching_cubes(error_np, iso_val)
                
                # 正規化された値に基づいて色を選択
                if err_max / err_min > 100:
                    normalized_val = (np.log10(iso_val) - np.log10(err_min)) / (np.log10(err_max) - np.log10(err_min))
                else:
                    normalized_val = (iso_val - err_min) / (err_max - err_min)
                
                # 座標をスケーリング
                vert_coords = np.zeros_like(verts)
                vert_coords[:, 0] = x_flat[0] + verts[:, 0] * (x_flat[-1] - x_flat[0]) / (nx_points - 1)
                vert_coords[:, 1] = y_flat[0] + verts[:, 1] * (y_flat[-1] - y_flat[0]) / (ny_points - 1)
                vert_coords[:, 2] = z_flat[0] + verts[:, 2] * (z_flat[-1] - z_flat[0]) / (nz_points - 1)
                
                # 等値面の描画
                mesh = ax.plot_trisurf(
                    vert_coords[:, 0], vert_coords[:, 1], vert_coords[:, 2],
                    triangles=faces,
                    color=plt.cm.hot(normalized_val),
                    alpha=0.7
                )
            except:
                continue
        
        # 軸ラベル
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Error Isosurfaces (ψ) - Max Error: {max_error:.2e}")
        
        # 軸範囲を明示的に設定
        ax.set_xlim(x_flat[0], x_flat[-1])
        ax.set_ylim(y_flat[0], y_flat[-1])
        ax.set_zlim(z_flat[0], z_flat[-1])
        
        # 視点を設定
        ax.view_init(elev=30, azim=-60)
        
        # カラーバー用のスカラーマッパブル
        if err_max / err_min > 100:
            norm = LogNorm(vmin=err_min, vmax=err_max)
        else:
            norm = plt.Normalize(vmin=err_min, vmax=err_max)
            
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7)
    
    def _plot_error_summary(self, ax, solution_names, errors, function_name):
        """
        誤差サマリープロットを描画
        
        Args:
            ax: 描画するAxis
            solution_names: 解成分の名前リスト
            errors: 誤差リスト
            function_name: 関数名
        """
        x_pos = np.arange(len(solution_names))
        bars = ax.bar(x_pos, errors)
        
        # 対数スケール（ゼロを小さな値に置き換え）
        error_values = errors.copy()
        for i, err in enumerate(error_values):
            if err == 0:
                error_values[i] = self.min_log_value
        
        ax.set_yscale('log')
        ax.set_title(f"Error Summary for {function_name}")
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
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
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