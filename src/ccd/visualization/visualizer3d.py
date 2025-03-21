"""
3次元可視化クラスモジュール

このモジュールは、CCDソルバーの3次元結果を可視化するクラスを提供します。
3次元座標系では、xは水平方向、yは奥行き方向、zは高さ方向として可視化されます。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional, Union, Any

from .base_visualizer import BaseVisualizer
from ..grid.grid3d import Grid3D

class Visualizer3D(BaseVisualizer):
    """
    CCDソルバーの3次元結果を可視化するクラス
    
    3次元座標系では、
    - xは水平方向（左右）
    - yは奥行き方向（前後）
    - zは高さ方向（上下）
    として可視化されます。
    """
    
    def __init__(self, output_dir: str = "results_3d"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス（デフォルト: "results_3d"）
        """
        super().__init__(output_dir)
    
    def generate_filename(self, func_name: str, nx_points: int, ny_points: int, nz_points: int, prefix: str = "") -> str:
        """
        保存ファイル名を生成
        
        Args:
            func_name: 関数名
            nx_points: x方向の格子点数
            ny_points: y方向の格子点数
            nz_points: z方向の格子点数
            prefix: ファイル名の接頭辞
            
        Returns:
            ファイルパス
        """
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{nx_points}x{ny_points}x{nz_points}_points.png"
    
    def visualize_solution(self, 
                         grid: Grid3D, 
                         function_name: str, 
                         numerical: List[np.ndarray], 
                         exact: List[np.ndarray], 
                         errors: List[float], 
                         prefix: str = "", 
                         save: bool = True, 
                         show: bool = False, 
                         dpi: int = 150) -> bool:
        """
        3次元解を可視化（スライス図として、主要な解を複数のスライスで表示）
        
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
            
        Returns:
            成功したかどうかのブール値
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # 可視化する主要なソリューションの選択
        # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]のうち主要な成分
        solution_indices = [0, 1, 4, 7, 2, 5, 8]  # ψ, ψ_x, ψ_y, ψ_z, ψ_xx, ψ_yy, ψ_zz
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz"]
        
        # カラーマップ
        cmap_sol = 'viridis'
        cmap_err = 'hot'
        
        # 各主要成分ごとにスライスを表示
        for sol_idx, sol_name in zip(solution_indices, solution_names):
            num = self._to_numpy(numerical[sol_idx])
            ex = self._to_numpy(exact[sol_idx])
            error = np.abs(num - ex)
            
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
            
            # ファイル保存
            if save:
                filepath = self.generate_filename(f"{function_name}_{sol_name}", nx_points, ny_points, nz_points, prefix)
                plt.savefig(filepath, dpi=dpi)
            
            # 表示/クローズ
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        # 3D表示（メインの解のみ）
        self._create_3d_visualization(grid, function_name, numerical[0], exact[0], errors[0], prefix, save, show, dpi)
        
        # 誤差サマリー図を作成
        self._create_error_summary(function_name, solution_names, errors[:7], prefix, save, show, dpi)
        
        return True
    
    def _create_3d_visualization(self, 
                               grid: Grid3D, 
                               function_name: str, 
                               numerical: np.ndarray, 
                               exact: np.ndarray, 
                               error_val: float, 
                               prefix: str = "", 
                               save: bool = True, 
                               show: bool = False, 
                               dpi: int = 150) -> None:
        """
        3D可視化を作成
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解
            exact: 厳密解
            error_val: 誤差値
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # NumPy配列に変換
        numerical = self._to_numpy(numerical)
        exact = self._to_numpy(exact)
        error = np.abs(numerical - exact)
        
        # 座標系の設定（x: 水平方向, y: 奥行き方向, z: 高さ方向）
        x = np.linspace(grid.x_min, grid.x_max, nx_points)
        y = np.linspace(grid.y_min, grid.y_max, ny_points)
        z = np.linspace(grid.z_min, grid.z_max, nz_points)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 3Dコンター図（スライス）の可視化
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"{function_name}: 3D Visualization ({nx_points}x{ny_points}x{nz_points} points)", fontsize=14)
        
        # 1. 3D散布図
        ax1 = fig.add_subplot(231, projection='3d')
        # サンプリング（全点ではなく間引いて表示）
        sample_step = max(1, nx_points // 10)  # 表示点数を減らす
        p1 = ax1.scatter(X[::sample_step, ::sample_step, ::sample_step], 
                       Y[::sample_step, ::sample_step, ::sample_step], 
                       Z[::sample_step, ::sample_step, ::sample_step], 
                       c=numerical[::sample_step, ::sample_step, ::sample_step], 
                       cmap='viridis', alpha=0.5, s=5)
        ax1.set_title("Numerical (Scatter)")
        ax1.set_xlabel('X (Horizontal)')
        ax1.set_ylabel('Y (Depth)')
        ax1.set_zlabel('Z (Height)')
        fig.colorbar(p1, ax=ax1, shrink=0.5)
        
        # 2. X=中央でのYZスライス
        ax2 = fig.add_subplot(232, projection='3d')
        slice_x = nx_points // 2
        Y_slice, Z_slice = np.meshgrid(y, z, indexing='ij')
        surf2 = ax2.plot_surface(
            np.ones_like(Y_slice) * x[slice_x],
            Y_slice,
            Z_slice,
            facecolors=plt.cm.viridis(plt.Normalize()(numerical[slice_x, :, :])),
            alpha=0.7
        )
        ax2.set_title(f"X-Slice (x={x[slice_x]:.2f})")
        ax2.set_xlabel('X (Horizontal)')
        ax2.set_ylabel('Y (Depth)')
        ax2.set_zlabel('Z (Height)')
        
        # 3. Y=中央でのXZスライス
        ax3 = fig.add_subplot(233, projection='3d')
        slice_y = ny_points // 2
        X_slice, Z_slice = np.meshgrid(x, z, indexing='ij')
        surf3 = ax3.plot_surface(
            X_slice,
            np.ones_like(X_slice) * y[slice_y],
            Z_slice,
            facecolors=plt.cm.viridis(plt.Normalize()(numerical[:, slice_y, :])),
            alpha=0.7
        )
        ax3.set_title(f"Y-Slice (y={y[slice_y]:.2f})")
        ax3.set_xlabel('X (Horizontal)')
        ax3.set_ylabel('Y (Depth)')
        ax3.set_zlabel('Z (Height)')
        
        # 4. Z=中央でのXYスライス
        ax4 = fig.add_subplot(234, projection='3d')
        slice_z = nz_points // 2
        X_slice, Y_slice = np.meshgrid(x, y, indexing='ij')
        surf4 = ax4.plot_surface(
            X_slice,
            Y_slice,
            np.ones_like(X_slice) * z[slice_z],
            facecolors=plt.cm.viridis(plt.Normalize()(numerical[:, :, slice_z])),
            alpha=0.7
        )
        ax4.set_title(f"Z-Slice (z={z[slice_z]:.2f})")
        ax4.set_xlabel('X (Horizontal)')
        ax4.set_ylabel('Y (Depth)')
        ax4.set_zlabel('Z (Height)')
        
        # 5. コンター表示（2次元）- XY平面
        ax5 = fig.add_subplot(235)
        cont5 = ax5.contourf(X[:, :, slice_z], Y[:, :, slice_z], numerical[:, :, slice_z], 20, cmap='viridis')
        ax5.set_title(f"XY Contour (z={z[slice_z]:.2f})")
        ax5.set_xlabel('X (Horizontal)')
        ax5.set_ylabel('Y (Depth)')
        fig.colorbar(cont5, ax=ax5)
        
        # 6. 誤差表示
        ax6 = fig.add_subplot(236)
        cont6 = ax6.contourf(X[:, :, slice_z], Y[:, :, slice_z], error[:, :, slice_z], 20, cmap='hot')
        ax6.set_title(f"Error (Max: {np.max(error):.2e})")
        ax6.set_xlabel('X (Horizontal)')
        ax6.set_ylabel('Y (Depth)')
        fig.colorbar(cont6, ax=ax6)
        
        plt.tight_layout()
        
        # ファイル保存
        if save:
            filepath = self.generate_filename(f"{function_name}_3d", nx_points, ny_points, nz_points, prefix)
            plt.savefig(filepath, dpi=dpi)
        
        # 表示/クローズ
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # ワイヤーフレーム表示 (別のビューとして)
        self._create_wireframe_visualization(grid, function_name, numerical, exact, error, 
                                          prefix, save, show, dpi)
    
    def _create_wireframe_visualization(self, 
                                      grid: Grid3D, 
                                      function_name: str, 
                                      numerical: np.ndarray, 
                                      exact: np.ndarray, 
                                      error: np.ndarray, 
                                      prefix: str = "", 
                                      save: bool = True, 
                                      show: bool = False, 
                                      dpi: int = 150) -> None:
        """
        ワイヤーフレームを使った3D可視化
        
        Args:
            grid: Grid3D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解
            exact: 厳密解
            error: 誤差配列
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
        """
        nx_points, ny_points, nz_points = grid.nx_points, grid.ny_points, grid.nz_points
        
        # 3D表示が大きいとパフォーマンスが落ちるのでダウンサンプリング
        sample_step = max(1, min(nx_points, ny_points, nz_points) // 20)
        
        # 座標系の設定
        x = np.linspace(grid.x_min, grid.x_max, nx_points)[::sample_step]
        y = np.linspace(grid.y_min, grid.y_max, ny_points)[::sample_step]
        z = np.linspace(grid.z_min, grid.z_max, nz_points)[::sample_step]
        
        # ダウンサンプリングしたデータ
        num_sampled = numerical[::sample_step, ::sample_step, ::sample_step]
        ex_sampled = exact[::sample_step, ::sample_step, ::sample_step]
        err_sampled = error[::sample_step, ::sample_step, ::sample_step]
        
        # プロット用のメッシュグリッド
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # ワイヤーフレーム表示（Z方向に複数のスライス）
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"{function_name}: 3D Wireframe ({nx_points}x{ny_points}x{nz_points} points)", fontsize=14)
        
        # 1. 数値解のワイヤーフレーム
        ax1 = fig.add_subplot(131, projection='3d')
        for k, zi in enumerate(z):
            ax1.plot_wireframe(X, Y, np.ones_like(X) * zi, 
                             rstride=1, cstride=1, 
                             linewidth=0.5,
                             color=plt.cm.viridis(plt.Normalize(vmin=np.min(num_sampled), 
                                                              vmax=np.max(num_sampled))(k/len(z))))
        ax1.set_title("Numerical (Wireframe)")
        ax1.set_xlabel('X (Horizontal)')
        ax1.set_ylabel('Y (Depth)')
        ax1.set_zlabel('Z (Height)')
        
        # 2. 厳密解のワイヤーフレーム
        ax2 = fig.add_subplot(132, projection='3d')
        for k, zi in enumerate(z):
            ax2.plot_wireframe(X, Y, np.ones_like(X) * zi, 
                             rstride=1, cstride=1, 
                             linewidth=0.5,
                             color=plt.cm.viridis(plt.Normalize(vmin=np.min(ex_sampled), 
                                                              vmax=np.max(ex_sampled))(k/len(z))))
        ax2.set_title("Exact (Wireframe)")
        ax2.set_xlabel('X (Horizontal)')
        ax2.set_ylabel('Y (Depth)')
        ax2.set_zlabel('Z (Height)')
        
        # 3. 誤差のワイヤーフレーム
        ax3 = fig.add_subplot(133, projection='3d')
        for k, zi in enumerate(z):
            ax3.plot_wireframe(X, Y, np.ones_like(X) * zi, 
                             rstride=1, cstride=1, 
                             linewidth=0.5,
                             color=plt.cm.hot(plt.Normalize(vmin=0, 
                                                          vmax=np.max(err_sampled))(k/len(z))))
        ax3.set_title(f"Error (Max: {np.max(error):.2e})")
        ax3.set_xlabel('X (Horizontal)')
        ax3.set_ylabel('Y (Depth)')
        ax3.set_zlabel('Z (Height)')
        
        plt.tight_layout()
        
        # ファイル保存
        if save:
            wireframe_filepath = self.generate_filename(f"{function_name}_wireframe", 
                                                     nx_points, ny_points, nz_points, prefix)
            plt.savefig(wireframe_filepath, dpi=dpi)
        
        # 表示/クローズ
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def compare_all_functions_errors(self, 
                                   results_summary: Dict[str, List[float]], 
                                   grid_size: Union[int, Tuple[int, ...]], 
                                   prefix: str = "", 
                                   dpi: int = 150, 
                                   show: bool = False) -> str:
        """
        すべてのテスト関数の誤差を比較するグラフを生成
        
        Args:
            results_summary: テスト関数ごとの誤差リスト
            grid_size: グリッドサイズ
            prefix: ファイル名の接頭辞
            dpi: 保存する図のDPI
            show: 図を表示するかどうか
            
        Returns:
            保存したファイルのパス
        """
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
                
                # プロット作成
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
        
        # グリッドサイズを文字列にする
        if isinstance(grid_size, (list, tuple)):
            grid_str = "x".join(map(str, grid_size))
        else:
            grid_str = f"{grid_size}x{grid_size}x{grid_size}"
        
        plt.suptitle(f"Error Comparison for All Functions ({grid_str} points)")
        plt.tight_layout()
        
        # ファイル保存
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison_{grid_str}.png"
        if not prefix:
            filename = f"{self.output_dir}/all_functions_comparison_{grid_str}.png"
            
        plt.savefig(filename, dpi=dpi)
        
        # 表示/クローズ
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filename
    
    def visualize_grid_convergence(self, 
                                 function_name: str, 
                                 grid_sizes: List[int], 
                                 results: Dict[int, List[float]], 
                                 prefix: str = "", 
                                 save: bool = True, 
                                 show: bool = False, 
                                 dpi: int = 150) -> bool:
        """
        グリッド収束性のグラフを生成
        
        Args:
            function_name: テスト関数の名前
            grid_sizes: グリッドサイズのリスト
            results: グリッドサイズごとの誤差リスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            成功したかどうかのブール値
        """
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        solution_names = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz"]
        
        # 格子間隔を計算
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
        
        # ファイル保存
        if save:
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi)
        
        # 表示/クローズ
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
