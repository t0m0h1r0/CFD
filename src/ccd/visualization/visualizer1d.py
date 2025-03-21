"""
1次元可視化クラスモジュール

このモジュールは、CCDソルバーの1次元結果を可視化するクラスを提供します。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any

from .base_visualizer import BaseVisualizer
from ..grid.grid1d import Grid1D

class Visualizer1D(BaseVisualizer):
    """CCDソルバーの1次元結果を可視化するクラス"""
    
    def __init__(self, output_dir: str = "results_1d"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス（デフォルト: "results_1d"）
        """
        super().__init__(output_dir)
    
    def generate_filename(self, func_name: str, n_points: int, prefix: str = "") -> str:
        """
        保存ファイル名を生成
        
        Args:
            func_name: 関数名
            n_points: 格子点数
            prefix: ファイル名の接頭辞
            
        Returns:
            ファイルパス
        """
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{n_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{n_points}_points.png"
    
    def visualize_solution(self, 
                         grid: Grid1D, 
                         function_name: str, 
                         numerical: List[np.ndarray], 
                         exact: List[np.ndarray], 
                         errors: List[float], 
                         prefix: str = "", 
                         save: bool = True, 
                         show: bool = False, 
                         dpi: int = 150) -> bool:
        """
        1次元解を可視化
        
        Args:
            grid: Grid1D オブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト [psi, psi_prime, psi_second, psi_third]
            exact: 厳密解のリスト
            errors: 誤差のリスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            成功したかどうかのブール値
        """
        x = grid.get_points()
        n_points = grid.n_points
        
        # NumPy配列に変換
        x_np = self._to_numpy(x)
        
        # グラフ作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]
        
        for i, ax in enumerate(axes.flat):
            exact_data = self._to_numpy(exact[i])
            num_data = self._to_numpy(numerical[i])
            
            ax.plot(x_np, exact_data, "b-", label="Exact")
            ax.plot(x_np, num_data, "r--", label="Numerical")
            ax.set_title(f"{titles[i]} (error: {errors[i]:.2e})")
            ax.legend()
            ax.grid(True)
        
        plt.suptitle(f"Results for {function_name} function ({n_points} points)")
        plt.tight_layout()
        
        # ファイル保存
        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, n_points, prefix)
            plt.savefig(filepath, dpi=dpi)
        
        # 表示/クローズ
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
    
    def compare_all_functions_errors(self, 
                                   results_summary: Dict[str, List[float]], 
                                   grid_size: Union[int, Tuple[int, ...]] = None, 
                                   prefix: str = "", 
                                   dpi: int = 150, 
                                   show: bool = False) -> str:
        """
        すべてのテスト関数の誤差を比較するグラフを生成
        
        Args:
            results_summary: テスト関数ごとの誤差リスト
            grid_size: グリッドサイズ（オプション）
            prefix: ファイル名の接頭辞
            dpi: 保存する図のDPI
            show: 図を表示するかどうか
            
        Returns:
            保存したファイルのパス
        """
        func_names = list(results_summary.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        error_types = ["ψ", "ψ'", "ψ''", "ψ'''"]
        
        for i, (ax, error_type) in enumerate(zip(axes.flat, error_types)):
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
        
        plt.tight_layout()
        
        # ファイル保存
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison.png"
        if not prefix:
            filename = f"{self.output_dir}/all_functions_comparison.png"
            
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]
        
        # 格子間隔を計算
        grid_spacings = [1.0 / (n - 1) for n in grid_sizes]
        
        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            errors = [results[n][i] for n in grid_sizes]
            
            # エラーが全て0の場合はグラフを描画しない
            if all(err == 0 for err in errors):
                ax.text(0.5, 0.5, f"All errors are 0 for {title}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                continue
                
            # エラーが0の場合は小さな値に置き換え
            plot_errors = [max(err, self.min_log_value) for err in errors]
            
            # 対数-対数プロット
            ax.loglog(grid_spacings, plot_errors, 'o-', label=f"Error ({title})")
            
            # 参照傾き線を描画
            x_range = [min(grid_spacings), max(grid_spacings)]
            for order, style, color in zip([2, 4, 6], ['--', '-.', ':'], ['r', 'g', 'b']):
                # 参照線の位置を調整（最後のポイントに合わせる）
                ref_y0 = plot_errors[-1] * (x_range[-1] / x_range[-1]) ** order
                y_ref = [ref_y0 * (x / x_range[-1]) ** order for x in x_range]
                ax.loglog(x_range, y_ref, style, color=color, label=f"O(h^{order})")
            
            ax.set_title(f"{title} Error Convergence")
            ax.set_xlabel("Grid Spacing (h)")
            ax.set_ylabel("Error")
            ax.grid(True)
            ax.legend()
        
        plt.suptitle(f"Grid Convergence for {function_name}")
        plt.tight_layout()
        
        # ファイル保存
        filepath = ""
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
