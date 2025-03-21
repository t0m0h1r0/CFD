"""
高精度コンパクト差分法 (CCD) の結果可視化の基底クラス

このモジュールは、CCDソルバーの計算結果を可視化するための
共通基底クラスと機能を提供します。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List


class BaseVisualizer(ABC):
    """CCDソルバーの結果可視化の基底クラス"""

    def __init__(self, output_dir="results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.min_log_value = 1e-16  # 対数スケール用の最小値

    def generate_filename(self, func_name, n_points, prefix=""):
        """
        ファイル名生成
        
        Args:
            func_name: 関数名
            n_points: 格子点数（文字列または数値）
            prefix: 接頭辞（オプション）
            
        Returns:
            生成されたファイルパス
        """
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{n_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{n_points}_points.png"

    def compare_all_functions_errors(self, results_summary, grid_size=None, prefix="", dpi=150, show=False):
        """
        全テスト関数の誤差比較グラフを生成
        
        Args:
            results_summary: 全関数の結果サマリー辞書
            grid_size: グリッドサイズ
            prefix: 出力ファイルの接頭辞
            dpi: 画像のDPI値
            show: 図を表示するか否か
            
        Returns:
            出力ファイルパス
        """
        func_names = list(results_summary.keys())
        
        # 次元に応じたラベル設定
        self.get_dimension_label()
        error_types = self.get_error_types()
        
        # 図とサブプロットの作成
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        
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
        
        plt.suptitle(f"Error Comparison for All Functions ({grid_size} points)")
        plt.tight_layout()
        
        # 出力ファイル名の生成
        if prefix:
            filename = f"{self.output_dir}/{prefix}_all_functions_comparison"
        else:
            filename = f"{self.output_dir}/all_functions_comparison"
            
        if grid_size:
            filename += f"_{grid_size}"
        
        filename += ".png"
            
        plt.savefig(filename, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filename
        
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_dimension_label(self) -> str:
        """
        次元ラベルを返す
        
        Returns:
            "1D" または "2D"
        """
        pass
    
    @abstractmethod
    def get_error_types(self) -> List[str]:
        """
        エラータイプのリストを返す
        
        Returns:
            エラータイプのリスト（次元に応じて異なる）
        """
        pass
    
    def _to_numpy(self, arr):
        """
        配列をNumPy形式に変換（必要な場合のみ）
        
        Args:
            arr: 入力配列（CuPyまたはNumPy）
            
        Returns:
            NumPy配列
        """
        return arr.get() if hasattr(arr, 'get') else arr
