# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from typing import List, Dict, Tuple, Optional
from grid import Grid

class CCDVisualizer:
    """CCDソルバーの結果を可視化するクラス"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = output_dir
        
        # 出力ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 対数スケール用の最小値
        self.min_log_value = 1e-16
    
    def generate_filename(self, func_name: str, n_points: int, prefix: str = "") -> str:
        """
        出力ファイル名を生成
        
        Args:
            func_name: テスト関数名
            n_points: グリッド点数（解像度）
            prefix: ファイル名の接頭辞
            
        Returns:
            生成されたファイル名
        """
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{n_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{n_points}_points.png"
    
    def visualize_derivatives(
        self,
        grid: Grid,
        function_name: str,
        numerical: List[cp.ndarray],
        exact: List[cp.ndarray],
        errors: List[float],
        prefix: str = "",
        save: bool = True,
        show: bool = False,
        dpi: int = 150
    ) -> str:
        """
        導関数の結果を可視化
        
        Args:
            grid: 計算格子
            function_name: テスト関数名
            numerical: 数値解 [psi, psi', psi'', psi''']
            exact: 解析解 [psi, psi', psi'', psi''']
            errors: 誤差 [err_psi, err_psi', err_psi'', err_psi''']
            prefix: ファイル名の接頭辞
            save: 結果を保存するかどうか
            show: 結果を画面に表示するかどうか
            dpi: 画像解像度
            
        Returns:
            保存したファイルのパス（saveがFalseの場合は空文字）
        """
        # グリッド点座標
        x = grid.get_points()
        n_points = grid.n_points
        
        # プロット作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]
        
        # CuPy配列をNumPy配列に変換
        x_np = x.get() if hasattr(x, 'get') else x
        
        for i, ax in enumerate(axes.flat):
            # CuPy配列をNumPy配列に変換
            exact_data = exact[i].get() if hasattr(exact[i], 'get') else exact[i]
            num_data = numerical[i].get() if hasattr(numerical[i], 'get') else numerical[i]
            
            ax.plot(x_np, exact_data, 'b-', label='Exact')
            ax.plot(x_np, num_data, 'r--', label='Numerical')
            ax.set_title(f"{titles[i]} (error: {errors[i]:.2e})")
            ax.legend()
            ax.grid(True)
        
        # 全体のタイトル
        plt.suptitle(f"Results for {function_name} function ({n_points} points)")
        plt.tight_layout()
        
        # 保存
        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, n_points, prefix)
            plt.savefig(filepath, dpi=dpi)
            print(f"Plot saved to {filepath}")
        
        # 表示
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filepath
    
    def visualize_error_comparison(
        self,
        function_name: str,
        methods: List[str],
        error_data: Dict[str, List[float]],
        prefix: str = "",
        save: bool = True,
        show: bool = False,
        dpi: int = 150
    ) -> str:
        """
        異なる手法間の誤差比較を可視化
        
        Args:
            function_name: テスト関数名
            methods: 比較する手法のリスト
            error_data: 誤差データ {method_name: [err_psi, err_psi', err_psi'', err_psi''']}
            prefix: ファイル名の接頭辞
            save: 結果を保存するかどうか
            show: 結果を画面に表示するかどうか
            dpi: 画像解像度
            
        Returns:
            保存したファイルのパス（saveがFalseの場合は空文字）
        """
        # プロット作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # データ準備
        x = range(len(methods))
        bar_width = 0.2
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]
        colors = ['C0', 'C1', 'C2', 'C3']
        
        # 各種誤差データをプロット
        for i in range(4):  # psi, psi', psi'', psi'''
            errors = [error_data[method][i] for method in methods]
            ax.bar([pos + i * bar_width for pos in x], errors, 
                  bar_width, label=titles[i], color=colors[i])
        
        # 軸の設定
        ax.set_xlabel('Method')
        ax.set_ylabel('Error (log scale)')
        ax.set_title(f'Error Comparison for {function_name}')
        ax.set_xticks([pos + 1.5 * bar_width for pos in x])
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_yscale('log')  # 対数スケール
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # 保存
        filepath = ""
        if save:
            filepath = f"{self.output_dir}/{prefix}error_comparison_{function_name.lower()}.png"
            plt.savefig(filepath, dpi=dpi)
            print(f"Error comparison plot saved to {filepath}")
        
        # 表示
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filepath
    
    def visualize_grid_convergence(
        self,
        function_name: str,
        grid_sizes: List[int],
        errors: Dict[int, List[float]],
        prefix: str = "",
        save: bool = True,
        show: bool = False,
        dpi: int = 150
    ) -> str:
        """
        グリッドサイズによる収束性を可視化
        
        Args:
            function_name: テスト関数名
            grid_sizes: グリッドサイズのリスト
            errors: グリッドサイズごとの誤差 {grid_size: [err_psi, err_psi', err_psi'', err_psi''']}
            prefix: ファイル名の接頭辞
            save: 結果を保存するかどうか
            show: 結果を画面に表示するかどうか
            dpi: 画像解像度
            
        Returns:
            保存したファイルのパス（saveがFalseの場合は空文字）
        """
        # プロット作成
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # データ準備
        h_values = [1.0 / (n - 1) for n in grid_sizes]  # グリッド幅
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]
        markers = ['o', 's', '^', 'd']
        
        # 各導関数の誤差を別々の線でプロット
        for i in range(4):  # psi, psi', psi'', psi'''
            error_values = [errors[size][i] for size in grid_sizes]
            ax.loglog(h_values, error_values, marker=markers[i], label=titles[i])
        
        # 傾きの参照線（6次精度を想定）
        x_ref = [min(h_values), max(h_values)]
        y_ref_6th = [min(h_values)**6, max(h_values)**6]
        ax.loglog(x_ref, y_ref_6th, 'k--', alpha=0.5, label='6th Order (h⁶)')
        
        # 軸の設定
        ax.set_xlabel('Grid Size (h)')
        ax.set_ylabel('Error')
        ax.set_title(f'Grid Convergence for {function_name}')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # 保存
        filepath = ""
        if save:
            filepath = f"{self.output_dir}/{prefix}grid_convergence_{function_name.lower()}.png"
            plt.savefig(filepath, dpi=dpi)
            print(f"Grid convergence plot saved to {filepath}")
        
        # 表示
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filepath
    
    def compare_all_functions_errors(
        self,
        results_summary: Dict[str, List[float]],
        prefix: str = "",
        dpi: int = 150,
        show: bool = False
    ) -> str:
        """
        全テスト関数の誤差比較グラフを生成（対数スケール強制版）
        
        Args:
            results_summary: 関数名ごとの誤差リスト {function_name: [err_psi, err_psi', err_psi'', err_psi''']}
            prefix: ファイル名の接頭辞
            dpi: 画像解像度
            show: 結果を画面に表示するかどうか
            
        Returns:
            保存したファイルのパス
        """
        # 関数名とエラーデータを抽出
        func_names = list(results_summary.keys())
        
        # 比較グラフの作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        error_types = ["ψ", "ψ'", "ψ''", "ψ'''"]
        
        for i, (ax, error_type) in enumerate(zip(axes.flat, error_types)):
            # 元の誤差値を取得
            original_errors = [results_summary[name][i] for name in func_names]
            
            # 対数スケール用に0を小さな値に置き換え
            errors = []
            for err in original_errors:
                if err == 0.0:
                    errors.append(self.min_log_value)  # 0の代わりに非常に小さな値を使用
                else:
                    errors.append(err)
            
            # x軸の位置を設定
            x_positions = np.arange(len(func_names))
            
            # 誤差の対数棒グラフ
            bars = ax.bar(x_positions, errors)
            
            # 対数スケールを設定
            ax.set_yscale('log')
            
            # グラフの装飾
            ax.set_title(f"{error_type} Error Comparison")
            ax.set_xlabel("Test Function")
            ax.set_ylabel("Error (log scale)")
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # 明示的に目盛りを設定してから、ラベルを設定
            ax.set_xticks(x_positions)
            ax.set_xticklabels(func_names, rotation=45, ha='right')
            
            # 値をバーの上に表示
            for j, (bar, orig_err) in enumerate(zip(bars, original_errors)):
                height = bar.get_height()
                # 高さが0だった場合は特別な表示
                if orig_err == 0.0:
                    label_text = "0.0"
                else:
                    label_text = f'{orig_err:.2e}'
                
                # y位置は常にバーの上に
                y_pos = height * 1.1
                
                ax.annotate(label_text,
                           xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                           xytext=(0, 3),  # オフセット
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        plt.tight_layout()
        
        # 保存
        filename = f"{self.output_dir}/{prefix}_all_functions_comparison.png" if prefix else f"{self.output_dir}/all_functions_comparison.png"
        plt.savefig(filename, dpi=dpi)
        print(f"All functions comparison plot saved to {filename}")
        
        # 表示
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filename