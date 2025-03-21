"""
高精度コンパクト差分法 (CCD) の1次元結果可視化

このモジュールは、1次元CCDソルバーの計算結果を可視化するための
クラスと機能を提供します。
"""

import matplotlib.pyplot as plt
from typing import List

from base_visualizer import BaseVisualizer


class CCDVisualizer1D(BaseVisualizer):
    """CCDソルバーの1D結果を可視化するクラス"""

    def __init__(self, output_dir="results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス
        """
        super().__init__(output_dir)

    def visualize_derivatives(self, grid, function_name, numerical, exact, errors, prefix="", save=True, show=False, dpi=150):
        """
        導関数の結果を可視化
        
        Args:
            grid: Gridオブジェクト
            function_name: テスト関数名
            numerical: 数値解のリスト [psi, psi', psi'', psi''']
            exact: 厳密解のリスト
            errors: 誤差リスト
            prefix: 出力ファイルの接頭辞
            save: 保存するかどうか
            show: 表示するかどうか
            dpi: 画像のDPI値
            
        Returns:
            出力ファイルパス
        """
        x = grid.get_points()
        n_points = grid.n_points
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]

        # NumPy配列に変換
        x_np = self._to_numpy(x)

        # 2×2のサブプロットを作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for i, ax in enumerate(axes.flat):
            if i < len(numerical):
                # データをNumPy配列に変換
                exact_data = self._to_numpy(exact[i])
                num_data = self._to_numpy(numerical[i])

                # 各成分のプロット
                ax.plot(x_np, exact_data, "b-", label="Exact")
                ax.plot(x_np, num_data, "r--", label="Numerical")
                ax.set_title(f"{titles[i]} (error: {errors[i]:.2e})")
                ax.legend()
                ax.grid(True)

        plt.suptitle(f"Results for {function_name} function ({n_points} points)")
        plt.tight_layout()

        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, n_points, prefix)
            plt.savefig(filepath, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filepath

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
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        titles = ["$\\psi$", "$\\psi'$", "$\\psi''$", "$\\psi'''$"]

        # グリッド間隔を計算（[-1, 1]の範囲を仮定）
        grid_spacings = [2.0 / (n - 1) for n in grid_sizes]

        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            if i < len(titles):
                # グリッドサイズごとの該当成分のエラーを取得
                errors = [results[n][i] for n in grid_sizes]
                
                # エラーが全て0の場合はグラフを描画しない
                if all(err == 0 for err in errors):
                    ax.text(0.5, 0.5, f"All errors are 0 for {title}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue
                    
                # エラーが0の場合は小さな値に置き換え（対数スケール用）
                plot_errors = [max(err, self.min_log_value) for err in errors]
                
                # 対数スケールで誤差をプロット
                ax.loglog(grid_spacings, plot_errors, 'o-', label=f"Error ({title})")
                
                # 基準となる傾きを示す参照線の追加
                x_range = [min(grid_spacings), max(grid_spacings)]
                
                # 2次、4次、6次の収束線を描画
                for order, style, color in zip([2, 4, 6], ['--', '-.', ':'], ['r', 'g', 'b']):
                    # 参照線の位置を最後のポイントに合わせる
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

        filepath = ""
        if save:
            if prefix:
                filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            else:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
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
            "1D"
        """
        return "1D"
    
    def get_error_types(self) -> List[str]:
        """
        エラータイプのリストを返す
        
        Returns:
            1D用のエラータイプリスト
        """
        return ["ψ", "ψ'", "ψ''", "ψ'''"]
