import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class MatrixVisualizer:
    """行列システム Ax = b の可視化を担当するクラス (CPU最適化版)"""
    
    def __init__(self, output_dir="results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize(self, A, b, x, exact_x, title, dimension, scaling=None):
        """
        行列システムを可視化して結果のファイルパスを返す
        
        Args:
            A: システム行列 (CPU/SciPy形式)
            b: 右辺ベクトル (CPU/NumPy形式)
            x: 解ベクトル (CPU/NumPy形式)
            exact_x: 厳密解ベクトル (CPU/NumPy形式)
            title: タイトル
            dimension: 次元 (1または2)
            scaling: スケーリング手法名 (optional)
            
        Returns:
            出力ファイルパス
        """
        # 出力パス生成
        scale_suffix = f"_{scaling}" if scaling else ""
        output_path = f"{self.output_dir}/{title}{scale_suffix}_matrix.png"
        
        # データ変換
        def to_numpy(arr):
            if arr is None:
                return None
            if hasattr(arr, 'toarray'):
                return arr.toarray() if not hasattr(arr, 'get') else arr.toarray()
            return arr.get() if hasattr(arr, 'get') else arr
        
        # 各データをNumPy形式に確実に変換
        A_np = to_numpy(A)
        b_np = to_numpy(b).reshape(-1, 1) if b is not None else None
        x_np = to_numpy(x).reshape(-1, 1) if x is not None else None
        exact_np = to_numpy(exact_x).reshape(-1, 1) if exact_x is not None else None
        error_np = np.abs(x_np - exact_np) if x_np is not None and exact_np is not None else None
        
        # 統合ビューのレイアウト設定
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # 1. システム行列の可視化（左上）
        ax_matrix = fig.add_subplot(gs[0, 0])
        
        # 非ゼロ要素の範囲を確認
        non_zero = A_np[A_np > 0]
        if len(non_zero) > 0:
            vmin = non_zero.min()
            vmax = A_np.max()
            # 行列の可視化（対数スケール）
            im = ax_matrix.imshow(np.abs(A_np), norm=LogNorm(vmin=vmin, vmax=vmax), 
                              cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax_matrix, label='Absolute Value (Log Scale)')
        else:
            ax_matrix.imshow(np.abs(A_np), cmap='viridis', aspect='auto')
            
        ax_matrix.set_title(f"System Matrix A")
        ax_matrix.set_xlabel("Column Index")
        ax_matrix.set_ylabel("Row Index")
        
        # 行列構造の可視化向上
        rows, cols = A_np.shape
        # 大きな行列の場合は目盛りを間引く
        if rows > 100:
            row_ticks = np.linspace(0, rows-1, 10, dtype=int)
            ax_matrix.set_yticks(row_ticks)
        if cols > 100:
            col_ticks = np.linspace(0, cols-1, 10, dtype=int)
            ax_matrix.set_xticks(col_ticks)
        
        # 2. 解ベクトルと厳密解の比較（右上）
        ax_solution = fig.add_subplot(gs[0, 1])
        
        if x_np is not None:
            # 解ベクトルをプロット
            ax_solution.plot(x_np, np.arange(len(x_np)), 'r-', label='Numerical', linewidth=1)
            
            if exact_np is not None:
                # 厳密解との比較
                ax_solution.plot(exact_np, np.arange(len(exact_np)), 'b--', label='Exact', linewidth=1)
                
                # 解の範囲を調整して見やすくする
                all_vals = np.concatenate([x_np.ravel(), exact_np.ravel()])
                min_val, max_val = all_vals.min(), all_vals.max()
                buffer = 0.1 * (max_val - min_val)
                ax_solution.set_xlim([min_val - buffer, max_val + buffer])
            
            ax_solution.set_title("Solution Vectors")
            ax_solution.set_xlabel("Value")
            ax_solution.set_ylabel("Index")
            ax_solution.legend(loc='upper right')
            
            # 大きな解ベクトルの場合は目盛りを間引く
            if len(x_np) > 100:
                solution_ticks = np.linspace(0, len(x_np)-1, 10, dtype=int)
                ax_solution.set_yticks(solution_ticks)
        
        # 3. 誤差分布の可視化（左下）
        ax_error = fig.add_subplot(gs[1, 0])
        
        if error_np is not None:
            # インデックスに対する誤差
            ax_error.semilogy(np.arange(len(error_np)), error_np, 'r-', linewidth=1)
            ax_error.set_title("Error Distribution")
            ax_error.set_xlabel("Index")
            ax_error.set_ylabel("Absolute Error (Log Scale)")
            
            # グリッド線の追加
            ax_error.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # 平均誤差と最大誤差をグラフに表示
            mean_err = np.mean(error_np)
            max_err = np.max(error_np)
            ax_error.axhline(y=mean_err, color='g', linestyle='--', label=f'Mean: {mean_err:.2e}')
            ax_error.axhline(y=max_err, color='orange', linestyle='--', label=f'Max: {max_err:.2e}')
            ax_error.legend(loc='upper right')
            
            # 大きなベクトルの場合は目盛りを間引く
            if len(error_np) > 100:
                error_ticks = np.linspace(0, len(error_np)-1, 10, dtype=int)
                ax_error.set_xticks(error_ticks)
        
        # 4. 統計情報（右下）
        ax_stats = fig.add_subplot(gs[1, 1])
        ax_stats.axis('off')  # 枠を非表示
        
        # 統計情報のテキスト生成
        info_text = []
        
        # 行列情報
        if A_np is not None:
            sparsity = 1.0 - (np.count_nonzero(A_np) / A_np.size)
            info_text.append(f"Matrix Size: {A_np.shape[0]}×{A_np.shape[1]}")
            info_text.append(f"Sparsity: {sparsity:.4f}")
            info_text.append(f"Non-zeros: {np.count_nonzero(A_np)}")
            # 条件数（小さい行列の場合のみ計算）
            if min(A_np.shape) < 1000:  # 大きな行列では計算コストが高い
                try:
                    cond = np.linalg.cond(A_np)
                    info_text.append(f"Condition Number: {cond:.2e}")
                except:
                    pass
        
        # 誤差情報
        if error_np is not None:
            info_text.append("\nError Statistics:")
            info_text.append(f"Max Error: {np.max(error_np):.4e}")
            info_text.append(f"Mean Error: {np.mean(error_np):.4e}")
            info_text.append(f"Median Error: {np.median(error_np):.4e}")
        
        # スケーリング情報
        if scaling:
            info_text.append(f"\nScaling Method: {scaling}")
        
        # 成分ごとの誤差（次元に応じて異なる処理）
        if error_np is not None:
            info_text.append("\nComponent Errors:")
            
            if dimension == 1 and len(x_np) % 4 == 0 and len(x_np) > 4:  # 1D
                components = ["ψ", "ψ'", "ψ''", "ψ'''"]
                for i, name in enumerate(components):
                    indices = range(i, len(x_np), 4)
                    comp_error = np.max(np.abs(x_np[indices] - exact_np[indices]))
                    info_text.append(f"{name}: {comp_error:.4e}")
                    
            elif dimension == 2 and len(x_np) % 7 == 0 and len(x_np) > 7:  # 2D
                components = ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
                for i, name in enumerate(components):
                    indices = range(i, len(x_np), 7)
                    comp_error = np.max(np.abs(x_np[indices] - exact_np[indices]))
                    info_text.append(f"{name}: {comp_error:.4e}")
            
            elif dimension == 3 and len(x_np) % 10 == 0 and len(x_np) > 10:  # 3D
                components = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
                for i, name in enumerate(components):
                    indices = range(i, len(x_np), 10)
                    comp_error = np.max(np.abs(x_np[indices] - exact_np[indices]))
                    info_text.append(f"{name}: {comp_error:.4e}")
        
        # 統計情報の表示
        ax_stats.text(0, 1, "\n".join(info_text), ha='left', va='top', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # 全体のタイトル設定
        plt.suptitle(f"{dimension}D {title}" + (f" (Scaling: {scaling})" if scaling else ""), 
                   fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path