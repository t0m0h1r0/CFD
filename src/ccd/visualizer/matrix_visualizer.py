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
        
        # 合成データ作成
        composite = np.abs(A_np)
        offset, cols = A_np.shape[1], {"A": A_np.shape[1] // 2}
        
        # ベクトル結合
        for name, data in [("x", x_np), ("exact", exact_np), ("error", error_np), ("b", b_np)]:
            if data is not None:
                composite = np.hstack((composite, data))
                cols[name] = offset
                offset += 1
        
        # 可視化
        plt.figure(figsize=(12, 8))
        non_zero = composite[composite > 0]
        if len(non_zero) > 0:
            plt.imshow(composite, norm=LogNorm(vmin=non_zero.min(), vmax=composite.max()), 
                      cmap='viridis', aspect='auto')
            plt.colorbar(label='Absolute Value (Log Scale)')
        
        # A行列とベクトル間に区切り線
        plt.axvline(x=A_np.shape[1]-0.5, color='r', linestyle='-')
        
        # タイトル・ラベル
        plt.title(f"{dimension}D {title}" + (f" (Scaling: {scaling})" if scaling else ""))
        plt.xlabel("Component"), plt.ylabel("Row/Index")
        
        # 列ラベル追加
        for name, pos in cols.items():
            plt.text(pos, -5, name, ha='center')
        
        # 統計情報
        info = []
        
        # 行列情報
        if A_np is not None:
            sparsity = 1.0 - (np.count_nonzero(A_np) / A_np.size)
            info.append(f"Matrix: {A_np.shape[0]}×{A_np.shape[1]}, Sparsity: {sparsity:.4f}")
        
        # 誤差情報
        if error_np is not None:
            info.append(f"Error: Max={np.max(error_np):.4e}, Avg={np.mean(error_np):.4e}")
            
            # 成分ごとの誤差
            if len(x_np) % 4 == 0 and len(x_np) > 4:  # 1D
                components = ["ψ", "ψ'", "ψ''", "ψ'''"]
                errors = []
                
                for i, name in enumerate(components):
                    indices = range(i, len(x_np), 4)
                    errors.append(f"{name}: {np.max(np.abs(x_np[indices] - exact_np[indices])):.4e}")
                
                info.append("Component Errors: " + ", ".join(errors))
            elif len(x_np) % 7 == 0 and len(x_np) > 7:  # 2D
                components = ["ψ", "ψ_x", "ψ_xx", "ψ_xxx", "ψ_y", "ψ_yy", "ψ_yyy"]
                errors = []
                
                for i, name in enumerate(components):
                    indices = range(i, len(x_np), 7)
                    errors.append(f"{name}: {np.max(np.abs(x_np[indices] - exact_np[indices])):.4e}")
                
                info.append("Component Errors: " + ", ".join(errors[:3]) + "...")
        
        # 情報表示
        for i, text in enumerate(info):
            plt.figtext(0.5, 0.01 - i*0.03, text, ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1 + len(info)*0.03)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path