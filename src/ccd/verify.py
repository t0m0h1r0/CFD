# verify.py - CCD Matrix Structure Verification Tool
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, Optional, Tuple, List, Any

# CCD関連コンポーネント
from grid import Grid
from equation_system import EquationSystem
from test_functions1d import TestFunctionFactory
from test_functions2d import TestFunction2DGenerator
from equation_sets import EquationSet
from solver import CCDSolver1D, CCDSolver2D
from scaling import plugin_manager

class MatrixVisualizer:
    """行列構造可視化クラス"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _to_numpy_dense(self, matrix):
        """
        行列をNumpy密行列に変換
        
        Args:
            matrix: CuPy/SciPy疎行列
            
        Returns:
            NumPy密行列
        """
        if hasattr(matrix, 'toarray'):
            return matrix.toarray().get() if hasattr(matrix, 'get') else matrix.toarray()
        elif hasattr(matrix, 'get'):
            return matrix.get()
        return matrix
    
    def visualize_matrix_structure(self, matrix, grid, title: str, prefix: str = ""):
        """
        行列構造を可視化
        
        Args:
            matrix: システム行列
            grid: グリッドオブジェクト
            title: グラフタイトル
            prefix: 出力ファイル名の接頭辞
        """
        plt.figure(figsize=(10, 8))
        
        # 行列を密行列に変換
        matrix_dense = self._to_numpy_dense(matrix)
        
        # 構造の可視化
        plt.imshow(matrix_dense != 0, cmap='binary', aspect='auto', interpolation='nearest')
        plt.title(f"Matrix Structure: {title}")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.colorbar(label='Non-zero Elements')
        
        # 保存とクリーンアップ
        filename = os.path.join(self.output_dir, f"{prefix}_matrix_structure.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def visualize_matrix_values(self, matrix, grid, title: str, prefix: str = ""):
        """
        行列値を可視化
        
        Args:
            matrix: システム行列
            grid: グリッドオブジェクト
            title: グラフタイトル
            prefix: 出力ファイル名の接頭辞
        """
        plt.figure(figsize=(10, 8))
        
        # 行列を密行列に変換
        matrix_dense = self._to_numpy_dense(matrix)
        
        # 値の可視化（対数スケール）
        # Compute vmin carefully to avoid potential warnings
        non_zero_abs = abs(matrix_dense[matrix_dense != 0])
        vmin = non_zero_abs.min() if len(non_zero_abs) > 0 else 1e-10
        vmax = abs(matrix_dense).max()
        
        norm = LogNorm(vmin=vmin, vmax=vmax)
        plt.imshow(abs(matrix_dense), cmap='viridis', norm=norm, 
                   aspect='auto', interpolation='nearest')
        plt.title(f"Matrix Values (Log Scale): {title}")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.colorbar(label='Absolute Value (Log Scale)')
        
        # 保存とクリーンアップ
        filename = os.path.join(self.output_dir, f"{prefix}_matrix_values.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename


class MatrixAnalyzer:
    """行列システムの解析を行うクラス"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.visualizer = MatrixVisualizer(output_dir)
    
    def _print_analysis_results(self, name: str, grid, total_size: int, nnz: int, sparsity: float):
        """
        分析結果を出力
        
        Args:
            name: 分析対象の名前
            grid: グリッドオブジェクト
            total_size: 行列サイズ
            nnz: 非ゼロ要素数
            sparsity: 疎性率
        """
        print(f"\n{name} 行列分析:")
        if grid.is_2d:
            print(f"  グリッドサイズ: {grid.nx_points}x{grid.ny_points}")
        else:
            print(f"  グリッドサイズ: {grid.n_points}")
        print(f"  行列サイズ: {total_size} x {total_size}")
        print(f"  非ゼロ要素数: {nnz}")
        print(f"  疎性率: {sparsity:.6f}")
        print(f"  メモリ使用量(密行列): {(total_size * total_size * 8) / (1024 * 1024):.2f} MB")
        print(f"  メモリ使用量(疎行列): {(nnz * 12) / (1024 * 1024):.2f} MB")
    
    def _get_prefix(self, name: str, grid_info: str) -> str:
        """
        出力ファイル名の接頭辞を生成
        
        Args:
            name: 分析対象の名前
            grid_info: グリッド情報
            
        Returns:
            接頭辞文字列
        """
        return f"{name.lower()}_{grid_info}"
    
    def _get_title(self, name: str, is_2d: bool) -> str:
        """
        タイトルを生成
        
        Args:
            name: 分析対象の名前
            is_2d: 2次元フラグ
            
        Returns:
            タイトル文字列
        """
        dim = "2D" if is_2d else "1D"
        return f"{name} {dim} Matrix"
    
    def analyze_system(self, system: EquationSystem, name: str = "") -> Dict:
        """
        方程式システムを分析し、行列構造を検証
        
        Args:
            system: EquationSystem のインスタンス
            name: 識別用の名前
            
        Returns:
            分析結果の辞書
        """
        try:
            # 行列システムを構築
            A, b = system.build_matrix_system()
            
            # グリッド情報の取得
            grid = system.grid
            is_2d = grid.is_2d
            
            if is_2d:
                nx, ny = grid.nx_points, grid.ny_points
                grid_info = f"{nx}x{ny}"
            else:
                n = grid.n_points
                grid_info = f"{n}"
            
            # 基本統計情報
            total_size = A.shape[0]
            nnz = A.nnz if hasattr(A, 'nnz') else np.count_nonzero(self.visualizer._to_numpy_dense(A))
            sparsity = 1.0 - (nnz / (total_size * total_size))
            
            # 分析結果の表示
            self._print_analysis_results(name, grid, total_size, nnz, sparsity)
            
            # 視覚化用のファイル名ベース
            prefix = self._get_prefix(name, grid_info)
            
            # 全体構造の可視化
            title = self._get_title(name, is_2d)
            self.visualizer.visualize_matrix_structure(A, grid, title, prefix)
            self.visualizer.visualize_matrix_values(A, grid, title, prefix)
            
            return {
                "size": total_size,
                "nnz": nnz,
                "sparsity": sparsity,
                "memory_dense_MB": (total_size * total_size * 8) / (1024 * 1024),
                "memory_sparse_MB": (nnz * 12) / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"分析中にエラーが発生しました: {e}")
            raise


def verify_1d_system(output_dir: str = "results"):
    """
    1次元方程式システムの検証
    
    Args:
        output_dir: 出力ディレクトリのパス
    """
    print("\n--- 1次元方程式システムの検証 ---")
    
    try:
        # グリッドの作成
        n_points = 3  # より多くの点で検証
        grid = Grid(n_points, x_range=(-1.0, 1.0))
        
        # テスト関数の取得
        test_funcs = TestFunctionFactory.create_standard_functions()
        
        # 可視化ツールの準備
        matrix_analyzer = MatrixAnalyzer(output_dir)
        
        # 各テスト関数で検証
        for test_func in test_funcs:
            # 方程式システムの作成
            system = EquationSystem(grid)
            
            # 方程式セットの取得と設定
            equation_set = EquationSet.create("poisson", dimension=1)
            equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
            
            # 1D専用ソルバーを作成
            solver = CCDSolver1D(system, grid)
            
            # 行列構造の分析と可視化
            matrix_analyzer.analyze_system(system, f"Poisson1D_{test_func.name}")
    
    except Exception as e:
        print(f"1D検証でエラーが発生しました: {e}")
        raise


def verify_2d_system(output_dir: str = "results"):
    """
    2次元方程式システムの検証
    
    Args:
        output_dir: 出力ディレクトリのパス
    """
    print("\n--- 2次元方程式システムの検証 ---")
    
    try:
        # グリッドの作成
        nx, ny = 3, 3  # より適切なサイズに
        grid = Grid(nx, ny, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
        
        # テスト関数の取得
        test_funcs = TestFunction2DGenerator.create_standard_functions()
        
        # 可視化ツールの準備
        matrix_analyzer = MatrixAnalyzer(output_dir)
        
        # 各テスト関数で検証
        for test_func in test_funcs:
            # 方程式システムの作成
            system = EquationSystem(grid)
            
            # 方程式セットの取得と設定
            equation_set = EquationSet.create("poisson", dimension=2)
            equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
            
            # 2D専用ソルバーを作成
            solver = CCDSolver2D(system, grid)
            
            # 行列構造の分析と可視化
            matrix_analyzer.analyze_system(system, f"Poisson2D_{test_func.name}")
    
    except Exception as e:
        print(f"2D検証でエラーが発生しました: {e}")
        raise


def main(output_dir: str = "results"):
    """
    メイン関数
    
    Args:
        output_dir: 出力ディレクトリのパス
    """
    print("==== CCD行列構造検証ツール ====")
    
    try:
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 1次元方程式システムの検証
        verify_1d_system(output_dir)
        
        # 2次元方程式システムの検証
        verify_2d_system(output_dir)
        
        print(f"\n検証が完了しました。結果は {output_dir} ディレクトリに保存されています。")
    
    except Exception as e:
        print(f"検証中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()