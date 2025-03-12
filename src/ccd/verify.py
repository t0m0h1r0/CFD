# verify.py - CCD Matrix Structure Verification Tool
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Any

# CCD related components
from grid import Grid
from equation_system import EquationSystem
from test_functions1d import TestFunctionFactory
from test_functions2d import TestFunction2DGenerator
from equation_sets import EquationSet
from solver import CCDSolver1D, CCDSolver2D  # 次元別ソルバーを直接インポート

# Output directory for results
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MatrixVisualizer:
    """Manages matrix visualization"""
    
    # ...既存のコードを維持...

class MatrixAnalyzer:
    """行列システムの解析を行うクラス"""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = output_dir
        self.visualizer = MatrixVisualizer(output_dir)
    
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
            self.visualize_matrix_structure(A, grid, title, prefix)
            
            # 特定点の視覚化
            self._visualize_sample_points(A, grid, prefix)
            
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
    
    # ...残りのメソッドは変更なし...

def verify_1d_system() -> None:
    """1次元方程式システムの検証"""
    print("\n--- 1次元方程式システムの検証 ---")
    
    try:
        # グリッドの作成
        n_points = 11
        grid = Grid(n_points, x_range=(-1.0, 1.0))
        
        # テスト関数の取得
        test_funcs = TestFunctionFactory.create_standard_functions()
        test_func = test_funcs[0]  # 最初の関数を使用
        
        # 方程式システムの作成
        system = EquationSystem(grid)
        
        # 方程式セットの取得と設定
        equation_set = EquationSet.create("poisson", dimension=1)
        equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
        
        # 1D専用ソルバーを作成
        solver = CCDSolver1D(system, grid)
        
        # 行列構造の分析と可視化
        analyzer = MatrixAnalyzer()
        analyzer.analyze_system(system, "Poisson1D")
    
    except Exception as e:
        print(f"1D検証でエラーが発生しました: {e}")
        raise


def verify_2d_system() -> None:
    """2次元方程式システムの検証"""
    print("\n--- 2次元方程式システムの検証 ---")
    
    try:
        # グリッドの作成
        nx, ny = 5, 5
        grid = Grid(nx, ny, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
        
        # テスト関数の取得
        test_funcs = TestFunction2DGenerator.create_standard_functions()
        test_func = test_funcs[0]  # 最初の関数を使用
        
        # 方程式システムの作成
        system = EquationSystem(grid)
        
        # 方程式セットの取得と設定
        equation_set = EquationSet.create("poisson", dimension=2)
        equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
        
        # 2D専用ソルバーを作成
        solver = CCDSolver2D(system, grid)
        
        # 行列構造の分析と可視化
        analyzer = MatrixAnalyzer()
        analyzer.analyze_system(system, "Poisson2D")
    
    except Exception as e:
        print(f"2D検証でエラーが発生しました: {e}")
        raise


def main() -> None:
    """メイン関数"""
    print("==== CCD行列構造検証ツール ====")
    
    try:
        # 1次元方程式システムの検証
        verify_1d_system()
        
        # 2次元方程式システムの検証
        verify_2d_system()
        
        print(f"\n検証が完了しました。結果は {OUTPUT_DIR} ディレクトリに保存されています。")
    
    except Exception as e:
        print(f"検証中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()