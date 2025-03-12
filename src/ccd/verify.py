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
        
        Returns:
            str: 生成された画像のファイルパス
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
        
        Returns:
            str: 生成された画像のファイルパス、または警告メッセージ
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # 行列を密行列に変換
            matrix_dense = self._to_numpy_dense(matrix)
            
            # 値の可視化（対数スケール）
            abs_matrix = abs(matrix_dense)
            
            # ゼロ以外の値のみを考慮
            non_zero_abs = abs_matrix[abs_matrix > 0]
            
            if len(non_zero_abs) == 0:
                print(f"警告: {title} の行列にゼロ以外の要素がありません。")
                plt.close()
                return ""
            
            # 最小値と最大値を慎重に計算
            vmin = non_zero_abs.min()
            vmax = abs_matrix.max()
            
            norm = LogNorm(vmin=vmin, vmax=vmax)
            plt.imshow(abs_matrix, cmap='viridis', norm=norm, 
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
        
        except Exception as e:
            print(f"マトリックス値の可視化中にエラーが発生しました: {e}")
            plt.close()
            return ""


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
    
    def _print_analysis_results(self, name: str, grid, total_size: int, nnz: int, sparsity: float, scaling_info: Optional[str] = None):
        """
        分析結果を出力
        
        Args:
            name: 分析対象の名前
            grid: グリッドオブジェクト
            total_size: 行列サイズ
            nnz: 非ゼロ要素数
            sparsity: 疎性率
            scaling_info: スケーリング情報
        """
        print(f"\n{name} 行列分析:")
        if scaling_info:
            print(f"  スケーリング: {scaling_info}")
        if grid.is_2d:
            print(f"  グリッドサイズ: {grid.nx_points}x{grid.ny_points}")
        else:
            print(f"  グリッドサイズ: {grid.n_points}")
        print(f"  行列サイズ: {total_size} x {total_size}")
        print(f"  非ゼロ要素数: {nnz}")
        print(f"  疎性率: {sparsity:.6f}")
        print(f"  メモリ使用量(密行列): {(total_size * total_size * 8) / (1024 * 1024):.2f} MB")
        print(f"  メモリ使用量(疎行列): {(nnz * 12) / (1024 * 1024):.2f} MB")
    
    def analyze_system(self, system: EquationSystem, name: str = "", scaling_method: Optional[str] = None) -> Dict:
        """
        方程式システムを分析し、行列構造を検証
        
        Args:
            system: EquationSystem のインスタンス
            name: 識別用の名前
            scaling_method: スケーリング手法の名前
            
        Returns:
            分析結果の辞書
        """
        try:
            # スケール付き行列システムを構築
            if scaling_method:
                # スケーリングプラグインを取得
                scaler = plugin_manager.get_plugin(scaling_method)
                # 行列システムを構築
                A, b = system.build_matrix_system()
                
                # スケーリングを適用
                A_scaled, b_scaled, _ = scaler.scale(A, b)
            else:
                # スケーリングなしで行列システムを構築
                A_scaled, b_scaled = system.build_matrix_system()
            
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
            total_size = A_scaled.shape[0]
            nnz = A_scaled.nnz if hasattr(A_scaled, 'nnz') else np.count_nonzero(self.visualizer._to_numpy_dense(A_scaled))
            sparsity = 1.0 - (nnz / (total_size * total_size))
            
            # 分析結果の表示
            self._print_analysis_results(
                name, 
                grid, 
                total_size, 
                nnz, 
                sparsity, 
                scaling_method
            )
            
            # 視覚化用のファイル名ベース
            prefix = f"{name.lower()}_{grid_info}"
            if scaling_method:
                prefix += f"_{scaling_method.lower()}"
            
            # 全体構造の可視化
            title = f"{name} {'2D' if is_2d else '1D'} Matrix"
            self.visualizer.visualize_matrix_structure(A_scaled, grid, title, prefix)
            self.visualizer.visualize_matrix_values(A_scaled, grid, title, prefix)
            
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


def verify_system(dimension: int, output_dir: str = "results"):
    """
    方程式システムを検証
    
    Args:
        dimension: 次元 (1 or 2)
        output_dir: 出力ディレクトリのパス
    """
    # 両方の方程式セットで検証を行う
    equation_set_types = ["poisson", "derivative"]
    
    for eq_set_type in equation_set_types:
        print(f"\n--- {dimension}次元 {eq_set_type.capitalize()} 方程式システムの検証 ---")
        
        try:
            # グリッドの作成
            if dimension == 1:
                grid = Grid(3, x_range=(-1.0, 1.0))
                test_funcs = TestFunctionFactory.create_standard_functions()
                test_func = test_funcs[3]  # Sine関数
            else:
                grid = Grid(3, 3, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
                test_funcs = TestFunction2DGenerator.create_standard_functions()
                test_func = test_funcs[0]  # Sine2D関数
            
            # 可視化ツールの準備
            matrix_analyzer = MatrixAnalyzer(output_dir)
            
            # 方程式システムの作成
            system = EquationSystem(grid)
            
            # 方程式セットの取得と設定
            equation_set = EquationSet.create(eq_set_type, dimension=dimension)
            equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
            
            # 行列構造の分析（スケーリングなし）
            matrix_analyzer.analyze_system(
                system, 
                f"{eq_set_type.capitalize()}{dimension}D_{test_func.name}",
            )
            
            # 行列構造の分析（対称スケーリング）
            matrix_analyzer.analyze_system(
                system, 
                f"{eq_set_type.capitalize()}{dimension}D_{test_func.name}", 
                scaling_method="SymmetricScaling"
            )
        
        except Exception as e:
            print(f"{dimension}D {eq_set_type} 検証でエラーが発生しました: {e}")
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
        verify_system(1, output_dir)
        
        # 2次元方程式システムの検証
        verify_system(2, output_dir)
        
        print(f"\n検証が完了しました。結果は {output_dir} ディレクトリに保存されています。")
    
    except Exception as e:
        print(f"検証中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()