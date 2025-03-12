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
        行列をNumPy密行列に変換
        
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
            
    def visualize_vector_values(self, vector, grid, title: str, prefix: str = ""):
        """
        ベクトル値を可視化
        
        Args:
            vector: 右辺ベクトル
            grid: グリッドオブジェクト
            title: グラフタイトル
            prefix: 出力ファイル名の接頭辞
        
        Returns:
            str: 生成された画像のファイルパス、または警告メッセージ
        """
        try:
            plt.figure(figsize=(3, 8))
            
            # ベクトルをNumPy配列に変換
            vector_dense = vector.get() if hasattr(vector, 'get') else vector
            vector_dense = vector_dense.reshape(-1, 1)
            
            # 値の可視化
            abs_values = abs(vector_dense)
            
            # ゼロ以外の値のみを考慮
            non_zero_abs = abs_values[abs_values > 0]
            
            if len(non_zero_abs) == 0:
                print(f"警告: {title} のベクトルにゼロ以外の要素がありません。")
                plt.close()
                return ""
            
            # 最小値と最大値を計算（線形スケールとlog10スケールの両方で表示）
            vmin = non_zero_abs.min()
            vmax = abs_values.max()
            
            # 線形スケールで表示
            plt.imshow(abs_values, cmap='plasma', aspect='auto', interpolation='nearest')
            plt.title(f"RHS Vector: {title}")
            plt.xlabel("Value")
            plt.ylabel("Index")
            plt.colorbar(label='Absolute Value')
            
            # 保存とクリーンアップ
            filename = os.path.join(self.output_dir, f"{prefix}_vector_values.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        
        except Exception as e:
            print(f"ベクトル値の可視化中にエラーが発生しました: {e}")
            plt.close()
            return ""

    def visualize_matrix_and_vector(self, matrix, vector, grid, title: str, prefix: str = ""):
        """
        行列とベクトルを並べて可視化
        
        Args:
            matrix: システム行列
            vector: 右辺ベクトル
            grid: グリッドオブジェクト
            title: グラフタイトル
            prefix: 出力ファイル名の接頭辞
        
        Returns:
            str: 生成された画像のファイルパス、または警告メッセージ
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), 
                                          gridspec_kw={'width_ratios': [4, 1]})
            
            # 行列を密行列に変換
            matrix_dense = self._to_numpy_dense(matrix)
            
            # ベクトルをNumPy配列に変換
            vector_dense = vector.get() if hasattr(vector, 'get') else vector
            vector_dense = vector_dense.reshape(-1, 1)
            
            # 行列の値の可視化（対数スケール）
            abs_matrix = abs(matrix_dense)
            non_zero_abs_matrix = abs_matrix[abs_matrix > 0]
            
            if len(non_zero_abs_matrix) > 0:
                vmin_matrix = non_zero_abs_matrix.min()
                vmax_matrix = abs_matrix.max()
                
                norm_matrix = LogNorm(vmin=vmin_matrix, vmax=vmax_matrix)
                im1 = ax1.imshow(abs_matrix, cmap='viridis', norm=norm_matrix, 
                           aspect='auto', interpolation='nearest')
                ax1.set_title(f"Matrix A (Log Scale)")
                ax1.set_xlabel("Column Index")
                ax1.set_ylabel("Row Index")
                fig.colorbar(im1, ax=ax1, label='Absolute Value (Log Scale)')
            else:
                ax1.text(0.5, 0.5, "Matrix has no non-zero elements", 
                        ha='center', va='center', transform=ax1.transAxes)
            
            # ベクトルの値の可視化
            abs_vector = abs(vector_dense)
            non_zero_abs_vector = abs_vector[abs_vector > 0]
            
            if len(non_zero_abs_vector) > 0:
                vmin_vector = non_zero_abs_vector.min()
                vmax_vector = abs_vector.max()
                
                # 線形スケールで表示
                im2 = ax2.imshow(abs_vector, cmap='plasma', aspect='auto', 
                           interpolation='nearest')
                ax2.set_title(f"RHS Vector b")
                ax2.set_xlabel("Value")
                ax2.set_yticks([])  # y軸の目盛りを非表示（左の行列と共有）
                fig.colorbar(im2, ax=ax2, label='Absolute Value')
            else:
                ax2.text(0.5, 0.5, "Vector has no non-zero elements", 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.suptitle(f"System Ax = b: {title}")
            plt.tight_layout()
            
            # 保存とクリーンアップ
            filename = os.path.join(self.output_dir, f"{prefix}_matrix_and_vector.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        
        except Exception as e:
            print(f"行列とベクトルの可視化中にエラーが発生しました: {e}")
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
    
    def _print_analysis_results(self, name: str, grid, total_size: int, nnz: int, sparsity: float, 
                              scaling_info: Optional[str] = None, rhs_stats: Optional[Dict] = None):
        """
        分析結果を出力
        
        Args:
            name: 分析対象の名前
            grid: グリッドオブジェクト
            total_size: 行列サイズ
            nnz: 非ゼロ要素数
            sparsity: 疎性率
            scaling_info: スケーリング情報
            rhs_stats: 右辺ベクトルの統計情報
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
        
        if rhs_stats:
            print("\n右辺ベクトル分析:")
            print(f"  ベクトルサイズ: {rhs_stats['size']}")
            print(f"  非ゼロ要素数: {rhs_stats['nnz']}")
            print(f"  ゼロ率: {rhs_stats['zero_ratio']:.6f}")
            print(f"  最小値: {rhs_stats['min']:.6e}")
            print(f"  最大値: {rhs_stats['max']:.6e}")
            print(f"  平均値: {rhs_stats['mean']:.6e}")
    
    def analyze_system_solver(self, solver, test_func, name: str = "", 
                          scaling_method: Optional[str] = None) -> Dict:
        """
        実際のソルバーを使用して方程式システムを分析
        
        Args:
            solver: CCDSolver1D または CCDSolver2D のインスタンス
            test_func: テスト関数
            name: 識別用の名前
            scaling_method: スケーリング手法の名前
            
        Returns:
            分析結果の辞書
        """
        try:
            # グリッド情報を取得
            grid = solver.grid
            is_2d = grid.is_2d
            
            if is_2d:
                nx, ny = grid.nx_points, grid.ny_points
                grid_info = f"{nx}x{ny}"
                
                # 右辺の値と境界条件を準備
                X, Y = grid.get_points()
                x_min, x_max = grid.x_min, grid.x_max
                y_min, y_max = grid.y_min, grid.y_max
                
                # 支配方程式の右辺
                f_values = cp.zeros((nx, ny))
                for i in range(nx):
                    for j in range(ny):
                        x, y = grid.get_point(i, j)
                        f_values[i, j] = test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)
                
                # 境界条件
                left_dirichlet = cp.array([test_func.f(x_min, y) for y in grid.y])
                right_dirichlet = cp.array([test_func.f(x_max, y) for y in grid.y])
                bottom_dirichlet = cp.array([test_func.f(x, y_min) for x in grid.x])
                top_dirichlet = cp.array([test_func.f(x, y_max) for x in grid.x])
                
                left_neumann = cp.array([test_func.df_dx(x_min, y) for y in grid.y])
                right_neumann = cp.array([test_func.df_dx(x_max, y) for y in grid.y])
                bottom_neumann = cp.array([test_func.df_dy(x, y_min) for x in grid.x])
                top_neumann = cp.array([test_func.df_dy(x, y_max) for x in grid.x])
                
                # ソルバーの設定（スケーリング方法の設定も含む）
                solver.scaling_method = scaling_method
                
                # 行列と右辺を準備するための呼び出し（解を計算せず行列とRHSだけ取得）
                A, b = solver.system.build_matrix_system()
                
                # 右辺の値を設定（実際の値に更新）
                solver._prepare_rhs(b, f_values=f_values,
                                  left_dirichlet=left_dirichlet, right_dirichlet=right_dirichlet,
                                  bottom_dirichlet=bottom_dirichlet, top_dirichlet=top_dirichlet,
                                  left_neumann=left_neumann, right_neumann=right_neumann,
                                  bottom_neumann=bottom_neumann, top_neumann=top_neumann)
                
            else:
                n = grid.n_points
                grid_info = f"{n}"
                
                # 右辺の値と境界条件を準備
                x = grid.get_points()
                x_min, x_max = grid.x_min, grid.x_max
                
                # 支配方程式（ポアソン方程式）の右辺
                f_values = cp.array([test_func.d2f(xi) for xi in x])
                
                # 境界条件
                left_dirichlet = test_func.f(x_min)
                right_dirichlet = test_func.f(x_max)
                left_neumann = test_func.df(x_min)
                right_neumann = test_func.df(x_max)
                
                # ソルバーの設定（スケーリング方法の設定も含む）
                solver.scaling_method = scaling_method
                
                # 行列と右辺を準備するための呼び出し（解を計算せず行列とRHSだけ取得）
                A, b = solver.system.build_matrix_system()
                
                # 右辺の値を設定（実際の値に更新）
                solver._prepare_rhs(b, f_values=f_values,
                                  left_dirichlet=left_dirichlet, right_dirichlet=right_dirichlet,
                                  left_neumann=left_neumann, right_neumann=right_neumann)
            
            # スケーリングを適用（必要な場合）
            scaling_info = None
            if scaling_method is not None:
                from scaling import plugin_manager
                scaler = plugin_manager.get_plugin(scaling_method)
                if scaler:
                    print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                    A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                else:
                    A_scaled, b_scaled = A, b
            else:
                A_scaled, b_scaled = A, b
            
            # 基本統計情報 - 行列
            total_size = A_scaled.shape[0]
            nnz = A_scaled.nnz if hasattr(A_scaled, 'nnz') else np.count_nonzero(self.visualizer._to_numpy_dense(A_scaled))
            sparsity = 1.0 - (nnz / (total_size * total_size))
            
            # 基本統計情報 - 右辺ベクトル
            b_np = b_scaled.get() if hasattr(b_scaled, 'get') else b_scaled
            rhs_nnz = np.count_nonzero(b_np)
            rhs_zero_ratio = 1.0 - (rhs_nnz / len(b_np))
            rhs_min = float(np.min(b_np))
            rhs_max = float(np.max(b_np))
            rhs_mean = float(np.mean(b_np))
            
            rhs_stats = {
                "size": len(b_np),
                "nnz": rhs_nnz,
                "zero_ratio": rhs_zero_ratio,
                "min": rhs_min,
                "max": rhs_max,
                "mean": rhs_mean
            }
            
            # 分析結果の表示
            self._print_analysis_results(
                name, 
                grid, 
                total_size, 
                nnz, 
                sparsity, 
                scaling_method,
                rhs_stats
            )
            
            # 視覚化用のファイル名ベース
            prefix = f"{name.lower()}_{grid_info}"
            if scaling_method:
                prefix += f"_{scaling_method.lower()}"
            
            # 全体構造の可視化
            title = f"{name} {'2D' if is_2d else '1D'} Matrix"
            self.visualizer.visualize_matrix_structure(A_scaled, grid, title, prefix)
            self.visualizer.visualize_matrix_values(A_scaled, grid, title, prefix)
            
            # 右辺ベクトルの可視化
            self.visualizer.visualize_vector_values(b_scaled, grid, title, prefix)
            
            # 行列と右辺ベクトルを並べて表示
            self.visualizer.visualize_matrix_and_vector(A_scaled, b_scaled, grid, title, prefix)
            
            return {
                "size": total_size,
                "nnz": nnz,
                "sparsity": sparsity,
                "memory_dense_MB": (total_size * total_size * 8) / (1024 * 1024),
                "memory_sparse_MB": (nnz * 12) / (1024 * 1024),
                "rhs_stats": rhs_stats
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
    
    # CCDSolver1D/CCDSolver2Dクラスを拡張して、右辺ベクトルを設定するメソッドを追加
    if dimension == 1:
        # 1Dソルバーに_prepare_rhsメソッドを追加
        def _prepare_rhs_1d(self, b, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        left_neumann=None, right_neumann=None):
            n = self.grid.n_points
            
            # Set governing equation values (f values)
            if f_values is not None:
                for i in range(n):
                    b[i * 4] = f_values[i] if i < len(f_values) else 0.0
            
            # Set boundary condition values
            if left_dirichlet is not None:
                b[1] = left_dirichlet  # Left boundary Dirichlet
            if right_dirichlet is not None:
                b[(n-1) * 4 + 1] = right_dirichlet  # Right boundary Dirichlet
            if left_neumann is not None:
                b[2] = left_neumann  # Left boundary Neumann
            if right_neumann is not None:
                b[(n-1) * 4 + 2] = right_neumann  # Right boundary Neumann
            
            return b
        
        # メソッドを動的に追加
        CCDSolver1D._prepare_rhs = _prepare_rhs_1d
    
    else:
        # 2Dソルバーに_prepare_rhsメソッドを追加
        def _prepare_rhs_2d(self, b, f_values=None, left_dirichlet=None, right_dirichlet=None, 
                         bottom_dirichlet=None, top_dirichlet=None, left_neumann=None, 
                         right_neumann=None, bottom_neumann=None, top_neumann=None):
            nx, ny = self.grid.nx_points, self.grid.ny_points
            n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
            
            # Set governing equation values (f values)
            if f_values is not None:
                for j in range(ny):
                    for i in range(nx):
                        idx = (j * nx + i) * n_unknowns
                        if isinstance(f_values, (list, cp.ndarray)) and len(f_values) > i and len(f_values[i]) > j:
                            b[idx] = f_values[i, j]
                        elif isinstance(f_values, (int, float)):
                            b[idx] = f_values
            
            # Set boundary condition values - Dirichlet
            # Left boundary (i=0)
            if left_dirichlet is not None:
                for j in range(ny):
                    idx = (j * nx + 0) * n_unknowns + 1  # ψ_x
                    if isinstance(left_dirichlet, (list, cp.ndarray)) and len(left_dirichlet) > j:
                        b[idx] = left_dirichlet[j]
                    elif isinstance(left_dirichlet, (int, float)):
                        b[idx] = left_dirichlet
            
            # Right boundary (i=nx-1)
            if right_dirichlet is not None:
                for j in range(ny):
                    idx = (j * nx + (nx-1)) * n_unknowns + 1  # ψ_x
                    if isinstance(right_dirichlet, (list, cp.ndarray)) and len(right_dirichlet) > j:
                        b[idx] = right_dirichlet[j]
                    elif isinstance(right_dirichlet, (int, float)):
                        b[idx] = right_dirichlet
            
            # Bottom boundary (j=0)
            if bottom_dirichlet is not None:
                for i in range(nx):
                    idx = (0 * nx + i) * n_unknowns + 4  # ψ_y
                    if isinstance(bottom_dirichlet, (list, cp.ndarray)) and len(bottom_dirichlet) > i:
                        b[idx] = bottom_dirichlet[i]
                    elif isinstance(bottom_dirichlet, (int, float)):
                        b[idx] = bottom_dirichlet
            
            # Top boundary (j=ny-1)
            if top_dirichlet is not None:
                for i in range(nx):
                    idx = ((ny-1) * nx + i) * n_unknowns + 4  # ψ_y
                    if isinstance(top_dirichlet, (list, cp.ndarray)) and len(top_dirichlet) > i:
                        b[idx] = top_dirichlet[i]
                    elif isinstance(top_dirichlet, (int, float)):
                        b[idx] = top_dirichlet
            
            # Set boundary condition values - Neumann
            # Left boundary (i=0)
            if left_neumann is not None:
                for j in range(ny):
                    idx = (j * nx + 0) * n_unknowns + 2  # ψ_xx
                    if isinstance(left_neumann, (list, cp.ndarray)) and len(left_neumann) > j:
                        b[idx] = left_neumann[j]
                    elif isinstance(left_neumann, (int, float)):
                        b[idx] = left_neumann
            
            # Right boundary (i=nx-1)
            if right_neumann is not None:
                for j in range(ny):
                    idx = (j * nx + (nx-1)) * n_unknowns + 2  # ψ_xx
                    if isinstance(right_neumann, (list, cp.ndarray)) and len(right_neumann) > j:
                        b[idx] = right_neumann[j]
                    elif isinstance(right_neumann, (int, float)):
                        b[idx] = right_neumann
            
            # Bottom boundary (j=0)
            if bottom_neumann is not None:
                for i in range(nx):
                    idx = (0 * nx + i) * n_unknowns + 5  # ψ_yy
                    if isinstance(bottom_neumann, (list, cp.ndarray)) and len(bottom_neumann) > i:
                        b[idx] = bottom_neumann[i]
                    elif isinstance(bottom_neumann, (int, float)):
                        b[idx] = bottom_neumann
            
            # Top boundary (j=ny-1)
            if top_neumann is not None:
                for i in range(nx):
                    idx = ((ny-1) * nx + i) * n_unknowns + 5  # ψ_yy
                    if isinstance(top_neumann, (list, cp.ndarray)) and len(top_neumann) > i:
                        b[idx] = top_neumann[i]
                    elif isinstance(top_neumann, (int, float)):
                        b[idx] = top_neumann
            
            return b
        
        # メソッドを動的に追加
        CCDSolver2D._prepare_rhs = _prepare_rhs_2d
    
    for eq_set_type in equation_set_types:
        print(f"\n--- {dimension}次元 {eq_set_type.capitalize()} 方程式システムの検証 ---")
        
        try:
            # グリッドの作成
            if dimension == 1:
                grid = Grid(3, x_range=(-1.0, 1.0))  # 点数を増やして見やすくする
                test_funcs = TestFunctionFactory.create_standard_functions()
                test_func = test_funcs[3]  # Sine関数
            else:
                grid = Grid(3, 3, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))  # 点数を増やして見やすくする
                test_funcs = TestFunction2DGenerator.create_standard_functions()
                test_func = test_funcs[0]  # Sine2D関数
            
            # 可視化ツールの準備
            matrix_analyzer = MatrixAnalyzer(output_dir)
            
            # 方程式システムの作成
            system = EquationSystem(grid)
            
            # 方程式セットの取得と設定
            equation_set = EquationSet.create(eq_set_type, dimension=dimension)
            equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
            
            # ソルバーを作成
            if dimension == 1:
                solver = CCDSolver1D(system, grid)
            else:
                solver = CCDSolver2D(system, grid)
            
            # 行列構造の分析（スケーリングなし）
            matrix_analyzer.analyze_system_solver(
                solver, 
                test_func,
                f"{eq_set_type.capitalize()}{dimension}D_{test_func.name}",
            )
            
            # 行列構造の分析（対称スケーリング）
            matrix_analyzer.analyze_system_solver(
                solver, 
                test_func,
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