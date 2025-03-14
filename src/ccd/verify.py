# verify.py - CCD Matrix Structure Verification Tool
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, Optional

# CCD関連コンポーネント
from grid import Grid
from equation_system import EquationSystem
from test_functions import TestFunctionFactory
from equation_sets import EquationSet
from solver import CCDSolver1D, CCDSolver2D
from scaling import plugin_manager

class MatrixVisualizer:
    """行列構造可視化クラス"""
    
    def __init__(self, output_dir: str = "results"):
        """初期化"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _to_numpy_dense(self, matrix):
        """行列をNumPy密行列に変換"""
        if hasattr(matrix, 'toarray'):
            return matrix.toarray().get() if hasattr(matrix, 'get') else matrix.toarray()
        elif hasattr(matrix, 'get'):
            return matrix.get()
        return matrix
    
    def visualize_system_values(self, A, b, x, grid, title: str, prefix: str = ""):
        """システムAx = bの値を可視化 (解ベクトルxも含む)"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 行列とベクトルを密行列に変換
            A_dense = self._to_numpy_dense(A)
            b_dense = self._to_numpy_dense(b)
            x_dense = self._to_numpy_dense(x) if x is not None else None
            
            # bを列ベクトルに整形
            b_reshaped = b_dense.reshape(-1, 1)
            
            if x_dense is not None:
                # xを列ベクトルに整形
                x_reshaped = x_dense.reshape(-1, 1)
                # A, x, bをSide-by-sideで配置
                combined = np.hstack([A_dense, x_reshaped, b_reshaped])
            else:
                # xが提供されていない場合はA, bだけ表示
                combined = np.hstack([A_dense, b_reshaped])
            
            # 対数スケールでシステム値を可視化
            abs_combined = abs(combined)
            
            # vminを決めるために非ゼロ値を探す
            non_zero_abs = abs_combined[abs_combined > 0]
            
            if len(non_zero_abs) == 0:
                print(f"警告: {title} のシステムにゼロ以外の要素がありません。")
                plt.close()
                return ""
            
            # 最小値と最大値を計算
            vmin = non_zero_abs.min()
            vmax = abs_combined.max()
            
            # 対数スケールを使用
            norm = LogNorm(vmin=vmin, vmax=vmax)
            plt.imshow(abs_combined, cmap='viridis', norm=norm, aspect='auto', interpolation='nearest')
            
            plt.title(f"System Values (Ax = b, Log Scale): {title}")
            plt.xlabel("Column Index (with x and b vectors)")
            plt.ylabel("Row Index")
            plt.colorbar(label='Absolute Value (Log Scale)')
            
            # 保存とクリーンアップ
            filename = os.path.join(self.output_dir, f"{prefix}_system_values.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        
        except Exception as e:
            print(f"システム値の可視化中にエラーが発生しました: {e}")
            plt.close()
            return ""


class MatrixAnalyzer:
    """行列システムの解析を行うクラス"""
    
    def __init__(self, output_dir: str = "results"):
        """初期化"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.visualizer = MatrixVisualizer(output_dir)
    
    def _print_analysis_results(self, name: str, grid, total_size: int, nnz: int, sparsity: float, scaling_info: Optional[str] = None):
        """分析結果を出力"""
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
    
    def analyze_system(self, equation_set, grid, name: str = "", scaling_method: Optional[str] = None, 
                     test_func=None) -> Dict:
        """方程式システムを分析し、行列構造を検証"""
        try:
            # ソルバーを作成
            if grid.is_2d:
                solver = CCDSolver2D(equation_set, grid)
            else:
                solver = CCDSolver1D(equation_set, grid)
            
            # スケーリング手法を設定
            solver.scaling_method = scaling_method
            
            # システム行列を取得
            A = solver.matrix_A
            
            # テスト関数から境界値と右辺ベクトルを生成
            if test_func is not None:
                # 右辺ベクトルを構築（次元に応じた方法で）
                if grid.is_2d:
                    nx, ny = grid.nx_points, grid.ny_points
                    x_min, x_max = grid.x_min, grid.x_max
                    y_min, y_max = grid.y_min, grid.y_max
                    
                    # fの値を計算
                    f_values = cp.zeros((nx, ny))
                    for i in range(nx):
                        for j in range(ny):
                            x, y = grid.get_point(i, j)
                            f_values[i, j] = test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y)
                    
                    # 境界値を計算
                    left_dirichlet = cp.array([test_func.f(x_min, y) for y in grid.y])
                    right_dirichlet = cp.array([test_func.f(x_max, y) for y in grid.y])
                    bottom_dirichlet = cp.array([test_func.f(x, y_min) for x in grid.x])
                    top_dirichlet = cp.array([test_func.f(x, y_max) for x in grid.x])
                    
                    left_neumann = cp.array([test_func.df_dx(x_min, y) for y in grid.y])
                    right_neumann = cp.array([test_func.df_dx(x_max, y) for y in grid.y])
                    bottom_neumann = cp.array([test_func.df_dy(x, y_min) for x in grid.x])
                    top_neumann = cp.array([test_func.df_dy(x, y_max) for x in grid.x])
                    
                    # 右辺ベクトルを構築
                    b = solver._build_rhs_vector(
                        f_values=f_values,
                        left_dirichlet=left_dirichlet, 
                        right_dirichlet=right_dirichlet, 
                        bottom_dirichlet=bottom_dirichlet, 
                        top_dirichlet=top_dirichlet,
                        left_neumann=left_neumann, 
                        right_neumann=right_neumann, 
                        bottom_neumann=bottom_neumann, 
                        top_neumann=top_neumann
                    )
                else:
                    n = grid.n_points
                    x_min, x_max = grid.x_min, grid.x_max
                    
                    # fの値を計算
                    f_values = cp.array([test_func.d2f(x) for x in grid.x])
                    
                    # 境界値を計算
                    left_dirichlet = test_func.f(x_min)
                    right_dirichlet = test_func.f(x_max)
                    left_neumann = test_func.df(x_min)
                    right_neumann = test_func.df(x_max)
                    
                    # 右辺ベクトルを構築
                    b = solver._build_rhs_vector(
                        f_values=f_values,
                        left_dirichlet=left_dirichlet,
                        right_dirichlet=right_dirichlet,
                        left_neumann=left_neumann,
                        right_neumann=right_neumann
                    )
            else:
                # テスト関数がない場合はゼロベクトルを使用
                if grid.is_2d:
                    nx, ny = grid.nx_points, grid.ny_points
                    n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
                    b = cp.zeros(nx * ny * n_unknowns)
                else:
                    n = grid.n_points
                    b = cp.zeros(n * 4)
            
            # スケーリングを適用（必要に応じて）
            if scaling_method:
                A_scaled, b_scaled, scaling_info, scaler = solver._apply_scaling(A, b)
            else:
                A_scaled, b_scaled = A, b
                scaling_info, scaler = None, None
            
            # 解ベクトル x を計算
            try:
                x = solver._solve_linear_system(A_scaled, b_scaled)
                
                # スケーリングが適用された場合は解をアンスケール
                if scaling_info is not None and scaler is not None:
                    x = scaler.unscale(x, scaling_info)
            except Exception as e:
                print(f"線形方程式系の解法でエラーが発生しました: {e}")
                x = None
            
            # グリッド情報
            if grid.is_2d:
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
            
            # システム (Ax = b) の可視化
            system_title = f"{name} {'2D' if grid.is_2d else '1D'} System"
            
            # 値の可視化はxベクトル含めて
            if x is not None:
                self.visualizer.visualize_system_values(A_scaled, b_scaled, x, grid, system_title, prefix)
            else:
                self.visualizer.visualize_system_values(A_scaled, b_scaled, None, grid, system_title, prefix)
            
            return {
                "size": total_size,
                "nnz": nnz,
                "sparsity": sparsity
            }
            
        except Exception as e:
            print(f"分析中にエラーが発生しました: {e}")
            raise


def verify_system(dimension: int, output_dir: str = "results"):
    """方程式システムを検証"""
    # 両方の方程式セットで検証を行う
    equation_set_types = ["poisson", "derivative"]
    
    for eq_set_type in equation_set_types:
        print(f"\n--- {dimension}次元 {eq_set_type.capitalize()} 方程式システムの検証 ---")
        
        try:
            # グリッドの作成
            if dimension == 1:
                grid = Grid(16, x_range=(-1.0, 1.0))
                # テスト関数を取得
                test_funcs = TestFunctionFactory.create_standard_1d_functions()
                test_func = test_funcs[3]  # Sine関数
            else:
                grid = Grid(9, 3, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
                # テスト関数を取得
                test_funcs = TestFunctionFactory.create_standard_2d_functions()
                test_func = test_funcs[0]  # Sine2D関数
            
            # 方程式セットを作成
            equation_set = EquationSet.create(eq_set_type, dimension=dimension)
            
            # 可視化ツールの準備
            matrix_analyzer = MatrixAnalyzer(output_dir)
            
            # 行列構造の分析（スケーリングなし）
            matrix_analyzer.analyze_system(
                equation_set, 
                grid, 
                f"{eq_set_type.capitalize()}{dimension}D_{test_func.name}",
                test_func=test_func
            )
            
            # 行列構造の分析（対称スケーリング）
            matrix_analyzer.analyze_system(
                equation_set, 
                grid, 
                f"{eq_set_type.capitalize()}{dimension}D_{test_func.name}", 
                scaling_method="SymmetricScaling",
                test_func=test_func
            )
        
        except Exception as e:
            print(f"{dimension}D {eq_set_type} 検証でエラーが発生しました: {e}")
            raise


def main(output_dir: str = "results"):
    """メイン関数"""
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