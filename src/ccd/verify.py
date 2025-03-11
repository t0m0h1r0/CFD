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

# Output directory for results
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MatrixVisualizer:
    """Manages matrix visualization"""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        """
        Initialize
        
        Args:
            output_dir: Path to output directory
        """
        self.output_dir = output_dir
        
    def visualize_structure(self, 
                           A: Any, 
                           grid: Optional[Grid] = None, 
                           title: str = "Matrix Structure Visualization", 
                           save_path: Optional[str] = None) -> None:
        """
        Visualize the overall matrix structure
        
        Args:
            A: Matrix (sparse or dense)
            grid: Grid object
            title: Figure title
            save_path: Save path (display only if not specified)
        """
        # Convert CuPy array to NumPy array
        A_dense = self._to_numpy_dense(A)
        
        # Matrix statistics
        total_size = A_dense.shape[0]
        nnz = self._count_nonzeros(A, A_dense)
        sparsity = 1.0 - (float(nnz) / (total_size * total_size))
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.spy(A_dense, markersize=0.5, color='blue')
        plt.title(f"{title}\n(Size: {total_size}×{total_size}, Non-zero elements: {nnz}, Sparsity: {sparsity:.4f})")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        
        # Grid lines based on grid information
        if grid is not None:
            self._add_grid_lines(grid, A_dense.shape[0])
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Matrix structure saved: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def visualize_block(self, 
                       A: Any, 
                       i: int, 
                       j: Optional[int] = None, 
                       grid: Optional[Grid] = None, 
                       save_path: Optional[str] = None) -> None:
        """
        Visualize a specific block of the matrix in detail
        
        Args:
            A: Matrix (sparse or dense)
            i: Row index
            j: Column index (for 2D)
            grid: Grid object
            save_path: Save path
        """
        # Convert to dense matrix
        A_dense = self._to_numpy_dense(A)
        
        # Determine dimension and block size
        is_2d, n_unknowns, labels, idx = self._determine_block_params(i, j, grid)
        
        # Extract block
        if idx + n_unknowns <= A_dense.shape[0]:
            block = A_dense[idx:idx+n_unknowns, idx:idx+n_unknowns]
        else:
            raise ValueError(f"Index {idx} exceeds matrix size {A_dense.shape[0]}")
        
        # Create figure
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(block) > 1e-10, cmap='Blues', interpolation='none')
        
        # Set title
        if is_2d:
            plt.title(f"Matrix Block for Grid Point ({i},{j})")
        else:
            plt.title(f"Matrix Block for Grid Point {i}")
        
        # Axis labels
        plt.xticks(range(n_unknowns), labels, rotation=45)
        plt.yticks(range(n_unknowns), labels)
        
        # Display values
        self._annotate_block_values(block)
        
        plt.colorbar(label="Value Presence")
        plt.grid(True, color='gray', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Block matrix saved: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def visualize_neighborhood(self, 
                              A: Any, 
                              i: int, 
                              j: Optional[int] = None, 
                              grid: Optional[Grid] = None, 
                              neighborhood_size: int = 1, 
                              save_path: Optional[str] = None) -> None:
        """
        Visualize the matrix around a specific point
        
        Args:
            A: Matrix (sparse or dense)
            i: Row index
            j: Column index (for 2D)
            grid: Grid object
            neighborhood_size: Number of adjacent grid points
            save_path: Save path
        """
        # Convert to dense matrix
        A_dense = self._to_numpy_dense(A)
        
        # Determine dimension and block size
        is_2d, n_unknowns, _, idx = self._determine_block_params(i, j, grid)
        
        # Extract target area
        start_row = max(0, idx - neighborhood_size * n_unknowns)
        start_col = max(0, idx - neighborhood_size * n_unknowns)
        
        end_row = min(A_dense.shape[0], idx + (neighborhood_size + 1) * n_unknowns)
        end_col = min(A_dense.shape[1], idx + (neighborhood_size + 1) * n_unknowns)
        
        block = A_dense[start_row:end_row, start_col:end_col]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        plt.imshow(block != 0, cmap='Blues', interpolation='none')
        
        # Set title
        if is_2d:
            plt.title(f"Matrix Structure Around Grid Point ({i},{j}) (Size: {block.shape[0]}×{block.shape[1]})")
        else:
            plt.title(f"Matrix Structure Around Grid Point {i} (Size: {block.shape[0]}×{block.shape[1]})")
        
        # Show grid point boundaries
        for pos in np.arange(n_unknowns, block.shape[0], n_unknowns):
            plt.axhline(y=pos-0.5, color='r', linestyle='-', alpha=0.3)
            plt.axvline(x=pos-0.5, color='r', linestyle='-', alpha=0.3)
        
        plt.colorbar(label="Value Presence")
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Neighborhood matrix saved: {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def _to_numpy_dense(self, A: Any) -> np.ndarray:
        """Convert matrix to NumPy dense matrix"""
        if hasattr(A, 'get'):
            return A.get().toarray() if hasattr(A, 'toarray') else A.get()
        else:
            return A.toarray() if hasattr(A, 'toarray') else A
    
    def _count_nonzeros(self, A: Any, A_dense: np.ndarray) -> int:
        """Count non-zero elements"""
        return A.nnz if hasattr(A, 'nnz') else np.count_nonzero(A_dense)
    
    def _determine_block_params(self, i: int, j: Optional[int], grid: Optional[Grid]) -> Tuple[bool, int, List[str], int]:
        """Determine block parameters"""
        is_2d = (grid.is_2d if grid is not None else j is not None)
        
        if is_2d:
            n_unknowns = 7
            labels = ["ψ", "ψ_x", "ψ_xx", "ψ_xxx", "ψ_y", "ψ_yy", "ψ_yyy"]
            
            if j is None:
                raise ValueError("For 2D, j must be specified")
            
            if grid is not None:
                nx = grid.nx_points
            else:
                raise ValueError("For 2D, grid must be specified")
            
            idx = (j * nx + i) * n_unknowns
        else:
            n_unknowns = 4
            labels = ["ψ", "ψ'", "ψ''", "ψ'''"]
            idx = i * n_unknowns
        
        return is_2d, n_unknowns, labels, idx
    
    def _add_grid_lines(self, grid: Grid, matrix_size: int) -> None:
        """Add grid lines to matrix structure figure"""
        if not grid.is_2d:  # 1D grid
            n = grid.n_points
            unknowns = 4  # 4 unknowns for 1D (ψ, ψ', ψ'', ψ''')
            for i in range(1, n):
                plt.axhline(y=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
                plt.axvline(x=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
        else:  # 2D grid
            # For 2D grid, matrix size is proportional to grid points
            if matrix_size != grid.nx_points * grid.ny_points * 7:
                # If matrix size doesn't match expected, estimate lines
                rows = int(np.sqrt(matrix_size / 7))
                for i in range(1, rows):
                    plt.axhline(y=i*7-0.5, color='r', linestyle='-', alpha=0.2)
                    plt.axvline(x=i*7-0.5, color='r', linestyle='-', alpha=0.2)
            else:
                # Draw accurate grid lines
                nx, ny = grid.nx_points, grid.ny_points
                unknowns = 7  # 7 unknowns for 2D
                for i in range(1, nx * ny):
                    plt.axhline(y=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
                    plt.axvline(x=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
    
    def _annotate_block_values(self, block: np.ndarray) -> None:
        """Display block values as annotations"""
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                value = block[i, j]
                if abs(value) > 1e-10:
                    plt.text(j, i, f"{value:.2g}", ha='center', va='center', 
                             color='black' if abs(value) < 0.5 else 'white',
                             fontsize=9)

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
    
    def visualize_matrix_structure(self, A: Any, grid: Grid, title: str, prefix: str) -> None:
        """全体構造の可視化"""
        save_path = os.path.join(self.output_dir, f"{prefix}_structure.png")
        self.visualizer.visualize_structure(A, grid, title, save_path)
    
    def _visualize_sample_points(self, A: Any, grid: Grid, prefix: str) -> None:
        """サンプル点（中央・境界・角など）を視覚化"""
        if grid.is_2d:
            nx, ny = grid.nx_points, grid.ny_points
            # 角と中央の点を可視化
            for i, j in [(0, 0), (nx//2, ny//2), (nx-1, ny-1)]:
                if i < nx and j < ny:
                    save_path = os.path.join(self.output_dir, f"{prefix}_block_i{i}_j{j}.png")
                    self.visualizer.visualize_block(A, i, j, grid, save_path)
            
            # 中央付近の周辺関係
            i, j = nx//2, ny//2
            save_path = os.path.join(self.output_dir, f"{prefix}_neighborhood_center.png")
            self.visualizer.visualize_neighborhood(A, i, j, grid, 1, save_path)
        else:
            n = grid.n_points
            # 左端、中央、右端の点を可視化
            for i in [0, n//2, n-1]:
                if i < n:
                    save_path = os.path.join(self.output_dir, f"{prefix}_block_i{i}.png")
                    self.visualizer.visualize_block(A, i, grid=grid, save_path=save_path)
            
            # 中央付近の周辺関係
            i = n//2
            save_path = os.path.join(self.output_dir, f"{prefix}_neighborhood_center.png")
            self.visualizer.visualize_neighborhood(A, i, grid=grid, neighborhood_size=1, save_path=save_path)
    
    def _print_analysis_results(self, name: str, grid: Grid, total_size: int, nnz: int, sparsity: float) -> None:
        """分析結果を表示"""
        print(f"\n行列構造分析{' ('+name+')' if name else ''}:")
        
        if grid.is_2d:
            print(f"  グリッドサイズ: {grid.nx_points}×{grid.ny_points} 点")
        else:
            print(f"  グリッドサイズ: {grid.n_points} 点")
            
        print(f"  行列サイズ: {total_size} × {total_size}")
        print(f"  非ゼロ要素数: {nnz}")
        print(f"  疎性率: {sparsity:.6f} ({sparsity*100:.2f}%)")
        
        # メモリ使用量の推定
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8 bytes per double
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 8 bytes for value + 4 bytes for indices
        memory_ratio = memory_dense_MB / memory_sparse_MB if memory_sparse_MB > 0 else float('inf')
        
        print(f"  メモリ使用量(密行列): {memory_dense_MB:.2f} MB")
        print(f"  メモリ使用量(疎行列): {memory_sparse_MB:.2f} MB")
        print(f"  メモリ削減率: {memory_ratio:.1f}倍")
    
    def _get_prefix(self, name: str, grid_info: str) -> str:
        """ファイル名のプレフィックスを取得"""
        return f"{name.lower()}_{grid_info}" if name else f"matrix_{grid_info}"
    
    def _get_title(self, name: str, is_2d: bool) -> str:
        """図のタイトルを取得"""
        return f"{name} 方程式システム行列" if name else f"{'2次元' if is_2d else '1次元'} 方程式システム行列"


def verify_1d_system() -> None:
    """1次元方程式システムの検証"""
    print("\n--- 1次元方程式システムの検証 ---")
    
    try:
        # グリッドの作成
        n_points = 21
        grid = Grid(n_points, x_range=(-1.0, 1.0))
        
        # テスト関数の取得
        test_funcs = TestFunctionFactory.create_standard_functions()
        test_func = test_funcs[0]  # 最初の関数を使用
        
        # 方程式システムの作成
        system = EquationSystem(grid)
        
        # 方程式セットの取得と設定
        equation_set = EquationSet.create("poisson", dimension=1)
        equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
        
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