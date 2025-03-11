# verify.py
import os
import numpy as np
import matplotlib.pyplot as plt

# 統合されたCCDコードをインポート
from grid1d import Grid
from grid2d import Grid2D
from equation_system import EquationSystem
from test_functions1d import TestFunctionFactory
from test_functions2d import TestFunction2DGenerator
from equation_sets1d import EquationSet
from equation_sets2d import EquationSet2D

# ディレクトリ作成（結果保存用）
os.makedirs("matrix_verification", exist_ok=True)

# ================================================
# 行列構造可視化用の関数
# ================================================

def visualize_matrix_structure(A, grid=None, title="行列構造の可視化", save_path=None):
    """
    行列の構造を可視化する関数
    
    Args:
        A: CSR形式の疎行列
        grid: グリッドオブジェクト（オプション）
        title: プロットのタイトル
        save_path: 保存パス（指定しない場合は表示のみ）
    """
    # CuPy配列をNumPy配列に変換
    if hasattr(A, 'get'):
        A_dense = A.get().toarray() if hasattr(A, 'toarray') else A.get()
    else:
        A_dense = A.toarray() if hasattr(A, 'toarray') else A
    
    # 行列サイズと非ゼロ要素数
    total_size = A.shape[0]
    nnz = A.nnz if hasattr(A, 'nnz') else np.count_nonzero(A_dense)
    sparsity = 1.0 - (float(nnz) / (total_size * total_size))
    
    # 可視化
    plt.figure(figsize=(10, 8))
    plt.spy(A_dense, markersize=0.5, color='blue')
    plt.title(f"{title}\n(サイズ: {total_size}×{total_size}, 非ゼロ要素: {nnz}, 疎性: {sparsity:.4f})")
    plt.xlabel("列インデックス")
    plt.ylabel("行インデックス")
    
    # グリッド線を追加（グリッドが指定されている場合）
    if grid is not None:
        is_2d = isinstance(grid, Grid2D)
        if not is_2d:  # 1D grid
            n = grid.n_points
            unknowns = 4  # 1Dの場合は4つの未知数 (ψ, ψ', ψ'', ψ''')
            for i in range(1, n):
                plt.axhline(y=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
                plt.axvline(x=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
        else:  # 2D grid
            nx, ny = grid.nx_points, grid.ny_points
            unknowns = 7  # 2Dの場合は7つの未知数 (ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy)
            for i in range(1, nx * ny):
                plt.axhline(y=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
                plt.axvline(x=i*unknowns-0.5, color='r', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"行列構造を保存しました: {save_path}")
    else:
        plt.show()

def visualize_matrix_block(A, i, j=None, is_2d=False, nx=None, save_path=None):
    """
    行列の特定ブロックを詳細に可視化
    
    Args:
        A: CSR形式の疎行列
        i, j: グリッドインデックス（1Dの場合はiのみ使用）
        is_2d: 2D問題かどうか
        nx: 2D問題のx方向のグリッド点数
        save_path: 保存パス（指定しない場合は表示のみ）
    """
    # CuPy配列をNumPy配列に変換
    if hasattr(A, 'get'):
        A_dense = A.get().toarray() if hasattr(A, 'toarray') else A.get()
    else:
        A_dense = A.toarray() if hasattr(A, 'toarray') else A
    
    # 未知数の数とラベルを設定
    if is_2d:
        n_unknowns = 7
        labels = ["ψ", "ψ_x", "ψ_xx", "ψ_xxx", "ψ_y", "ψ_yy", "ψ_yyy"]
        if j is None:
            raise ValueError("2Dの場合、iとjの両方を指定する必要があります")
        if nx is None:
            raise ValueError("2Dの場合、nxを指定する必要があります")
        idx = (j * nx + i) * n_unknowns
    else:
        n_unknowns = 4
        labels = ["ψ", "ψ'", "ψ''", "ψ'''"]
        idx = i * n_unknowns
    
    # ブロックを抽出（または大きすぎる場合はサブセット）
    if idx + n_unknowns <= A_dense.shape[0]:
        block = A_dense[idx:idx+n_unknowns, idx:idx+n_unknowns]
    else:
        print(f"警告: インデックス {idx} が行列のサイズを超えています")
        return
    
    # 可視化
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(block) > 1e-10, cmap='Blues', interpolation='none')
    
    if is_2d:
        plt.title(f"格子点 ({i},{j}) の行列ブロック")
    else:
        plt.title(f"格子点 {i} の行列ブロック")
    
    plt.xticks(range(n_unknowns), labels, rotation=45)
    plt.yticks(range(n_unknowns), labels)
    
    # 値を表示
    for ii in range(n_unknowns):
        for jj in range(n_unknowns):
            value = block[ii, jj]
            if abs(value) > 1e-10:
                plt.text(jj, ii, f"{value:.2g}", ha='center', va='center', 
                        color='black' if abs(value) < 0.5 else 'white',
                        fontsize=9)
    
    plt.colorbar(label="値の有無")
    plt.grid(True, color='gray', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"ブロック行列を保存しました: {save_path}")
    else:
        plt.show()

def visualize_matrix_neighborhood(A, i, j=None, is_2d=False, nx=None, neighborhood_size=1, save_path=None):
    """
    行列の特定点の周囲を可視化
    
    Args:
        A: CSR形式の疎行列
        i, j: グリッドインデックス（1Dの場合はiのみ使用）
        is_2d: 2D問題かどうか
        nx: 2D問題のx方向のグリッド点数
        neighborhood_size: 隣接する格子点の数
        save_path: 保存パス（指定しない場合は表示のみ）
    """
    # CuPy配列をNumPy配列に変換
    if hasattr(A, 'get'):
        A_dense = A.get().toarray() if hasattr(A, 'toarray') else A.get()
    else:
        A_dense = A.toarray() if hasattr(A, 'toarray') else A
    
    # 未知数の数を設定
    if is_2d:
        n_unknowns = 7
        if j is None or nx is None:
            raise ValueError("2Dの場合、jとnxを指定する必要があります")
        idx = (j * nx + i) * n_unknowns
    else:
        n_unknowns = 4
        idx = i * n_unknowns
    
    # 対象領域を抽出
    start_row = max(0, idx - neighborhood_size * n_unknowns)
    start_col = max(0, idx - neighborhood_size * n_unknowns)
    
    end_row = min(A_dense.shape[0], idx + (neighborhood_size + 1) * n_unknowns)
    end_col = min(A_dense.shape[1], idx + (neighborhood_size + 1) * n_unknowns)
    
    block = A_dense[start_row:end_row, start_col:end_col]
    
    # 可視化
    plt.figure(figsize=(12, 10))
    plt.imshow(block != 0, cmap='Blues', interpolation='none')
    
    if is_2d:
        plt.title(f"格子点 ({i},{j}) 周辺の行列構造 (サイズ: {block.shape[0]}×{block.shape[1]})")
    else:
        plt.title(f"格子点 {i} 周辺の行列構造 (サイズ: {block.shape[0]}×{block.shape[1]})")
    
    # グリッド点の境界を表示
    for pos in np.arange(n_unknowns, block.shape[0], n_unknowns):
        plt.axhline(y=pos-0.5, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=pos-0.5, color='r', linestyle='-', alpha=0.3)
    
    plt.colorbar(label="値の有無")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"周辺行列を保存しました: {save_path}")
    else:
        plt.show()

def analyze_equation_system(system, name=""):
    """
    方程式システムを分析し、行列構造を検証する
    
    Args:
        system: EquationSystem のインスタンス
        name: 識別用の名前
    """
    # 行列システムを構築
    A, b = system.build_matrix_system()
    
    # グリッド情報とシステムの次元を取得
    is_2d = system.is_2d
    
    if is_2d:
        grid = system.grid
        nx, ny = grid.nx_points, grid.ny_points
        grid_info = f"{nx}x{ny}"
    else:
        grid = system.grid
        n = grid.n_points
        grid_info = f"{n}"
    
    # 基本統計情報
    total_size = A.shape[0]
    nnz = A.nnz if hasattr(A, 'nnz') else np.count_nonzero(A.toarray())
    sparsity = 1.0 - (nnz / (total_size * total_size))
    
    print(f"\n行列構造分析{' ('+name+')' if name else ''}:")
    if is_2d:
        print(f"  グリッドサイズ: {nx}×{ny} 点")
    else:
        print(f"  グリッドサイズ: {n} 点")
    print(f"  行列サイズ: {total_size} × {total_size}")
    print(f"  非ゼロ要素数: {nnz}")
    print(f"  疎性率: {sparsity:.6f} ({sparsity*100:.2f}%)")
    
    # ファイル名のプレフィックス
    prefix = f"{name.lower()}_{grid_info}" if name else f"{'2d' if is_2d else '1d'}_{grid_info}"
    
    # 全体構造の可視化
    title = f"{name} 方程式システム行列" if name else f"{'2次元' if is_2d else '1次元'} 方程式システム行列"
    visualize_matrix_structure(A, system.grid, title, 
                               save_path=f"matrix_verification/{prefix}_structure.png")
    
    # 特定点の詳細可視化
    if is_2d:
        # 角と中央の点を可視化
        nx, ny = grid.nx_points, grid.ny_points
        for i, j in [(0, 0), (nx//2, ny//2), (nx-1, ny-1)]:
            if i < nx and j < ny:
                visualize_matrix_block(A, i, j, is_2d=True, nx=nx,
                                      save_path=f"matrix_verification/{prefix}_block_i{i}_j{j}.png")
        
        # 中央付近の周辺関係
        i, j = nx//2, ny//2
        visualize_matrix_neighborhood(A, i, j, is_2d=True, nx=nx, neighborhood_size=1,
                                     save_path=f"matrix_verification/{prefix}_neighborhood_center.png")
    else:
        # 左端、中央、右端の点を可視化
        n = grid.n_points
        for i in [0, n//2, n-1]:
            if i < n:
                visualize_matrix_block(A, i, is_2d=False, 
                                      save_path=f"matrix_verification/{prefix}_block_i{i}.png")
        
        # 中央付近の周辺関係
        i = n//2
        visualize_matrix_neighborhood(A, i, is_2d=False, neighborhood_size=1,
                                     save_path=f"matrix_verification/{prefix}_neighborhood_center.png")
    
    return {"size": total_size, "nnz": nnz, "sparsity": sparsity}

# ================================================
# 検証用関数
# ================================================

def verify_1d_system():
    """1次元方程式システムの検証"""
    print("\n--- 1次元方程式システムの検証 ---")
    
    # グリッドの作成
    n_points = 21
    grid = Grid(n_points, (-1.0, 1.0))
    
    # テスト関数の取得
    test_funcs = TestFunctionFactory.create_standard_functions()
    test_func = test_funcs[0]  # 最初の関数を使用
    
    # 統合された方程式システムの作成
    system = EquationSystem(grid)
    
    # 方程式セットの取得と設定
    equation_set = EquationSet.create("poisson")
    equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
    
    # 行列構造の分析と可視化
    analyze_equation_system(system, "Poisson1D")

def verify_2d_system():
    """2次元方程式システムの検証"""
    print("\n--- 2次元方程式システムの検証 ---")
    
    # グリッドの作成
    nx, ny = 11, 11
    grid = Grid2D(nx, ny, (-1.0, 1.0), (-1.0, 1.0))
    
    # テスト関数の取得
    test_funcs = TestFunction2DGenerator.create_standard_functions()
    test_func = test_funcs[0]  # 最初の関数を使用
    
    # 統合された方程式システムの作成
    system = EquationSystem(grid)
    
    # 方程式セットの取得と設定
    equation_set = EquationSet2D.create("poisson")
    equation_set.setup_equations(system, grid, test_func, use_dirichlet=True, use_neumann=True)
    
    # 行列構造の分析と可視化
    analyze_equation_system(system, "Poisson2D")

# ================================================
# メイン関数
# ================================================

def main():
    print("==== CCD行列構造検証ツール ====")
    
    # 1次元方程式システムの検証
    verify_1d_system()
    
    # 2次元方程式システムの検証
    verify_2d_system()
    
    print("\n検証が完了しました。結果は matrix_verification ディレクトリに保存されています。")

if __name__ == "__main__":
    main()