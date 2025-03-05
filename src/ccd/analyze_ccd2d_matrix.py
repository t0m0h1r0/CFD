#!/usr/bin/env python3
"""
CCD2Dソルバーの行列構造分析ツール

CCD2Dソルバーの行列構造をさまざまな視点から可視化・分析します。
"""

import argparse
import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

# インポートパスを追加
sys.path.append('.')
sys.path.append('..')

# モジュールをインポート
from grid_2d_config import Grid2DConfig
from fast_ccd2d_solver import FastSparseCCD2DSolver
from matrix_visualization import analyze_and_visualize_matrix


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="CCD2Dソルバーの行列構造分析")
    
    parser.add_argument("--nx", type=int, default=32, help="x方向のグリッド点数")
    parser.add_argument("--ny", type=int, default=32, help="y方向のグリッド点数")
    parser.add_argument("--xrange", type=float, nargs=2, default=[0.0, 1.0], help="x軸の範囲 (開始点 終了点)")
    parser.add_argument("--yrange", type=float, nargs=2, default=[0.0, 1.0], help="y軸の範囲 (開始点 終了点)")
    parser.add_argument("--coeffs", type=float, nargs="+", default=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 
                       help="係数 [a, b, c, d, e, f]: a*u + b*ux + c*uxx + d*uy + e*uyy + f*uxy")
    parser.add_argument("--out", type=str, default="matrix_analysis", help="出力ディレクトリ")
    parser.add_argument("--dirichlet", action="store_true", help="ディリクレ境界条件を使用（デフォルトはゼロ値）")
    parser.add_argument("--compare", action="store_true", help="異なるグリッドサイズで構造を比較")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64], help="比較するグリッドサイズ")
    parser.add_argument("--dpi", type=int, default=150, help="出力画像の解像度")
    
    return parser.parse_args()


def create_grid_config(nx: int, ny: int, x_range: Tuple[float, float], y_range: Tuple[float, float], 
                      coeffs: list, use_dirichlet: bool = False) -> Grid2DConfig:
    """
    グリッド設定を作成
    
    Args:
        nx: x方向のグリッド点数
        ny: y方向のグリッド点数
        x_range: x軸の範囲 (開始点, 終了点)
        y_range: y軸の範囲 (開始点, 終了点)
        coeffs: 係数 [a, b, c, d, e, f]
        use_dirichlet: ディリクレ境界条件を使用するかどうか
        
    Returns:
        グリッド設定
    """
    # グリッド幅を計算
    hx = (x_range[1] - x_range[0]) / (nx - 1)
    hy = (y_range[1] - y_range[0]) / (ny - 1)
    
    # グリッド設定を作成
    grid_config = Grid2DConfig(
        nx=nx,
        ny=ny,
        hx=hx,
        hy=hy,
        coeffs=coeffs
    )
    
    # ディリクレ境界条件を設定（必要な場合）
    if use_dirichlet:
        # ゼロ値の境界条件
        grid_config.dirichlet_values = {
            'left': np.zeros(ny),
            'right': np.zeros(ny),
            'bottom': np.zeros(nx),
            'top': np.zeros(nx)
        }
    
    return grid_config


def analyze_matrix_structure(args):
    """行列構造を分析"""
    # グリッド設定を作成
    grid_config = create_grid_config(
        args.nx, args.ny, args.xrange, args.yrange, args.coeffs, args.dirichlet
    )
    
    # ソルバーを初期化（行列の構築のみ）
    print(f"グリッドサイズ {args.nx}x{args.ny} の行列を構築中...")
    solver = FastSparseCCD2DSolver(grid_config, precompute=False)
    
    # システム行列を構築
    matrix = solver._build_2d_operator_matrix_sparse()
    
    # 境界条件を適用
    matrix = solver._apply_boundary_conditions(matrix)
    
    # 出力ディレクトリを作成
    os.makedirs(args.out, exist_ok=True)
    
    # 係数文字列を作成
    coeff_str = '_'.join(map(str, args.coeffs))
    
    # 行列の分析と可視化
    prefix = f"ccd2d_{args.nx}x{args.ny}_coeffs_{coeff_str[:10]}"
    if args.dirichlet:
        prefix += "_dirichlet"
    
    analyze_and_visualize_matrix(matrix, output_dir=args.out, prefix=prefix, dpi=args.dpi)


def compare_matrix_structures(args):
    """異なるグリッドサイズでの行列構造を比較"""
    # 出力ディレクトリを作成
    os.makedirs(args.out, exist_ok=True)
    
    # 係数文字列を作成
    coeff_str = '_'.join(map(str, args.coeffs))
    coeff_short = coeff_str[:10]  # 短い表示用
    
    # 4x4の大きなプロットを作成
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"CCD2D Matrix Structure Comparison (coeffs=[{coeff_short}])", fontsize=20)
    
    # グリッドサイズごとに行列パターンを生成・可視化
    for i, size in enumerate(args.sizes):
        print(f"グリッドサイズ {size}x{size} の行列を構築中...")
        
        # グリッド設定を作成
        grid_config = create_grid_config(
            size, size, args.xrange, args.yrange, args.coeffs, args.dirichlet
        )
        
        # ソルバーを初期化（行列の構築のみ）
        solver = FastSparseCCD2DSolver(grid_config, precompute=False)
        
        # システム行列を構築
        matrix = solver._build_2d_operator_matrix_sparse()
        
        # 境界条件を適用
        matrix = solver._apply_boundary_conditions(matrix)
        
        # サブプロットに表示
        plt.subplot(2, 2, i+1)
        plt.spy(matrix, markersize=0.2)
        
        # 行列情報を表示
        matrix_size = matrix.shape[0]
        nnz = matrix.nnz
        density = nnz / (matrix_size * matrix_size)
        
        plt.title(f"Grid: {size}x{size}, Matrix: {matrix_size}x{matrix_size}\nNNZ: {nnz}, Density: {density:.2e}")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # プロットを保存
    if args.dirichlet:
        suffix = "_dirichlet"
    else:
        suffix = ""
    
    plt.savefig(os.path.join(args.out, f"matrix_comparison_coeffs_{coeff_short}{suffix}.png"), dpi=args.dpi)
    plt.show()


def analyze_kronecker_structure(args):
    """クロネッカー積の構造を詳細に分析"""
    # グリッド設定を作成
    grid_config = create_grid_config(
        args.nx, args.ny, args.xrange, args.yrange, args.coeffs, args.dirichlet
    )
    
    # ソルバーを初期化
    solver = FastSparseCCD2DSolver(grid_config, precompute=False)
    
    # 1次元演算子行列を取得
    Lx, Ly = solver._build_1d_operator_matrices()
    
    # 出力ディレクトリを作成
    os.makedirs(args.out, exist_ok=True)
    
    # 1次元演算子行列の可視化
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.spy(Lx, markersize=1.0)
    plt.title(f"X-direction Operator (size: {Lx.shape[0]}x{Lx.shape[1]}, nnz: {Lx.nnz})")
    
    plt.subplot(1, 2, 2)
    plt.spy(Ly, markersize=1.0)
    plt.title(f"Y-direction Operator (size: {Ly.shape[0]}x{Ly.shape[1]}, nnz: {Ly.nnz})")
    
    plt.tight_layout()
    
    # 保存
    coeff_str = '_'.join(map(str, args.coeffs))
    coeff_short = coeff_str[:10]
    suffix = "_dirichlet" if args.dirichlet else ""
    
    plt.savefig(os.path.join(args.out, f"1d_operators_{args.nx}x{args.ny}_coeffs_{coeff_short}{suffix}.png"), dpi=args.dpi)
    
    # クロネッカー積の構造の可視化（小さいサイズのみ）
    if args.nx <= 16 and args.ny <= 16:
        # 単位行列
        Ix = sp.eye(Lx.shape[0])
        Iy = sp.eye(Ly.shape[0])
        
        # クロネッカー積
        L_x_2d = sp.kron(Iy, Lx)
        L_y_2d = sp.kron(Ly, Ix)
        
        # 最終的な行列
        L_2d = L_x_2d + L_y_2d
        
        # 4つの行列を可視化
        plt.figure(figsize=(20, 16))
        
        plt.subplot(2, 2, 1)
        plt.spy(L_x_2d, markersize=0.5)
        plt.title(f"Kronecker Product: I_y ⊗ L_x\nSize: {L_x_2d.shape[0]}x{L_x_2d.shape[1]}, NNZ: {L_x_2d.nnz}")
        
        plt.subplot(2, 2, 2)
        plt.spy(L_y_2d, markersize=0.5)
        plt.title(f"Kronecker Product: L_y ⊗ I_x\nSize: {L_y_2d.shape[0]}x{L_y_2d.shape[1]}, NNZ: {L_y_2d.nnz}")
        
        plt.subplot(2, 2, 3)
        plt.spy(L_2d, markersize=0.5)
        plt.title(f"Final Matrix: L_x_2d + L_y_2d\nSize: {L_2d.shape[0]}x{L_2d.shape[1]}, NNZ: {L_2d.nnz}")
        
        # ブロック構造の可視化
        plt.subplot(2, 2, 4)
        block_size = 4  # 各点で4つの変数
        
        # 行列のサイズ
        m, n = L_2d.shape
        
        # ブロック数を計算
        m_blocks = (m + block_size - 1) // block_size
        n_blocks = (n + block_size - 1) // block_size
        
        # ブロックごとの非ゼロ要素数を格納する配列
        block_density = np.zeros((m_blocks, n_blocks))
        
        # 各ブロックの非ゼロ要素数をカウント
        for i in range(m_blocks):
            i_start = i * block_size
            i_end = min((i + 1) * block_size, m)
            
            for j in range(n_blocks):
                j_start = j * block_size
                j_end = min((j + 1) * block_size, n)
                
                # ブロック内の要素数
                block_nnz = L_2d[i_start:i_end, j_start:j_end].nnz
                
                # ブロックサイズ（端のブロックは小さくなる可能性あり）
                block_size_actual = (i_end - i_start) * (j_end - j_start)
                
                if block_size_actual > 0:
                    # 密度 = 非ゼロ要素数 / ブロックサイズ
                    block_density[i, j] = block_nnz / block_size_actual
        
        plt.imshow(block_density, cmap='viridis')
        plt.colorbar(label='Block Density')
        plt.title(f"Block Structure (block size = {block_size})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"kronecker_structure_{args.nx}x{args.ny}_coeffs_{coeff_short}{suffix}.png"), dpi=args.dpi)
        plt.show()


def analyze_block_structure(args):
    """グリッド構造とブロック構造の関係を詳細に分析"""
    # グリッド設定を作成
    grid_config = create_grid_config(
        args.nx, args.ny, args.xrange, args.yrange, args.coeffs, args.dirichlet
    )
    
    # ソルバーを初期化
    solver = FastSparseCCD2DSolver(grid_config, precompute=False)
    
    # システム行列を構築
    matrix = solver._build_2d_operator_matrix_sparse()
    
    # 境界条件を適用
    matrix = solver._apply_boundary_conditions(matrix)
    
    # 行と列における非ゼロ要素の分布を可視化
    m, n = matrix.shape
    
    # 行と列ごとの非ゼロ要素数
    matrix_csr = matrix.tocsr()
    row_nnz = np.diff(matrix_csr.indptr)
    
    matrix_csc = matrix.tocsc()
    col_nnz = np.diff(matrix_csc.indptr)
    
    # 出力ディレクトリを作成
    os.makedirs(args.out, exist_ok=True)
    
    # 係数文字列を作成
    coeff_str = '_'.join(map(str, args.coeffs))
    coeff_short = coeff_str[:10]
    suffix = "_dirichlet" if args.dirichlet else ""
    
    # 行と列の非ゼロ要素分布の可視化
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.plot(row_nnz)
    plt.grid(True)
    plt.title(f"Nonzeros per Row (average: {np.mean(row_nnz):.2f})")
    plt.xlabel("Row Index")
    plt.ylabel("Nonzero Count")
    
    plt.subplot(1, 2, 2)
    plt.plot(col_nnz)
    plt.grid(True)
    plt.title(f"Nonzeros per Column (average: {np.mean(col_nnz):.2f})")
    plt.xlabel("Column Index")
    plt.ylabel("Nonzero Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"nnz_distribution_{args.nx}x{args.ny}_coeffs_{coeff_short}{suffix}.png"), dpi=args.dpi)
    
    # グリッド点と行列インデックスの対応関係の可視化
    depth = 4  # 各点での未知数の数
    
    plt.figure(figsize=(15, 12))
    
    # スパースパターン
    plt.subplot(2, 2, 1)
    plt.spy(matrix, markersize=0.1)
    plt.title(f"Sparsity Pattern ({matrix.nnz} nonzeros, density: {matrix.nnz / (m * n):.2e})")
    
    # 2Dグリッド点のインデックスマップ
    grid_indices = np.arange(args.nx * args.ny).reshape(args.ny, args.nx)
    
    plt.subplot(2, 2, 2)
    plt.imshow(grid_indices, cmap='viridis')
    plt.colorbar(label='Grid Point Index')
    plt.title(f"Grid Point Indices ({args.nx}x{args.ny} grid)")
    
    # 行列インデックスとグリッド点の対応関係
    matrix_rows = np.arange(m) // depth
    matrix_vars = np.arange(m) % depth
    
    plt.subplot(2, 2, 3)
    plt.scatter(range(m), matrix_rows, c=matrix_vars, cmap='tab10', s=1)
    plt.colorbar(label='Variable Index (0-3)')
    plt.title("Matrix Row -> Grid Point Mapping")
    plt.xlabel("Matrix Row Index")
    plt.ylabel("Grid Point Index")
    
    # ブロック構造の種類を抽出
    if args.nx <= 32 and args.ny <= 32:
        # 小さな行列のみでブロックパターンの詳細分析
        patterns = {}
        max_patterns = 20
        
        for i in range(0, m, depth):
            if len(patterns) >= max_patterns:
                break
            
            # 各グリッド点に対応する4x4ブロックを抽出
            row_indices = []
            col_indices = []
            values = []
            
            for di in range(depth):
                row_i = i + di
                if row_i >= m:
                    continue
                
                # 行iの非ゼロ要素を取得
                row_start = matrix_csr.indptr[row_i]
                row_end = matrix_csr.indptr[row_i + 1]
                
                for j in range(row_start, row_end):
                    col_j = matrix_csr.indices[j]
                    val = matrix_csr.data[j]
                    
                    # グリッド点とサブ変数のインデックスに変換
                    grid_col = col_j // depth
                    var_col = col_j % depth
                    
                    row_indices.append(di)
                    col_indices.append(var_col)
                    values.append(val)
            
            # ブロックパターンの文字列表現
            pattern = ','.join(f"({r},{c})" for r, c in zip(row_indices, col_indices))
            
            if pattern not in patterns:
                patterns[pattern] = 1
            else:
                patterns[pattern] += 1
        
        # ブロックパターンの数を表示
        plt.subplot(2, 2, 4)
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        labels = [f"Pattern {i+1}" for i in range(len(sorted_patterns))]
        counts = [p[1] for p in sorted_patterns]
        
        plt.bar(labels, counts)
        plt.title(f"Block Pattern Distribution ({len(patterns)} unique patterns)")
        plt.xlabel("Pattern ID")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        # パターンの詳細をプリント
        print(f"\n=== ブロックパターンの分析 ===")
        for i, (pattern, count) in enumerate(sorted_patterns):
            print(f"パターン {i+1} ({count}個): {pattern}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"grid_matrix_mapping_{args.nx}x{args.ny}_coeffs_{coeff_short}{suffix}.png"), dpi=args.dpi)
    plt.show()


def main():
    """メイン関数"""
    args = parse_args()
    
    if args.compare:
        # 異なるグリッドサイズでの比較
        compare_matrix_structures(args)
    else:
        # 基本的な行列構造の分析
        analyze_matrix_structure(args)
        
        # クロネッカー積の構造分析
        analyze_kronecker_structure(args)
        
        # ブロック構造と格子点との関係分析
        analyze_block_structure(args)


if __name__ == "__main__":
    main()