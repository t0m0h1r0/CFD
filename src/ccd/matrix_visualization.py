"""
行列構造可視化モジュール

スパース行列の構造と密度を様々な形式で可視化するユーティリティ関数群。
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os
from matplotlib.colors import LogNorm
from typing import Optional, Tuple


def visualize_sparsity_pattern(
    matrix: sp.spmatrix,
    title: str = "Sparsity Pattern",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """
    スパース行列の非ゼロ要素パターンを可視化

    Args:
        matrix: 可視化するスパース行列
        title: プロットのタイトル
        figsize: 図のサイズ
        save_path: 保存先のパス（省略可）
        dpi: 保存する画像の解像度
    """
    plt.figure(figsize=figsize)
    plt.spy(matrix, markersize=0.5)
    plt.title(
        f"{title}\nSize: {matrix.shape}, Nonzeros: {matrix.nnz}, Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2e}"
    )
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"スパース行列パターンを {save_path} に保存しました")

    plt.show()


def visualize_matrix_blocks(
    matrix: sp.spmatrix,
    block_size: int,
    title: str = "Matrix Block Structure",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """
    スパース行列のブロック構造を可視化

    ブロックごとの非ゼロ要素の分布をヒートマップで表示

    Args:
        matrix: 可視化するスパース行列
        block_size: ブロックのサイズ
        title: プロットのタイトル
        figsize: 図のサイズ
        save_path: 保存先のパス（省略可）
        dpi: 保存する画像の解像度
    """
    # 行列のサイズ
    m, n = matrix.shape

    # ブロック数を計算
    m_blocks = (m + block_size - 1) // block_size
    n_blocks = (n + block_size - 1) // block_size

    # ブロックごとの非ゼロ要素数を格納する配列
    block_density = np.zeros((m_blocks, n_blocks))

    # CSR形式に変換して処理を高速化
    matrix_csr = matrix.tocsr()

    # 各ブロックの非ゼロ要素数をカウント
    for i in range(m_blocks):
        i_start = i * block_size
        i_end = min((i + 1) * block_size, m)

        for j in range(n_blocks):
            j_start = j * block_size
            j_end = min((j + 1) * block_size, n)

            # ブロック内の要素数
            block_nnz = matrix_csr[i_start:i_end, j_start:j_end].nnz

            # ブロックサイズ（端のブロックは小さくなる可能性あり）
            block_size_actual = (i_end - i_start) * (j_end - j_start)

            if block_size_actual > 0:
                # 密度 = 非ゼロ要素数 / ブロックサイズ
                block_density[i, j] = block_nnz / block_size_actual

    # 可視化
    plt.figure(figsize=figsize)

    # LogNorm使用（非ゼロブロックの違いを強調）
    # vmin = 0より大きい値を設定して0密度のブロックを特別扱い
    vmin = (
        np.min(block_density[block_density > 0]) if np.any(block_density > 0) else 1e-10
    )
    im = plt.imshow(block_density, cmap="viridis", norm=LogNorm(vmin=vmin, vmax=1.0))

    plt.colorbar(im, label="Block Density (log scale)")

    # 軸ラベル
    plt.xlabel(f"Column Block Index (block size = {block_size})")
    plt.ylabel(f"Row Block Index (block size = {block_size})")

    # タイトル
    plt.title(
        f"{title}\nMatrix Size: {matrix.shape}, NNZ: {matrix.nnz}, Density: {matrix.nnz / (m * n):.2e}"
    )

    # グリッドを追加（大きなブロック構造を見やすく）
    grid_step = 5  # 5ブロックごとにグリッド線
    if m_blocks > grid_step and n_blocks > grid_step:
        plt.grid(True, color="white", linestyle="-", linewidth=0.5)
        plt.xticks(np.arange(0, n_blocks, grid_step))
        plt.yticks(np.arange(0, m_blocks, grid_step))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"ブロック構造を {save_path} に保存しました")

    plt.show()


def visualize_matrix_values(
    matrix: sp.spmatrix,
    max_size: int = 1000,
    title: str = "Matrix Values",
    figsize: Tuple[int, int] = (12, 10),
    use_log: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """
    スパース行列の値の分布を可視化

    非ゼロ要素の値をヒートマップで表示

    Args:
        matrix: 可視化するスパース行列
        max_size: 可視化する最大サイズ
        title: プロットのタイトル
        figsize: 図のサイズ
        use_log: 対数スケールを使用するかどうか
        save_path: 保存先のパス（省略可）
        dpi: 保存する画像の解像度
    """
    # 行列のサイズ
    m, n = matrix.shape

    if m > max_size or n > max_size:
        # 大きな行列の場合は中心部分を抽出
        print(f"行列が大きいため、中心の {max_size}×{max_size} 部分のみ表示します")
        start_row = (m - max_size) // 2 if m > max_size else 0
        start_col = (n - max_size) // 2 if n > max_size else 0
        end_row = start_row + min(max_size, m)
        end_col = start_col + min(max_size, n)

        # 部分行列を抽出
        submatrix = matrix[start_row:end_row, start_col:end_col].toarray()
    else:
        # 小さな行列はそのまま表示
        submatrix = matrix.toarray()

    # 可視化
    plt.figure(figsize=figsize)

    # 非ゼロ要素のみを考慮してカラースケールを設定
    masked_matrix = np.ma.masked_where(submatrix == 0, submatrix)

    if use_log:
        # 絶対値をとり対数スケールで表示
        abs_vals = np.abs(masked_matrix)
        vmin = np.min(abs_vals[~abs_vals.mask]) if np.any(~abs_vals.mask) else 1e-10
        vmax = np.max(abs_vals)

        # 正と負の値を別々に表示
        pos_mask = masked_matrix > 0
        neg_mask = masked_matrix < 0

        # 正の値（赤系）
        pos_vals = np.ma.masked_where(~pos_mask, masked_matrix)
        plt.imshow(
            pos_vals,
            cmap="Reds",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="none",
        )

        # 負の値（青系）
        neg_vals = -np.ma.masked_where(~neg_mask, masked_matrix)  # 負の値を正に変換
        plt.imshow(
            neg_vals,
            cmap="Blues",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="none",
        )

        plt.colorbar(label="Absolute Element Value (log scale)", pad=0.05)
    else:
        # 線形スケールで表示
        vmax = max(abs(np.min(masked_matrix)), abs(np.max(masked_matrix)))
        plt.imshow(
            masked_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="none"
        )
        plt.colorbar(label="Element Value")

    # タイトル
    plt.title(
        f"{title}\nMatrix Size: {matrix.shape}, NNZ: {matrix.nnz}, Density: {matrix.nnz / (m * n):.2e}"
    )

    # 軸ラベル
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")

    # 表示サイズが小さい場合はグリッドを追加
    if min(submatrix.shape) <= 100:
        plt.grid(True, color="gray", linestyle="-", linewidth=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"行列値を {save_path} に保存しました")

    plt.show()


def visualize_matrix_properties(
    matrix: sp.spmatrix,
    title: str = "Matrix Properties",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """
    行列の詳細な性質を可視化

    非ゼロ要素分布、対角要素、行/列ごとの非ゼロ要素数などを可視化

    Args:
        matrix: 可視化するスパース行列
        title: プロットのタイトル
        figsize: 図のサイズ
        save_path: 保存先のパス（省略可）
        dpi: 保存する画像の解像度
    """
    # CSR形式に変換して処理を高速化
    matrix_csr = matrix.tocsr()

    # 行・列ごとの非ゼロ要素数を計算
    row_nnz = np.diff(matrix_csr.indptr)

    # CSC形式に変換して列ごとの非ゼロ要素数を計算
    matrix_csc = matrix.tocsc()
    col_nnz = np.diff(matrix_csc.indptr)

    # 対角要素を抽出
    diag_elements = matrix.diagonal()

    # 非ゼロ要素の値のヒストグラム用データ
    nnz_values = matrix_csr.data

    # サブプロット
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"{title}\nSize: {matrix.shape}, NNZ: {matrix.nnz}, Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2e}",
        fontsize=16,
    )

    # スパースパターン
    axes[0, 0].spy(matrix, markersize=0.3)
    axes[0, 0].set_title("Sparsity Pattern")

    # 行ごとの非ゼロ要素数
    axes[0, 1].plot(row_nnz, "b-")
    axes[0, 1].set_title("Nonzeros per Row")
    axes[0, 1].set_xlabel("Row Index")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True)

    # 列ごとの非ゼロ要素数
    axes[1, 0].plot(col_nnz, "r-")
    axes[1, 0].set_title("Nonzeros per Column")
    axes[1, 0].set_xlabel("Column Index")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True)

    # 非ゼロ要素の値のヒストグラム
    axes[1, 1].hist(nnz_values, bins=50, color="green", alpha=0.7)
    axes[1, 1].set_title("Distribution of Nonzero Values")
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True)

    # 対角要素情報を追加
    diag_info = f"Diagonal: min={np.min(diag_elements):.2e}, max={np.max(diag_elements):.2e}, mean={np.mean(diag_elements):.2e}"
    plt.figtext(
        0.5,
        0.01,
        diag_info,
        ha="center",
        fontsize=12,
        bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"行列プロパティを {save_path} に保存しました")

    plt.show()


def analyze_and_visualize_matrix(
    matrix: sp.spmatrix,
    output_dir: str = "matrix_analysis",
    prefix: str = "matrix",
    dpi: int = 150,
) -> None:
    """
    行列を総合的に分析・可視化して結果を保存

    Args:
        matrix: 分析するスパース行列
        output_dir: 出力ディレクトリ
        prefix: 出力ファイル名の接頭辞
        dpi: 画像の解像度
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 行列サイズ
    m, n = matrix.shape
    matrix_size_str = f"{m}x{n}"

    # 基本情報の表示
    print(f"\n===== 行列分析: {prefix} =====")
    print(f"サイズ: {matrix_size_str}")
    print(f"非ゼロ要素数: {matrix.nnz}")
    print(f"密度: {matrix.nnz / (m * n):.2e}")

    # 条件数計算（小さい行列のみ）
    if max(m, n) <= 2000:
        try:
            # フルランクで行が列以上なら特異値を計算
            if m >= n and matrix.rank() == n:
                svd_vals = sp.linalg.svds(
                    matrix, k=min(6, n - 1), return_singular_vectors=False
                )
                cond_number = max(svd_vals) / min(svd_vals)
                print(f"条件数（推定）: {cond_number:.2e}")
                print(
                    f"最大特異値: {max(svd_vals):.2e}, 最小特異値: {min(svd_vals):.2e}"
                )
        except:
            print("条件数の計算ができませんでした")

    # ブロックサイズの決定（行列サイズに基づく）
    if max(m, n) > 10000:
        block_size = 256
    elif max(m, n) > 5000:
        block_size = 128
    elif max(m, n) > 1000:
        block_size = 64
    else:
        block_size = 32

    # 1. スパース構造の可視化
    pattern_path = os.path.join(output_dir, f"{prefix}_sparsity_{matrix_size_str}.png")
    visualize_sparsity_pattern(
        matrix, title="Sparsity Pattern", save_path=pattern_path, dpi=dpi
    )

    # 2. ブロック構造の可視化
    block_path = os.path.join(
        output_dir, f"{prefix}_blocks_{matrix_size_str}_{block_size}.png"
    )
    visualize_matrix_blocks(
        matrix,
        block_size=block_size,
        title=f"Matrix Block Structure (Block Size: {block_size})",
        save_path=block_path,
        dpi=dpi,
    )

    # 3. 小さい行列の場合は値も可視化
    if max(m, n) <= 2000:
        values_path = os.path.join(output_dir, f"{prefix}_values_{matrix_size_str}.png")
        visualize_matrix_values(
            matrix, max_size=1000, title="Matrix Values", save_path=values_path, dpi=dpi
        )

    # 4. 行列の統計情報
    stats_path = os.path.join(output_dir, f"{prefix}_stats_{matrix_size_str}.png")
    visualize_matrix_properties(
        matrix, title="Matrix Properties", save_path=stats_path, dpi=dpi
    )

    print(f"\n行列分析が完了しました。結果は {output_dir} に保存されています。")


if __name__ == "__main__":
    # テスト用の行列を生成
    size = 100
    density = 0.05

    # ブロック対角行列の生成
    random_matrix = sp.random(size, size, density=density, format="csr")
    block_matrix = sp.block_diag([random_matrix, random_matrix])

    # 総合分析を実行
    analyze_and_visualize_matrix(block_matrix, prefix="test_matrix")
