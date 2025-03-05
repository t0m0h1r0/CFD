"""
クロネッカー積ユーティリティモジュール

2次元CCDのための効率的なクロネッカー積演算関数を提供します。
"""

import cupy as cp
import numpy as np
import cupyx.scipy.sparse as cpx_sparse
from typing import Union, Tuple, Optional


def kron(A: Union[cp.ndarray, cpx_sparse.spmatrix], 
         B: Union[cp.ndarray, cpx_sparse.spmatrix]) -> Union[cp.ndarray, cpx_sparse.spmatrix]:
    """
    2つの行列のクロネッカー積を計算（CuPy/CuPy sparse対応）
    
    スパース行列同士、または密行列同士のクロネッカー積を計算します。
    入力が混在する場合は、適切に変換して計算します。
    
    Args:
        A: 左側の行列
        B: 右側の行列
        
    Returns:
        計算されたクロネッカー積
    """
    # どちらもスパース行列の場合
    if cpx_sparse.issparse(A) and cpx_sparse.issparse(B):
        return cpx_sparse.kron(A, B, format='csr')
    
    # どちらも密行列の場合
    elif isinstance(A, cp.ndarray) and isinstance(B, cp.ndarray):
        return cp.kron(A, B)
    
    # 混合している場合 - スパース優先で変換
    elif cpx_sparse.issparse(A):
        B_sparse = cpx_sparse.csr_matrix(B)
        return cpx_sparse.kron(A, B_sparse, format='csr')
    
    elif cpx_sparse.issparse(B):
        A_sparse = cpx_sparse.csr_matrix(A)
        return cpx_sparse.kron(A_sparse, B, format='csr')
    
    else:
        raise TypeError("サポートされていない型の行列です。")


def identity_matrix(size: int, sparse: bool = True) -> Union[cp.ndarray, cpx_sparse.spmatrix]:
    """
    指定したサイズの単位行列を生成（CuPy/CuPy sparse対応）
    
    Args:
        size: 行列のサイズ
        sparse: スパース行列として生成するかどうか
        
    Returns:
        単位行列
    """
    if sparse:
        return cpx_sparse.eye(size, format='csr')
    else:
        return cp.eye(size)


def block_diag(matrices: list, format: str = 'csr') -> cpx_sparse.spmatrix:
    """
    与えられた行列からブロック対角行列を生成（CuPy sparse対応）
    
    Args:
        matrices: 対角ブロックに配置する行列のリスト
        format: 出力するスパース行列のフォーマット
        
    Returns:
        ブロック対角行列
    """
    return cpx_sparse.block_diag(matrices, format=format)


def kronecker_2d_operator(Dx: Union[cp.ndarray, cpx_sparse.spmatrix], 
                         Dy: Union[cp.ndarray, cpx_sparse.spmatrix], 
                         nx: int, 
                         ny: int) -> Tuple[cpx_sparse.spmatrix, cpx_sparse.spmatrix]:
    """
    2次元微分演算子を生成
    
    Args:
        Dx: x方向の1次元差分演算子
        Dy: y方向の1次元差分演算子
        nx: x方向のグリッド点数
        ny: y方向のグリッド点数
        
    Returns:
        (Dx_2D, Dy_2D): 2次元の微分演算子のタプル
    """
    # 単位行列
    Ix = identity_matrix(nx)
    Iy = identity_matrix(ny)
    
    # 2次元演算子
    Dx_2D = kron(Iy, Dx)  # x方向の微分
    Dy_2D = kron(Dy, Ix)  # y方向の微分
    
    return Dx_2D, Dy_2D


def kronecker_mixed_operator(Dx: Union[cp.ndarray, cpx_sparse.spmatrix], 
                            Dy: Union[cp.ndarray, cpx_sparse.spmatrix], 
                            order_x: int = 1, 
                            order_y: int = 1) -> cpx_sparse.spmatrix:
    """
    混合微分演算子を生成（例：∂²/∂x∂y）
    
    Args:
        Dx: x方向の1階微分演算子
        Dy: y方向の1階微分演算子
        order_x: x方向の微分階数
        order_y: y方向の微分階数
        
    Returns:
        混合微分演算子
    """
    # Dxをorder_x回適用した行列を計算
    if order_x > 1:
        Dx_power = Dx
        for _ in range(1, order_x):
            Dx_power = Dx_power @ Dx
    else:
        Dx_power = Dx
    
    # Dyをorder_y回適用した行列を計算
    if order_y > 1:
        Dy_power = Dy
        for _ in range(1, order_y):
            Dy_power = Dy_power @ Dy
    else:
        Dy_power = Dy
    
    # 混合演算子を生成（クロネッカー積）
    mixed_operator = kron(Dy_power, Dx_power)
    
    return mixed_operator


def apply_boundary_conditions(matrix: cpx_sparse.spmatrix,
                              nx: int,
                              ny: int,
                              dirichlet_x: bool = False,
                              dirichlet_y: bool = False,
                              neumann_x: bool = False,
                              neumann_y: bool = False) -> cpx_sparse.spmatrix:
    """
    行列に境界条件を適用
    
    Args:
        matrix: 修正する行列
        nx: x方向のグリッド点数
        ny: y方向のグリッド点数
        dirichlet_x: x方向ディリクレ境界条件を適用するか
        dirichlet_y: y方向ディリクレ境界条件を適用するか
        neumann_x: x方向ノイマン境界条件を適用するか
        neumann_y: y方向ノイマン境界条件を適用するか
        
    Returns:
        境界条件を適用した行列
    """
    # 行列を変更可能な形式に変換
    if not isinstance(matrix, cpx_sparse.csr_matrix):
        matrix = cpx_sparse.csr_matrix(matrix)
    
    # COO形式に変換
    matrix_coo = matrix.tocoo()
    
    # 行、列、データの配列をNumPy配列に変換（ハッシュ可能にするため）
    rows = cp.asnumpy(matrix_coo.row)
    cols = cp.asnumpy(matrix_coo.col)
    data = cp.asnumpy(matrix_coo.data)
    
    # 新しい行列データを構築するためのリスト
    new_rows = []
    new_cols = []
    new_data = []
    
    # 境界インデックスの計算とNumPy配列に変換
    x_left_indices = np.array([i * ny for i in range(nx)])  # 左端
    x_right_indices = np.array([(i + 1) * ny - 1 for i in range(nx)])  # 右端
    y_bottom_indices = np.array(list(range(ny)))  # 下端
    y_top_indices = np.array(list(range((nx - 1) * ny, nx * ny)))  # 上端
    
    # 境界インデックスをセットとして保存（整数に変換）
    boundary_indices = set()
    
    # x方向のディリクレ境界条件
    if dirichlet_x:
        boundary_indices.update(x_left_indices.tolist())
        boundary_indices.update(x_right_indices.tolist())
    
    # y方向のディリクレ境界条件
    if dirichlet_y:
        boundary_indices.update(y_bottom_indices.tolist())
        boundary_indices.update(y_top_indices.tolist())
    
    # 既存の要素を処理
    for i in range(len(rows)):
        row_idx = int(rows[i])  # 整数に変換
        col_idx = int(cols[i])  # 整数に変換
        val = float(data[i])    # 浮動小数点数に変換
        
        # 境界行かどうかをチェック
        if row_idx in boundary_indices:
            # 対角要素のみ1に設定、その他は無視
            if row_idx == col_idx:
                new_rows.append(row_idx)
                new_cols.append(col_idx)
                new_data.append(1.0)
        else:
            # 非境界要素はそのまま追加
            new_rows.append(row_idx)
            new_cols.append(col_idx)
            new_data.append(val)
    
    # 境界条件の行に対角成分が設定されたことを確認
    for idx in boundary_indices:
        # 対角成分が追加済みかチェック
        if idx not in [r for r, c in zip(new_rows, new_cols) if r == c == idx]:
            new_rows.append(idx)
            new_cols.append(idx)
            new_data.append(1.0)
    
    # リストをNumPy配列に変換
    new_rows_np = np.array(new_rows, dtype=np.int64)
    new_cols_np = np.array(new_cols, dtype=np.int64)
    new_data_np = np.array(new_data, dtype=np.float64)
    
    # NumPy配列をCuPy配列に変換
    new_rows_cp = cp.array(new_rows_np)
    new_cols_cp = cp.array(new_cols_np)
    new_data_cp = cp.array(new_data_np)
    
    # 新しい行列を構築
    shape = matrix.shape
    matrix_new = cpx_sparse.coo_matrix((new_data_cp, (new_rows_cp, new_cols_cp)), shape=shape)
    
    # CSR形式に変換して返す
    return matrix_new.tocsr()