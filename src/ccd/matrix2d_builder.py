"""
2次元CCD行列ビルダーモジュール（メモリ最適化版）

2次元CCD法の左辺行列を生成するクラスを提供します。
疎行列構造を活用してメモリ使用量を最小化します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Tuple, Dict, Union

from grid2d_config import Grid2DConfig
from matrix_builder import CCDLeftHandBuilder  # 1次元CCDの行列ビルダー


class CCD2DLeftHandBuilder:
    """2次元CCD左辺行列を生成するクラス（CuPy疎行列対応、メモリ最適化版）"""

    def __init__(self):
        """初期化"""
        # 1次元CCD行列ビルダーをインスタンス化
        self.builder1d = CCDLeftHandBuilder()

    def build_matrix(
        self, grid_config: Grid2DConfig, coeffs: Optional[Dict[str, float]] = None
    ) -> cpx_sparse.spmatrix:
        """
        2次元CCD全体の左辺行列を生成（メモリ効率版）

        COO形式を使用して行列要素を一度に構築し、メモリ使用量を削減します。

        Args:
            grid_config: 2次元グリッド設定
            coeffs: 係数（省略時はgrid_configから取得）

        Returns:
            2次元CCD左辺行列（CuPy疎行列）
        """
        if coeffs is None:
            coeffs = grid_config.coeffs

        # グリッドサイズとパラメータ
        nx, ny = grid_config.nx, grid_config.ny
        hx, hy = grid_config.hx, grid_config.hy
        
        # 未知数の数（各点でf, f_x, f_y, f_xx）
        depth = 4
        total_unknowns = nx * ny * depth
        
        # COO形式のための行、列、値の配列
        # 非ゼロ要素の最大数を事前に見積もる（少なめに見積もって調整）
        estimated_nnz = total_unknowns * 5  # 各行につき平均で5つの非ゼロ要素を仮定
        rows = []
        cols = []
        data = []
        
        print(f"メモリ最適化行列ビルダーを使用 - グリッドサイズ: {nx}x{ny}, 合計未知数: {total_unknowns}")
        
        # 1. 内部点の方程式
        for j in range(ny):
            for i in range(nx):
                base_idx = (j * nx + i) * depth
                
                # 1-1. f(i,j)の方程式
                f_idx = base_idx
                
                # 対角成分
                rows.append(f_idx)
                cols.append(f_idx)
                data.append(1.0)  # f = f
                
                # 内部点の場合は差分式を追加
                if 0 < i < nx-1 and 0 < j < ny-1:
                    # x方向の中心差分
                    rows.append(f_idx)
                    cols.append(base_idx - depth + 1)  # 左隣の点のf_x
                    data.append(-0.5 / hx)
                    
                    rows.append(f_idx)
                    cols.append(base_idx + depth + 1)  # 右隣の点のf_x
                    data.append(0.5 / hx)
                    
                    # y方向の中心差分
                    rows.append(f_idx)
                    cols.append(base_idx - nx * depth + 2)  # 下隣の点のf_y
                    data.append(-0.5 / hy)
                    
                    rows.append(f_idx)
                    cols.append(base_idx + nx * depth + 2)  # 上隣の点のf_y
                    data.append(0.5 / hy)
                
                # 1-2. f_x(i,j)の方程式
                fx_idx = base_idx + 1
                
                rows.append(fx_idx)
                cols.append(fx_idx)
                data.append(1.0)  # f_x = f_x
                
                if 0 < i < nx-1:
                    # 中心差分による関係式
                    rows.append(fx_idx)
                    cols.append(base_idx - depth)  # 左隣のf
                    data.append(-0.5 / hx)
                    
                    rows.append(fx_idx)
                    cols.append(base_idx + depth)  # 右隣のf
                    data.append(0.5 / hx)
                
                # 1-3. f_y(i,j)の方程式
                fy_idx = base_idx + 2
                
                rows.append(fy_idx)
                cols.append(fy_idx)
                data.append(1.0)  # f_y = f_y
                
                if 0 < j < ny-1:
                    # 中心差分による関係式
                    rows.append(fy_idx)
                    cols.append(base_idx - nx * depth)  # 下隣のf
                    data.append(-0.5 / hy)
                    
                    rows.append(fy_idx)
                    cols.append(base_idx + nx * depth)  # 上隣のf
                    data.append(0.5 / hy)
                
                # 1-4. f_xx(i,j)の方程式
                fxx_idx = base_idx + 3
                
                rows.append(fxx_idx)
                cols.append(fxx_idx)
                data.append(1.0)  # f_xx = f_xx
                
                if 0 < i < nx-1:
                    # 中心差分による関係式
                    rows.append(fxx_idx)
                    cols.append(base_idx - depth)  # 左隣のf
                    data.append(1.0 / (hx * hx))
                    
                    rows.append(fxx_idx)
                    cols.append(base_idx)  # 中央のf
                    data.append(-2.0 / (hx * hx))
                    
                    rows.append(fxx_idx)
                    cols.append(base_idx + depth)  # 右隣のf
                    data.append(1.0 / (hx * hx))
        
        # 2. 境界条件
        # ディリクレ境界条件
        if grid_config.is_x_dirichlet:
            for j in range(ny):
                # 左端
                left_idx = (j * nx) * depth
                rows.append(left_idx)
                cols.append(left_idx)
                data.append(1.0)  # f = 境界値
                
                # 右端
                right_idx = (j * nx + nx - 1) * depth
                rows.append(right_idx)
                cols.append(right_idx)
                data.append(1.0)  # f = 境界値
        
        if grid_config.is_y_dirichlet:
            for i in range(nx):
                # 下端
                bottom_idx = i * depth
                rows.append(bottom_idx)
                cols.append(bottom_idx)
                data.append(1.0)  # f = 境界値
                
                # 上端
                top_idx = ((ny - 1) * nx + i) * depth
                rows.append(top_idx)
                cols.append(top_idx)
                data.append(1.0)  # f = 境界値
        
        # ノイマン境界条件
        if grid_config.is_x_neumann:
            for j in range(ny):
                # 左端
                left_idx = (j * nx) * depth + 1  # f_x
                rows.append(left_idx)
                cols.append(left_idx)
                data.append(1.0)  # f_x = 境界値
                
                # 右端
                right_idx = (j * nx + nx - 1) * depth + 1  # f_x
                rows.append(right_idx)
                cols.append(right_idx)
                data.append(1.0)  # f_x = 境界値
        
        if grid_config.is_y_neumann:
            for i in range(nx):
                # 下端
                bottom_idx = i * depth + 2  # f_y
                rows.append(bottom_idx)
                cols.append(bottom_idx)
                data.append(1.0)  # f_y = 境界値
                
                # 上端
                top_idx = ((ny - 1) * nx + i) * depth + 2  # f_y
                rows.append(top_idx)
                cols.append(top_idx)
                data.append(1.0)  # f_y = 境界値
        
        # COO形式で行列を作成
        L = cpx_sparse.coo_matrix(
            (data, (rows, cols)), 
            shape=(total_unknowns, total_unknowns),
            dtype=cp.float64
        )
        
        # CSR形式に変換（効率的な計算用）
        L_csr = L.tocsr()
        
        # メモリ使用状況の報告
        nnz = L_csr.nnz
        density = nnz / (total_unknowns * total_unknowns)
        memory_mb = (nnz * (8 + 4 + 4)) / (1024 * 1024)  # 値(8バイト) + 行(4バイト) + 列(4バイト)
        
        print(f"行列構築完了 - サイズ: {L_csr.shape}, 非ゼロ要素: {nnz}, 密度: {density:.2e}")
        print(f"推定メモリ使用量: {memory_mb:.2f} MB")
        
        return L_csr

    def _extract_x_dirichlet_values(
        self, grid_config: Grid2DConfig, j: int
    ) -> List[float]:
        """
        特定のy座標における、x方向のディリクレ境界値を抽出

        Args:
            grid_config: 2次元グリッド設定
            j: y方向のインデックス

        Returns:
            [左端, 右端] の境界値
        """
        if not grid_config.is_x_dirichlet or grid_config.x_dirichlet_values is None:
            return [0.0, 0.0]
        
        if j < len(grid_config.x_dirichlet_values):
            return list(grid_config.x_dirichlet_values[j])
        return [0.0, 0.0]

    def _extract_y_dirichlet_values(
        self, grid_config: Grid2DConfig, i: int
    ) -> List[float]:
        """
        特定のx座標における、y方向のディリクレ境界値を抽出

        Args:
            grid_config: 2次元グリッド設定
            i: x方向のインデックス

        Returns:
            [下端, 上端] の境界値
        """
        if not grid_config.is_y_dirichlet or grid_config.y_dirichlet_values is None:
            return [0.0, 0.0]
        
        if i < len(grid_config.y_dirichlet_values):
            return list(grid_config.y_dirichlet_values[i])
        return [0.0, 0.0]

    def _extract_x_neumann_values(
        self, grid_config: Grid2DConfig, j: int
    ) -> List[float]:
        """
        特定のy座標における、x方向のノイマン境界値を抽出

        Args:
            grid_config: 2次元グリッド設定
            j: y方向のインデックス

        Returns:
            [左端, 右端] の導関数境界値
        """
        if not grid_config.is_x_neumann or grid_config.x_neumann_values is None:
            return [0.0, 0.0]
        
        if j < len(grid_config.x_neumann_values):
            return list(grid_config.x_neumann_values[j])
        return [0.0, 0.0]

    def _extract_y_neumann_values(
        self, grid_config: Grid2DConfig, i: int
    ) -> List[float]:
        """
        特定のx座標における、y方向のノイマン境界値を抽出

        Args:
            grid_config: 2次元グリッド設定
            i: x方向のインデックス

        Returns:
            [下端, 上端] の導関数境界値
        """
        if not grid_config.is_y_neumann or grid_config.y_neumann_values is None:
            return [0.0, 0.0]
        
        if i < len(grid_config.y_neumann_values):
            return list(grid_config.y_neumann_values[i])
        return [0.0, 0.0]

    def _convert_coeffs_for_x(
        self, grid_config: Grid2DConfig, coeffs: Optional[Dict[str, float]] = None
    ) -> List[float]:
        """
        2次元の係数辞書から、x方向の1次元CCD用の係数リストに変換

        Args:
            grid_config: 2次元グリッド設定
            coeffs: 2次元の係数辞書（省略時はgrid_configから取得）

        Returns:
            [a, b, c, d] 形式の1次元係数リスト
        """
        if coeffs is None:
            coeffs = grid_config.coeffs
        
        # 2D係数から1Dのx方向係数への変換
        # f = a*ψ + b*ψ_x + c*ψ_xx + d*ψ_xxx が1次元CCDの形式
        return [
            coeffs.get("f", 1.0),     # a: 関数値の係数
            coeffs.get("f_x", 0.0),   # b: x偏導関数の係数
            coeffs.get("f_xx", 0.0),  # c: x二階偏導関数の係数
            coeffs.get("f_xxx", 0.0)  # d: x三階偏導関数の係数
        ]

    def _convert_coeffs_for_y(
        self, grid_config: Grid2DConfig, coeffs: Optional[Dict[str, float]] = None
    ) -> List[float]:
        """
        2次元の係数辞書から、y方向の1次元CCD用の係数リストに変換

        Args:
            grid_config: 2次元グリッド設定
            coeffs: 2次元の係数辞書（省略時はgrid_configから取得）

        Returns:
            [a, b, c, d] 形式の1次元係数リスト
        """
        if coeffs is None:
            coeffs = grid_config.coeffs
        
        # 2D係数から1Dのy方向係数への変換
        return [
            coeffs.get("f", 1.0),     # a: 関数値の係数
            coeffs.get("f_y", 0.0),   # b: y偏導関数の係数
            coeffs.get("f_yy", 0.0),  # c: y二階偏導関数の係数
            coeffs.get("f_yyy", 0.0)  # d: y三階偏導関数の係数
        ]