"""
2次元CCD行列ビルダーモジュール（修正版）

2次元CCD法の左辺行列を生成するクラスを提供します。
1次元CCDの行列を基に、クロネッカー積を用いて2次元の行列を構築します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Tuple, Dict, Union

from grid2d_config import Grid2DConfig
from matrix_builder import CCDLeftHandBuilder  # 1次元CCDの行列ビルダー


class CCD2DLeftHandBuilder:
    """2次元CCD左辺行列を生成するクラス（CuPy疎行列対応）"""

    def __init__(self):
        """初期化"""
        # 1次元CCD行列ビルダーをインスタンス化
        self.builder1d = CCDLeftHandBuilder()

    def build_x_matrix(
        self, grid_config: Grid2DConfig, coeffs: Optional[Dict[str, float]] = None
    ) -> cpx_sparse.spmatrix:
        """
        x方向の1次元CCD行列を生成

        Args:
            grid_config: 2次元グリッド設定
            coeffs: 係数（省略時はgrid_configから取得）

        Returns:
            x方向の1次元CCD行列（CuPy疎行列）
        """
        # 1次元グリッド設定を作成
        from grid_config import GridConfig
        
        grid1d_x = GridConfig(
            n_points=grid_config.nx,
            h=grid_config.hx,
            dirichlet_values=self._extract_x_dirichlet_values(grid_config, 0),  # y=0の行のディリクレ値
            neumann_values=self._extract_x_neumann_values(grid_config, 0),      # y=0の行のノイマン値
            coeffs=self._convert_coeffs_for_x(grid_config, coeffs)
        )
        
        # 1次元の行列を取得
        return self.builder1d.build_matrix(
            grid1d_x,
            coeffs=self._convert_coeffs_for_x(grid_config, coeffs),
            dirichlet_enabled=grid_config.is_x_dirichlet,
            neumann_enabled=grid_config.is_x_neumann
        )

    def build_y_matrix(
        self, grid_config: Grid2DConfig, coeffs: Optional[Dict[str, float]] = None
    ) -> cpx_sparse.spmatrix:
        """
        y方向の1次元CCD行列を生成

        Args:
            grid_config: 2次元グリッド設定
            coeffs: 係数（省略時はgrid_configから取得）

        Returns:
            y方向の1次元CCD行列（CuPy疎行列）
        """
        # 1次元グリッド設定を作成
        from grid_config import GridConfig
        
        grid1d_y = GridConfig(
            n_points=grid_config.ny,
            h=grid_config.hy,
            dirichlet_values=self._extract_y_dirichlet_values(grid_config, 0),  # x=0の列のディリクレ値
            neumann_values=self._extract_y_neumann_values(grid_config, 0),      # x=0の列のノイマン値
            coeffs=self._convert_coeffs_for_y(grid_config, coeffs)
        )
        
        # 1次元の行列を取得
        return self.builder1d.build_matrix(
            grid1d_y,
            coeffs=self._convert_coeffs_for_y(grid_config, coeffs),
            dirichlet_enabled=grid_config.is_y_dirichlet,
            neumann_enabled=grid_config.is_y_neumann
        )

    def build_matrix(
        self, grid_config: Grid2DConfig, coeffs: Optional[Dict[str, float]] = None
    ) -> cpx_sparse.spmatrix:
        """
        2次元CCD全体の左辺行列を生成（単純化版）
        """
        if coeffs is None:
            coeffs = grid_config.coeffs
        
        nx, ny = grid_config.nx, grid_config.ny
        total_unknowns = nx * ny * 4  # 4つの未知数（f, f_x, f_y, f_xx）
        
        # 行列の初期化（単位行列ベース）
        L = cpx_sparse.eye(total_unknowns, format='csr')
        
        # 基本的な連立方程式を設定
        # この例では単純な中心差分法を使用
        idx = 0
        for j in range(ny):
            for i in range(nx):
                base_idx = idx * 4
                
                # 関数値の方程式
                if 0 < i < nx-1 and 0 < j < ny-1:
                    # 内部点
                    L[base_idx, base_idx] = 1.0  # f(i,j)

                    # 一階導関数との関係
                    hx = grid_config.hx
                    hy = grid_config.hy
                    
                    # x方向の差分
                    if i > 0 and i < nx-1:
                        L[base_idx, base_idx-4+1] = -1/(2*hx)  # f_x(i-1,j)
                        L[base_idx, base_idx+4+1] = 1/(2*hx)   # f_x(i+1,j)
                    
                    # y方向の差分
                    if j > 0 and j < ny-1:
                        L[base_idx, base_idx-(nx*4)+2] = -1/(2*hy)  # f_y(i,j-1)
                        L[base_idx, base_idx+(nx*4)+2] = 1/(2*hy)   # f_y(i,j+1)
                
                # 次のポイントへ
                idx += 1
        
        return L

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