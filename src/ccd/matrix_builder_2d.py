"""
2次元行列ビルダーモジュール - 根本的バグ修正版

1次元CCDの行列ビルダーを利用して、2次元CCD用の行列を構築します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Optional, List, Tuple

from grid_config_2d import GridConfig2D
from matrix_builder import CCDLeftHandBuilder  # 1次元の行列ビルダー


class CCDLeftHandBuilder2D:
    """2次元CCD行列のビルダークラス"""

    def __init__(self):
        """初期化"""
        # 1次元ビルダーを内部で使用
        self.builder_1d = CCDLeftHandBuilder()

    def build_matrix(
        self,
        grid_config: GridConfig2D,
        coeffs: Optional[List[float]] = None,
        dirichlet_enabled: bool = None,
        neumann_enabled: bool = None,
    ) -> cpx_sparse.spmatrix:
        """
        クロネッカー積を用いて2次元CCD行列を構築

        Args:
            grid_config: 2次元グリッド設定
            coeffs: 係数（省略時はgrid_configから取得）
            dirichlet_enabled: ディリクレ境界条件の有効/無効（省略時はgrid_configから判断）
            neumann_enabled: ノイマン境界条件の有効/無効（省略時はgrid_configから判断）

        Returns:
            2次元CCD用の疎行列
        """
        # 係数の設定
        if coeffs is None:
            coeffs = grid_config.coeffs

        # 境界条件の状態を決定
        if dirichlet_enabled is None:
            dirichlet_enabled = grid_config.is_dirichlet()
        if neumann_enabled is None:
            neumann_enabled = grid_config.is_neumann()

        # x方向とy方向の1次元グリッド設定を取得
        grid_x = grid_config.get_grid_x()
        grid_y = grid_config.get_grid_y()

        # x方向とy方向の行列を構築
        Lx = self.builder_1d.build_matrix(
            grid_x, coeffs, dirichlet_enabled, neumann_enabled
        )
        Ly = self.builder_1d.build_matrix(
            grid_y, coeffs, dirichlet_enabled, neumann_enabled
        )

        # デバッグ情報の出力
        print(f"Lxのサイズ: {Lx.shape}, Lyのサイズ: {Ly.shape}")
        print(f"nx: {grid_config.nx_points}, ny: {grid_config.ny_points}")

        nx, ny = grid_config.nx_points, grid_config.ny_points
        depth = 4  # 各点での状態の次元
        
        # 正しいサイズの単位行列を作成
        Ix = cpx_sparse.eye(nx * depth, format='csr')
        Iy = cpx_sparse.eye(ny * depth, format='csr')
        
        # クロネッカー積を計算
        print(f"クロネッカー積を計算: {Ix.shape} ⊗ {Ly.shape} + {Lx.shape} ⊗ {Iy.shape}")
        
        # 正しいクロネッカー積の組み合わせで2D行列を構築
        try:
            # クロネッカー積を計算 (テンソル積)
            L2D = cpx_sparse.kron(Ix, Ly) + cpx_sparse.kron(Lx, Iy)
            
            # 実際のサイズとあるべきサイズをチェック
            actual_size = L2D.shape
            expected_size = (nx * ny * depth, nx * ny * depth)
            
            if actual_size != expected_size:
                print(f"警告: 2D行列のサイズ {actual_size} が期待値 {expected_size} と一致しません")
            
            print(f"2D行列のサイズ: {L2D.shape}, NNZ: {L2D.nnz}")
            return L2D
            
        except Exception as e:
            print(f"クロネッカー積の計算でエラー: {e}")
            raise