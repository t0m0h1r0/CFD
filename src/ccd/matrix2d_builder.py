"""
2次元行列ビルダーモジュール

2次元CCD法の左辺行列を生成するクラスを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Tuple, Dict, Any

from grid2d_config import Grid2DConfig
from utils.kronecker_utils import (
    kron, 
    identity_matrix, 
    kronecker_2d_operator, 
    kronecker_mixed_operator,
    apply_boundary_conditions
)


class CCD2DLeftHandBuilder:
    """2次元左辺ブロック行列を生成するクラス"""

    def _build_interior_blocks_1d(
        self, order: int, h: float, coeffs: Optional[Dict[str, float]] = None
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        内部点の1次元ブロック行列A, B, Cを生成
        
        Args:
            order: 微分演算子の最大階数
            h: グリッド幅
            coeffs: 係数辞書
            
        Returns:
            (A, B, C): 左、中央、右のブロック行列
        """
        # デフォルト係数: f = psi
        if coeffs is None:
            coeffs = {"a": 1.0}
        
        # CCD内部点での係数行列（1次元）
        # このマトリックスは1次元CCDの実装から取得されています
        
        # 行列サイズは最大階数+1（値、1階、2階、3階...）
        matrix_size = order + 1
        
        # 左ブロック行列 A
        A = cp.zeros((matrix_size, matrix_size))
        if order >= 3:  # 3階微分まで対応
            A[1, 0] = 35/32
            A[1, 1] = 19/32
            A[1, 2] = 1/8
            A[1, 3] = 1/96
            
            A[2, 0] = -4
            A[2, 1] = -29/16
            A[2, 2] = -5/16
            A[2, 3] = -1/48
            
            A[3, 0] = -105/16
            A[3, 1] = -105/16
            A[3, 2] = -15/8
            A[3, 3] = -3/16

        # 中央ブロック行列 B
        B = cp.zeros((matrix_size, matrix_size))
        B[0, 0] = coeffs.get("a", 1.0)  # 関数値の係数
        
        # 1階微分の係数
        if matrix_size > 1:
            B[0, 1] = coeffs.get("bx", 0.0)  # x方向1階微分の係数
            B[1, 1] = 1  # 1階微分方程式
        
        # 2階微分の係数
        if matrix_size > 2:
            B[0, 2] = coeffs.get("cxx", 0.0)  # x方向2階微分の係数
            B[2, 0] = 8
            B[2, 2] = 1  # 2階微分方程式
        
        # 3階微分の係数
        if matrix_size > 3:
            B[0, 3] = coeffs.get("dxxx", 0.0)  # x方向3階微分の係数
            B[3, 3] = 1  # 3階微分方程式

        # 右ブロック行列 C
        C = cp.zeros((matrix_size, matrix_size))
        if order >= 3:  # 3階微分まで対応
            C[1, 0] = -35/32
            C[1, 1] = 19/32
            C[1, 2] = -1/8
            C[1, 3] = 1/96
            
            C[2, 0] = -4
            C[2, 1] = 29/16
            C[2, 2] = -5/16
            C[2, 3] = 1/48
            
            C[3, 0] = 105/16
            C[3, 1] = -105/16
            C[3, 2] = 15/8
            C[3, 3] = -3/16

        # 次元の調整行列
        DEGREE = cp.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i == j:
                    DEGREE[i, j] = h**(i-j)
                elif i > j:
                    DEGREE[i, j] = h**(i-j)
                else:  # i < j
                    DEGREE[i, j] = h**(i-j)

        # 次元調整を適用
        A = A * DEGREE
        B = B * DEGREE
        C = C * DEGREE

        return A, B, C

    def _build_boundary_blocks_1d(
        self,
        order: int,
        h: float,
        coeffs: Optional[Dict[str, float]] = None,
        dirichlet_enabled: bool = True,
        neumann_enabled: bool = False,
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        境界点の1次元ブロック行列を生成
        
        Args:
            order: 微分演算子の最大階数
            h: グリッド幅
            coeffs: 係数辞書
            dirichlet_enabled: ディリクレ境界条件が有効か
            neumann_enabled: ノイマン境界条件が有効か
            
        Returns:
            (B0, C0, D0, ZR, AR, BR): 境界点のブロック行列
        """
        # デフォルト係数
        if coeffs is None:
            coeffs = {"a": 1.0}

        matrix_size = order + 1

        # 基本のブロック行列（1次元CCDから取得）
        B0 = cp.zeros((matrix_size, matrix_size))
        B0[0, 0] = coeffs.get("a", 1.0)  # 関数値の係数
        if matrix_size > 1:
            B0[0, 1] = coeffs.get("bx", 0.0)  # x方向1階微分の係数
        if matrix_size > 2:
            B0[0, 2] = coeffs.get("cxx", 0.0)  # x方向2階微分の係数
        if matrix_size > 3:
            B0[0, 3] = coeffs.get("dxxx", 0.0)  # x方向3階微分の係数
            
        if order >= 3:
            B0[1, 0] = 11/2
            B0[1, 1] = 1
            B0[2, 0] = -51/2
            B0[2, 2] = 1
            B0[3, 0] = 387/4
            B0[3, 3] = 1

        C0 = cp.zeros((matrix_size, matrix_size))
        if order >= 3:
            C0[1, 0] = 24
            C0[1, 1] = 24
            C0[1, 2] = 4
            C0[1, 3] = 4/3
            
            C0[2, 0] = -264
            C0[2, 1] = -216
            C0[2, 2] = -44
            C0[2, 3] = -12
            
            C0[3, 0] = 1644
            C0[3, 1] = 1236
            C0[3, 2] = 282
            C0[3, 3] = 66

        D0 = cp.zeros((matrix_size, matrix_size))
        if order >= 3:
            D0[1, 0] = -59/2
            D0[1, 1] = 10
            D0[1, 2] = -1
            
            D0[2, 0] = 579/2
            D0[2, 1] = -99
            D0[2, 2] = 10
            
            D0[3, 0] = -6963/4
            D0[3, 1] = 1203/2
            D0[3, 2] = -123/2

        ZR = cp.zeros((matrix_size, matrix_size))
        if order >= 3:
            ZR[1, 0] = 59/2
            ZR[1, 1] = 10
            ZR[1, 2] = 1
            
            ZR[2, 0] = 579/2
            ZR[2, 1] = 99
            ZR[2, 2] = 10
            
            ZR[3, 0] = 6963/4
            ZR[3, 1] = 1203/2
            ZR[3, 2] = 123/2

        AR = cp.zeros((matrix_size, matrix_size))
        if order >= 3:
            AR[1, 0] = -24
            AR[1, 1] = 24
            AR[1, 2] = -4
            AR[1, 3] = 4/3
            
            AR[2, 0] = -264
            AR[2, 1] = 216
            AR[2, 2] = -44
            AR[2, 3] = 12
            
            AR[3, 0] = -1644
            AR[3, 1] = 1236
            AR[3, 2] = -282
            AR[3, 3] = 66

        BR = cp.zeros((matrix_size, matrix_size))
        BR[0, 0] = coeffs.get("a", 1.0)  # 関数値の係数
        if matrix_size > 1:
            BR[0, 1] = coeffs.get("bx", 0.0)  # x方向1階微分の係数
        if matrix_size > 2:
            BR[0, 2] = coeffs.get("cxx", 0.0)  # x方向2階微分の係数
        if matrix_size > 3:
            BR[0, 3] = coeffs.get("dxxx", 0.0)  # x方向3階微分の係数
            
        if order >= 3:
            BR[1, 0] = -11/2
            BR[1, 1] = 1
            BR[2, 0] = -51/2
            BR[2, 2] = 1
            BR[3, 0] = -387/4
            BR[3, 3] = 1

        # 境界条件に応じて行を更新
        # ディリクレ境界条件
        if dirichlet_enabled:
            # 左端の第4行
            if order >= 3:
                B0_new = B0.copy()
                B0_new[3] = cp.zeros(matrix_size)
                B0_new[3, 0] = 1
                B0 = B0_new

                C0_new = C0.copy()
                C0_new[3] = cp.zeros(matrix_size)
                C0 = C0_new

                D0_new = D0.copy()
                D0_new[3] = cp.zeros(matrix_size)
                D0 = D0_new

                # 右端の第4行
                BR_new = BR.copy()
                BR_new[3] = cp.zeros(matrix_size)
                BR_new[3, 0] = 1
                BR = BR_new

                AR_new = AR.copy()
                AR_new[3] = cp.zeros(matrix_size)
                AR = AR_new

                ZR_new = ZR.copy()
                ZR_new[3] = cp.zeros(matrix_size)
                ZR = ZR_new

        # ノイマン境界条件
        if neumann_enabled and order >= 2:
            # 左端の第2行
            B0_new = B0.copy()
            B0_new[1] = cp.zeros(matrix_size)
            B0_new[1, 1] = 1
            B0 = B0_new

            C0_new = C0.copy()
            C0_new[1] = cp.zeros(matrix_size)
            C0 = C0_new

            D0_new = D0.copy()
            D0_new[1] = cp.zeros(matrix_size)
            D0 = D0_new

            # 右端の第2行
            BR_new = BR.copy()
            BR_new[1] = cp.zeros(matrix_size)
            BR_new[1, 1] = 1
            BR = BR_new

            AR_new = AR.copy()
            AR_new[1] = cp.zeros(matrix_size)
            AR = AR_new

            ZR_new = ZR.copy()
            ZR_new[1] = cp.zeros(matrix_size)
            ZR = ZR_new

        # 次数行列
        DEGREE = cp.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i == j:
                    DEGREE[i, j] = h**(i-j)
                elif i > j:
                    DEGREE[i, j] = h**(i-j)
                else:  # i < j
                    DEGREE[i, j] = h**(i-j)

        # 次元調整を適用
        B0 = B0 * DEGREE
        C0 = C0 * DEGREE
        D0 = D0 * DEGREE
        ZR = ZR * DEGREE
        AR = AR * DEGREE
        BR = BR * DEGREE

        return B0, C0, D0, ZR, AR, BR

    def _build_1d_system_matrix(
        self,
        n: int,
        h: float,
        order: int,
        coeffs: Optional[Dict[str, float]] = None,
        dirichlet_enabled: bool = True,
        neumann_enabled: bool = False,
    ) -> cp.ndarray:
        """
        1次元システム行列全体を構築
        
        Args:
            n: グリッド点の数
            h: グリッド幅
            order: 微分演算子の最大階数
            coeffs: 係数辞書
            dirichlet_enabled: ディリクレ境界条件が有効か
            neumann_enabled: ノイマン境界条件が有効か
            
        Returns:
            構築された1次元システム行列
        """
        # 内部点と境界点のブロック行列を生成
        A, B, C = self._build_interior_blocks_1d(order, h, coeffs)
        B0, C0, D0, ZR, AR, BR = self._build_boundary_blocks_1d(
            order, h, coeffs, dirichlet_enabled, neumann_enabled
        )

        # 行列サイズ
        matrix_size = order + 1
        system_size = matrix_size * n
        L = cp.zeros((system_size, system_size))

        # 左境界
        L[:matrix_size, :matrix_size] = B0
        L[:matrix_size, matrix_size:2*matrix_size] = C0
        if n > 2:
            L[:matrix_size, 2*matrix_size:3*matrix_size] = D0

        # 内部点
        for i in range(1, n-1):
            row_start = matrix_size * i
            L[row_start:row_start+matrix_size, row_start-matrix_size:row_start] = A
            L[row_start:row_start+matrix_size, row_start:row_start+matrix_size] = B
            L[row_start:row_start+matrix_size, row_start+matrix_size:row_start+2*matrix_size] = C

        # 右境界
        row_start = matrix_size * (n-1)
        if n > 2:
            L[row_start:row_start+matrix_size, row_start-2*matrix_size:row_start-matrix_size] = ZR
        L[row_start:row_start+matrix_size, row_start-matrix_size:row_start] = AR
        L[row_start:row_start+matrix_size, row_start:row_start+matrix_size] = BR

        return L

    def build_matrix(
        self,
        grid_config: Grid2DConfig,
        coeffs: Optional[Dict[str, float]] = None,
        dirichlet_enabled_x: bool = None,
        dirichlet_enabled_y: bool = None,
        neumann_enabled_x: bool = None,
        neumann_enabled_y: bool = None,
    ) -> cpx_sparse.spmatrix:
        """
        2次元左辺のブロック行列全体を生成（CuPy対応）
        
        Args:
            grid_config: 2次元グリッド設定
            coeffs: 係数辞書
            dirichlet_enabled_x: x方向ディリクレ境界条件の有効/無効
            dirichlet_enabled_y: y方向ディリクレ境界条件の有効/無効
            neumann_enabled_x: x方向ノイマン境界条件の有効/無効
            neumann_enabled_y: y方向ノイマン境界条件の有効/無効
            
        Returns:
            2次元システム行列（スパース形式）
        """
        nx, ny = grid_config.nx, grid_config.ny
        hx, hy = grid_config.hx, grid_config.hy
        
        # coeffsが指定されていない場合はgrid_configから取得
        if coeffs is None:
            coeffs = grid_config.coeffs
        
        # 境界条件の状態を決定
        if dirichlet_enabled_x is None:
            dirichlet_enabled_x = grid_config.is_dirichlet_x
        
        if dirichlet_enabled_y is None:
            dirichlet_enabled_y = grid_config.is_dirichlet_y
            
        if neumann_enabled_x is None:
            neumann_enabled_x = grid_config.is_neumann_x
            
        if neumann_enabled_y is None:
            neumann_enabled_y = grid_config.is_neumann_y
            
        # x方向とy方向の最大微分階数
        x_order = grid_config.x_deriv_order
        y_order = grid_config.y_deriv_order
        
        # 1次元システム行列の構築
        Lx = self._build_1d_system_matrix(
            nx, hx, x_order, coeffs, dirichlet_enabled_x, neumann_enabled_x
        )
        
        Ly = self._build_1d_system_matrix(
            ny, hy, y_order, coeffs, dirichlet_enabled_y, neumann_enabled_y
        )
        
        # スパース形式に変換
        Lx_sparse = cpx_sparse.csr_matrix(Lx)
        Ly_sparse = cpx_sparse.csr_matrix(Ly)
        
        # 単位行列
        Ix = identity_matrix(Lx_sparse.shape[0])
        Iy = identity_matrix(Ly_sparse.shape[0])
        
        # 2次元システム行列をクロネッカー積で構築
        # Lxy = kronecker(Iy, Lx) + kronecker(Ly, Ix)
        Lx_2D = kron(Iy, Lx_sparse)
        Ly_2D = kron(Ly_sparse, Ix)
        
        # 方程式の係数に基づいて行列を結合
        a_coef = coeffs.get("a", 1.0)
        bx_coef = coeffs.get("bx", 0.0)
        by_coef = coeffs.get("by", 0.0)
        cxx_coef = coeffs.get("cxx", 0.0)
        cyy_coef = coeffs.get("cyy", 0.0)
        cxy_coef = coeffs.get("cxy", 0.0)
        
        # 基本的な2次元システム行列
        L_2D = a_coef * Lx_2D
        
        # x方向の1階微分項を追加
        if bx_coef != 0:
            L_2D = L_2D + bx_coef * Lx_2D
        
        # y方向の1階微分項を追加
        if by_coef != 0:
            L_2D = L_2D + by_coef * Ly_2D
        
        # x方向の2階微分項を追加
        if cxx_coef != 0 and x_order >= 2:
            L_2D = L_2D + cxx_coef * Lx_2D
        
        # y方向の2階微分項を追加
        if cyy_coef != 0 and y_order >= 2:
            L_2D = L_2D + cyy_coef * Ly_2D
        
        # 混合微分項を追加 (∂²/∂x∂y)
        if cxy_coef != 0 and x_order >= 1 and y_order >= 1:
            # クロネッカー積を使用して混合微分演算子を構築
            # 実装はより複雑になる可能性があります
            pass
        
        # 境界条件を適用
        L_2D = apply_boundary_conditions(
            L_2D, 
            nx, 
            ny, 
            dirichlet_enabled_x, 
            dirichlet_enabled_y, 
            neumann_enabled_x, 
            neumann_enabled_y
        )
        
        return L_2D
