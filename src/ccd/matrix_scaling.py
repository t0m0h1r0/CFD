# matrix_scaling.py
import cupy as cp
import cupyx.scipy.sparse as sp
from typing import Tuple, Optional, Union


class MatrixRehuScaling:
    """行列システム全体に対してReynolds-Hugoniotスケーリングを適用するクラス (スパース行列最適化版)"""

    def __init__(
        self,
        characteristic_velocity: float = 1.0,
        reference_length: float = 1.0,
        rehu_number: float = 1.0,
    ):
        """
        初期化

        Args:
            characteristic_velocity: 特性速度（例：音速）
            reference_length: 代表長さ
            rehu_number: Reynolds-Hugoniot数
        """
        self.u_ref = characteristic_velocity
        self.L_ref = reference_length
        self.t_ref = self.L_ref / self.u_ref  # 特性時間
        self.rehu_number = rehu_number

    def scale_matrix_system(
        self, A: Union[cp.ndarray, sp.spmatrix], b: cp.ndarray, n_points: int
    ) -> Tuple[Union[cp.ndarray, sp.spmatrix], cp.ndarray]:
        """
        行列システム全体に対してスケーリングを適用 (スパースおよび密行列対応)

        Args:
            A: 元の係数行列（スパースまたは密）
            b: 元の右辺ベクトル
            n_points: グリッド点の数

        Returns:
            スケーリングされた行列システム (A_scaled, b_scaled)
        """
        # スケーリング行列の作成
        S = self._create_scaling_matrix(n_points)
        S_inv = cp.linalg.inv(S)  # 逆行列
        
        # 行列がスパースかどうか判定
        is_sparse = isinstance(A, sp.spmatrix)
        
        if is_sparse:
            # スパース行列用のスケーリング
            # 対角スケーリング行列を使用して効率的に計算
            S_diag = sp.dia_matrix((S.diagonal(), [0]), shape=S.shape)
            S_inv_diag = sp.dia_matrix((S_inv.diagonal(), [0]), shape=S_inv.shape)
            
            # A' = S⁻¹ A S の計算
            A_scaled = S_inv_diag @ A @ S_diag
        else:
            # 密行列用のスケーリング
            A_scaled = S_inv @ A @ S
        
        # 右辺ベクトルのスケーリング
        b_scaled = S_inv @ b

        return A_scaled, b_scaled

    def unscale_solution(self, solution: cp.ndarray, n_points: int) -> cp.ndarray:
        """
        スケーリングされた解を元のスケールに戻す

        Args:
            solution: スケーリングされた解ベクトル
            n_points: グリッド点の数

        Returns:
            元のスケールに戻された解ベクトル
        """
        # スケーリング行列の作成
        S = self._create_scaling_matrix(n_points)

        # 解のスケーリングを戻す
        # x = S x'
        unscaled_solution = S @ solution

        return unscaled_solution

    def _create_scaling_matrix(self, n_points: int) -> cp.ndarray:
        """
        スケーリング行列を作成

        Args:
            n_points: グリッド点の数

        Returns:
            スケーリング行列 S
        """
        # 行列のサイズ (各点に4つの未知数: ψ, ψ', ψ'', ψ''')
        n = 4 * n_points

        # 対角行列を初期化 (密行列で十分、後でスパース対角行列に変換可能)
        S = cp.eye(n)

        # Reynolds-Hugoniot則に基づくスケーリング係数を設定
        for i in range(n_points):
            # ψ項のスケーリング - そのまま
            S[4 * i, 4 * i] = 1.0

            # ψ'項のスケーリング - 1/L_ref
            # 移流項に対してはRehu数も考慮
            S[4 * i + 1, 4 * i + 1] = 1.0 / self.L_ref * self.rehu_number

            # ψ''項のスケーリング - 1/L_ref^2
            S[4 * i + 2, 4 * i + 2] = 1.0 / (self.L_ref**2)

            # ψ'''項のスケーリング - 1/L_ref^3
            S[4 * i + 3, 4 * i + 3] = 1.0 / (self.L_ref**3)

        return S