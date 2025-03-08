# matrix_scaling.py
import cupy as cp
import numpy as np
from typing import Tuple, Dict, Optional, List

class MatrixRehuScaling:
    """行列システム全体に対してReynolds-Hugoniotスケーリングを適用するクラス"""
    
    def __init__(self, 
                 characteristic_velocity: float = 1.0, 
                 reference_length: float = 1.0,
                 rehu_number: float = 1.0):
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
    
    def scale_matrix_system(self, A: cp.ndarray, b: cp.ndarray, n_points: int) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        行列システム全体に対してスケーリングを適用
        
        Args:
            A: 元の係数行列
            b: 元の右辺ベクトル
            n_points: グリッド点の数
            
        Returns:
            スケーリングされた行列システム (A_scaled, b_scaled)
        """
        # 行列のサイズを取得
        n_rows, n_cols = A.shape
        
        # スケーリング行列の作成
        S = self._create_scaling_matrix(n_points)
        
        # 行列システムのスケーリング
        # A' = S⁻¹ A S
        # b' = S⁻¹ b
        S_inv = cp.linalg.inv(S)  # あるいはS_invを直接計算することも可能
        
        A_scaled = S_inv @ A @ S
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
        
        # 対角行列を初期化
        S = cp.eye(n)
        
        # Reynolds-Hugoniot則に基づくスケーリング係数を設定
        for i in range(n_points):
            # ψ項のスケーリング - そのまま
            S[4*i, 4*i] = 1.0
            
            # ψ'項のスケーリング - 1/L_ref
            # 移流項に対してはRehu数も考慮
            S[4*i+1, 4*i+1] = 1.0 / self.L_ref * self.rehu_number
            
            # ψ''項のスケーリング - 1/L_ref^2
            S[4*i+2, 4*i+2] = 1.0 / (self.L_ref ** 2)
            
            # ψ'''項のスケーリング - 1/L_ref^3
            S[4*i+3, 4*i+3] = 1.0 / (self.L_ref ** 3)
        
        return S