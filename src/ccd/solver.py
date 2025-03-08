# solver.py
import cupy as cp
from typing import Tuple, List, Optional, Dict
from equation_system import EquationSystem
from grid import Grid
from matrix_scaling import MatrixRehuScaling


class CCDSolver:
    """CCDソルバークラス (行列スケーリング機能付き)"""

    def __init__(self, system: EquationSystem, grid: Grid):
        """
        初期化

        Args:
            system: 方程式システム
            grid: 計算格子
        """
        self.system = system
        self.grid = grid
        self.scaler = None  # スケーリングオブジェクト
        
    def set_rehu_scaling(self, 
                         rehu_number: float = 1.0, 
                         characteristic_velocity: float = 1.0, 
                         reference_length: float = 1.0):
        """
        Reynolds-Hugoniotスケーリングを設定

        Args:
            rehu_number: Reynolds-Hugoniot数
            characteristic_velocity: 特性速度
            reference_length: 代表長さ
        """
        self.scaler = MatrixRehuScaling(
            characteristic_velocity=characteristic_velocity,
            reference_length=reference_length,
            rehu_number=rehu_number
        )
        print(f"Matrix-based Reynolds-Hugoniot scaling set with Rehu number: {rehu_number}")

    def solve(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        方程式を解く
        
        Returns:
            (psi, psi', psi'', psi''')の組
        """
        # 行列システムを構築
        A, b = self.system.build_matrix_system()
        
        # スケーリングが設定されていれば適用
        if self.scaler is not None:
            print("Applying matrix scaling...")
            A, b = self.scaler.scale_matrix_system(A, b, self.grid.n_points)
        
        # システムを解く
        print("Solving matrix system...")
        sol = cp.linalg.solve(A, b)
        
        # スケーリングが適用されていた場合、結果を元のスケールに戻す
        if self.scaler is not None:
            print("Unscaling solution...")
            sol = self.scaler.unscale_solution(sol, self.grid.n_points)
        
        # 解から各成分を抽出
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]
        
        return psi, psi_prime, psi_second, psi_third