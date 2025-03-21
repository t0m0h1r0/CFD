"""
高精度コンパクト差分法 (CCD) を用いた2次元偏微分方程式ソルバーモジュール

このモジュールは、2次元でのポアソン方程式および高階微分方程式を
解くためのソルバークラスを提供します。
"""

import numpy as np

from base_solver import BaseCCDSolver
from equation_system2d import EquationSystem2D
from rhs_builder2d import RHSBuilder2D


class CCDSolver2D(BaseCCDSolver):
    """2次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid, backend="cpu"):
        """
        2Dソルバーを初期化
        
        Args:
            equation_set: 方程式セット
            grid: 2D グリッドオブジェクト
            backend: 計算バックエンド
        """
        if not grid.is_2d:
            raise ValueError("2Dソルバーは1Dグリッドでは使用できません")
            
        super().__init__(equation_set, grid, backend)
    
    def _create_equation_system(self, grid):
        """
        2次元方程式システムを作成
        
        Args:
            grid: 2Dグリッドオブジェクト
            
        Returns:
            EquationSystem2D: 2次元方程式システム
        """
        return EquationSystem2D(grid)
    
    def _create_rhs_builder(self):
        """2次元RHSビルダーを作成"""
        self.rhs_builder = RHSBuilder2D(
            self.system, 
            self.grid,
            enable_dirichlet=self.enable_dirichlet, 
            enable_neumann=self.enable_neumann
        )

    def _extract_solution(self, sol):
        """
        解ベクトルから各成分を抽出
        
        Args:
            sol: 線形ソルバーからの解ベクトル
            
        Returns:
            Tuple[np.ndarray, ...]: 以下の7つの成分を含むタプル
                psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # 解配列を初期化 (NumPy配列)
        psi = np.zeros((nx, ny))
        psi_x = np.zeros((nx, ny))
        psi_xx = np.zeros((nx, ny))
        psi_xxx = np.zeros((nx, ny))
        psi_y = np.zeros((nx, ny))
        psi_yy = np.zeros((nx, ny))
        psi_yyy = np.zeros((nx, ny))
        
        # 2D形式の解ベクトルからデータを抽出
        # 各格子点 (i,j) のすべての未知数が連続して格納されている
        for j in range(ny):
            for i in range(nx):
                idx = (j * nx + i) * n_unknowns
                psi[i, j] = sol[idx]
                psi_x[i, j] = sol[idx + 1]
                psi_xx[i, j] = sol[idx + 2]
                psi_xxx[i, j] = sol[idx + 3]
                psi_y[i, j] = sol[idx + 4]
                psi_yy[i, j] = sol[idx + 5]
                psi_yyy[i, j] = sol[idx + 6]
        
        return psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy