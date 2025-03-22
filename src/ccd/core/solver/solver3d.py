"""
高精度コンパクト差分法 (CCD) を用いた3次元偏微分方程式ソルバーモジュール

このモジュールは、3次元でのポアソン方程式および高階微分方程式を
解くためのソルバークラスを提供します。
"""

import numpy as np

from core.base.base_solver import BaseCCDSolver
from core.equation_system.equation_system3d import EquationSystem3D
from core.rhs_builder.rhs_builder3d import RHSBuilder3D


class CCDSolver3D(BaseCCDSolver):
    """3次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid, backend="cpu"):
        """
        3Dソルバーを初期化
        
        Args:
            equation_set: 方程式セット
            grid: 3D グリッドオブジェクト
            backend: 計算バックエンド
        """
        if not hasattr(grid, 'is_3d') or not grid.is_3d:
            raise ValueError("3Dソルバーは非3Dグリッドでは使用できません")
            
        super().__init__(equation_set, grid, backend)
    
    def _create_equation_system(self, grid):
        """
        3次元方程式システムを作成
        
        Args:
            grid: 3Dグリッドオブジェクト
            
        Returns:
            EquationSystem3D: 3次元方程式システム
        """
        return EquationSystem3D(grid)
    
    def _create_rhs_builder(self):
        """3次元RHSビルダーを作成"""
        self.rhs_builder = RHSBuilder3D(
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
            Tuple[np.ndarray, ...]: 以下の10個の成分を含むタプル
                psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy, psi_z, psi_zz, psi_zzz
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        n_unknowns = 10  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz
        
        # 解配列を初期化 (NumPy配列)
        psi = np.zeros((nx, ny, nz))
        psi_x = np.zeros((nx, ny, nz))
        psi_xx = np.zeros((nx, ny, nz))
        psi_xxx = np.zeros((nx, ny, nz))
        psi_y = np.zeros((nx, ny, nz))
        psi_yy = np.zeros((nx, ny, nz))
        psi_yyy = np.zeros((nx, ny, nz))
        psi_z = np.zeros((nx, ny, nz))
        psi_zz = np.zeros((nx, ny, nz))
        psi_zzz = np.zeros((nx, ny, nz))
        
        # 3D形式の解ベクトルからデータを抽出
        # 各格子点 (i,j,k) のすべての未知数が連続して格納されている
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = (k * ny * nx + j * nx + i) * n_unknowns
                    psi[i, j, k] = sol[idx]
                    psi_x[i, j, k] = sol[idx + 1]
                    psi_xx[i, j, k] = sol[idx + 2]
                    psi_xxx[i, j, k] = sol[idx + 3]
                    psi_y[i, j, k] = sol[idx + 4]
                    psi_yy[i, j, k] = sol[idx + 5]
                    psi_yyy[i, j, k] = sol[idx + 6]
                    psi_z[i, j, k] = sol[idx + 7]
                    psi_zz[i, j, k] = sol[idx + 8]
                    psi_zzz[i, j, k] = sol[idx + 9]
        
        return psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy, psi_z, psi_zz, psi_zzz
