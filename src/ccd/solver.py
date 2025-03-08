# solver.py
import cupy as cp
from typing import Tuple
from equation_system import EquationSystem
from grid import Grid


class CCDSolver:
    """CCDソルバークラス"""

    def __init__(self, system: EquationSystem, grid: Grid):
        """
        初期化

        Args:
            system: 方程式システム
            grid: 計算格子
        """
        self.system = system
        self.grid = grid

    def solve(self) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        方程式を解く
        
        Returns:
            (psi, psi', psi'', psi''')の組
        """
        # 行列システムを構築
        A, b = self.system.build_matrix_system()
        
        # システムを解く
        sol = cp.linalg.solve(A, b)
        
        # 解から各成分を抽出
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]
        
        return psi, psi_prime, psi_second, psi_third