"""
高精度コンパクト差分法 (CCD) を用いた1次元偏微分方程式ソルバーモジュール

このモジュールは、1次元でのポアソン方程式および高階微分方程式を
解くためのソルバークラスを提供します。
"""


from base_solver import BaseCCDSolver
from equation_system1d import EquationSystem1D
from rhs_builder1d import RHSBuilder1D


class CCDSolver1D(BaseCCDSolver):
    """1次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid, backend="cpu"):
        """
        1Dソルバーを初期化
        
        Args:
            equation_set: 方程式セット
            grid: 1D グリッドオブジェクト
            backend: 計算バックエンド
        """
        if grid.is_2d:
            raise ValueError("1Dソルバーは2Dグリッドでは使用できません")
            
        super().__init__(equation_set, grid, backend)
    
    def _create_equation_system(self, grid):
        """
        1次元方程式システムを作成
        
        Args:
            grid: 1Dグリッドオブジェクト
            
        Returns:
            EquationSystem1D: 1次元方程式システム
        """
        return EquationSystem1D(grid)
    
    def _create_rhs_builder(self):
        """1次元RHSビルダーを作成"""
        self.rhs_builder = RHSBuilder1D(
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
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                psi, psi_prime, psi_second, psi_third の各成分
        """
        n = self.grid.n_points
        
        # 1D形式の解ベクトルは [ψ₁, ψ'₁, ψ"₁, ψ'"₁, ψ₂, ψ'₂, ψ"₂, ψ'"₂, ...]
        # というように、各格子点のすべての変数が連続して格納されている
        psi = sol[0::4][:n]  # インデックス 0, 4, 8, ...
        psi_prime = sol[1::4][:n]  # インデックス 1, 5, 9, ...
        psi_second = sol[2::4][:n]  # インデックス 2, 6, 10, ...
        psi_third = sol[3::4][:n]  # インデックス 3, 7, 11, ...

        return psi, psi_prime, psi_second, psi_third