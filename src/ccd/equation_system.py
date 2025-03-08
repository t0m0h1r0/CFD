# equation_system.py
import cupy as cp  # NumPyではなくCuPyを使用
from typing import List, Tuple, Dict
from equation.base import Equation
from grid import Grid

class EquationSystem:
    """方程式システムを管理するクラス"""
    
    def __init__(self, grid: Grid):
        """
        初期化
        
        Args:
            grid: 計算格子
        """
        self.grid = grid
        # 位置ごとに方程式をグループ化
        self.left_boundary_equations: List[Equation] = []
        self.interior_equations: List[Equation] = []
        self.right_boundary_equations: List[Equation] = []
            
    def add_left_boundary_equation(self, equation: Equation) -> None:
        """左境界用の方程式を追加"""
        self.left_boundary_equations.append(equation)
    
    def add_interior_equation(self, equation: Equation) -> None:
        """内部点用の方程式を追加"""
        self.interior_equations.append(equation)
    
    def add_right_boundary_equation(self, equation: Equation) -> None:
        """右境界用の方程式を追加"""
        self.right_boundary_equations.append(equation)
    
    def build_matrix_system(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """行列システムを構築"""
        n = self.grid.n_points
        h = self.grid.h
        print(h)
        
        # マトリックスサイズ: 各点に4つの値 (psi, psi', psi'', psi''') がある
        size = 4 * n
        A = cp.zeros((size, size))
        b = cp.zeros(size)
        
        # 各格子点での方程式を評価
        for i in range(n):
            equations_to_use = []
            
            if i == 0:  # 左境界
                equations_to_use = self.left_boundary_equations
            elif i == n - 1:  # 右境界
                equations_to_use = self.right_boundary_equations
            else:  # 内部点
                equations_to_use = self.interior_equations
            
            # この点での方程式がない場合はエラー
            if not equations_to_use:
                raise ValueError(f"点 {i} に対する方程式が設定されていません")
            
            # 4つの方程式（ψ, ψ', ψ'', ψ'''の各状態に対応）が必要
            if len(equations_to_use) != 4:
                raise ValueError(f"点 {i} に対する方程式が4つではありません（{len(equations_to_use)}個）")
            
            # 各方程式を行列に反映
            for k, eq in enumerate(equations_to_use):
                # 方程式のステンシル係数を取得
                stencil_coeffs = eq.get_stencil_coefficients(i, n, h)
                rhs_value = eq.get_rhs(i, n, h)
                
                # 各ステンシル点の係数を行列に設定
                for offset, coeffs in stencil_coeffs.items():
                    j = i + offset  # ステンシル点のインデックス
                    A[i*4 + k, j*4:(j+1)*4] = coeffs
                
                # 右辺ベクトルを設定
                b[i*4 + k] = rhs_value
        cp.set_printoptions(precision=2,suppress=True,linewidth=300)
        print(A)
        print(b.reshape((n,4)))
        
        return A, b