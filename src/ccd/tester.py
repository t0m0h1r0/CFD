# tester.py
import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple
from grid import Grid
from solver import CCDSolver
from equation_system import EquationSystem
from equation.essential import EssentialEquation
from equation.poisson import PoissonEquation
from equation.boundary import DirichletBoundaryEquation, NeumannBoundaryEquation
from equation.compact_internal import (
    Internal1stDerivativeEquation, 
    Internal2ndDerivativeEquation, 
    Internal3rdDerivativeEquation
)
from equation.compact_left_boundary import (
    LeftBoundary1stDerivativeEquation,
    LeftBoundary2ndDerivativeEquation,
    LeftBoundary3rdDerivativeEquation
)
from equation.compact_right_boundary import (
    RightBoundary1stDerivativeEquation,
    RightBoundary2ndDerivativeEquation,
    RightBoundary3rdDerivativeEquation
)
from ccd.test_functions import TestFunction, TestFunctionFactory

class CCDTester:
    """CCDメソッドのテストを行うクラス"""
    
    def __init__(self, grid: Grid):
        """
        初期化
        
        Args:
            grid: 計算格子
        """
        self.grid = grid
    
    def run_test_with_options(self, 
                             test_func: TestFunction, 
                             use_dirichlet: bool = True,
                             use_neumann: bool = True,
                             rehu_number: Optional[float] = None) -> Dict:
        """
        より柔軟なオプションでテストを実行
        
        Args:
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか（デフォルトはTrue）
            use_neumann: ノイマン境界条件を使用するかどうか（デフォルトはTrue）
            rehu_number: Reynolds-Hugoniot数（Noneの場合はスケーリングなし）
            
        Returns:
            テスト結果の辞書
        """
        # システムを構築
        system = EquationSystem(self.grid)
        
        # グリッド情報
        x_min = self.grid.x_min
        x_max = self.grid.x_max
        
        # 内部点の方程式を設定
        system.add_interior_equation(PoissonEquation(test_func.d2f))
        system.add_interior_equation(Internal1stDerivativeEquation())
        system.add_interior_equation(Internal2ndDerivativeEquation())
        system.add_interior_equation(Internal3rdDerivativeEquation())
        
        # 左境界の方程式を設定
        system.add_left_boundary_equation(PoissonEquation(test_func.d2f))
        system.add_left_boundary_equation(DirichletBoundaryEquation(test_func.f(x_min), is_left=True))
        system.add_left_boundary_equation(NeumannBoundaryEquation(test_func.df(x_min), is_left=True))
        system.add_left_boundary_equation(
            LeftBoundary1stDerivativeEquation()
            + LeftBoundary2ndDerivativeEquation()
            + LeftBoundary3rdDerivativeEquation()
        )

        # 右境界の方程式を設定
        system.add_right_boundary_equation(PoissonEquation(test_func.d2f))
        system.add_right_boundary_equation(DirichletBoundaryEquation(test_func.f(x_max), is_left=False))
        system.add_right_boundary_equation(NeumannBoundaryEquation(test_func.df(x_max), is_left=False))
        system.add_right_boundary_equation(
            RightBoundary1stDerivativeEquation()
            + RightBoundary2ndDerivativeEquation()
            + RightBoundary3rdDerivativeEquation()
        )
        
        # ソルバーを作成
        solver = CCDSolver(system, self.grid)
        
        # Rehuスケーリングの設定
        if rehu_number is not None:
            solver.set_rehu_scaling(
                rehu_number=rehu_number,
                characteristic_velocity=1.0,
                reference_length=1.0
            )
        
        # 解く
        psi, psi_prime, psi_second, psi_third = solver.solve()
        
        # 解析解を計算
        x = self.grid.get_points()
        exact_psi = cp.array([test_func.f(xi) for xi in x])
        exact_psi_prime = cp.array([test_func.df(xi) for xi in x])
        exact_psi_second = cp.array([test_func.d2f(xi) for xi in x])
        exact_psi_third = cp.array([test_func.d3f(xi) for xi in x])
        
        # 誤差を計算（CuPy配列のまま）
        err_psi = float(cp.max(cp.abs(psi - exact_psi)))
        err_psi_prime = float(cp.max(cp.abs(psi_prime - exact_psi_prime)))
        err_psi_second = float(cp.max(cp.abs(psi_second - exact_psi_second)))
        err_psi_third = float(cp.max(cp.abs(psi_third - exact_psi_third)))
        
        # 結果を返す
        return {
            "function": test_func.name,
            "numerical": [psi, psi_prime, psi_second, psi_third],
            "exact": [exact_psi, exact_psi_prime, exact_psi_second, exact_psi_third],
            "errors": [err_psi, err_psi_prime, err_psi_second, err_psi_third],
        }
    
    def run_grid_convergence_test(
        self, 
        test_func: TestFunction, 
        grid_sizes: List[int],
        x_range: Tuple[float, float],
        use_dirichlet: bool = True,
        use_neumann: bool = True,
        rehu_number: Optional[float] = None
    ) -> Dict[int, List[float]]:
        """
        グリッドサイズによる収束性テストを実行
        
        Args:
            test_func: テスト関数
            grid_sizes: グリッドサイズのリスト
            x_range: 計算範囲
            use_dirichlet: ディリクレ境界条件を使用するかどうか（デフォルトはTrue）
            use_neumann: ノイマン境界条件を使用するかどうか（デフォルトはTrue）
            rehu_number: Reynolds-Hugoniot数（Noneの場合はスケーリングなし）
            
        Returns:
            グリッドサイズごとの誤差 {grid_size: [err_psi, err_psi', err_psi'', err_psi''']}
        """
        results = {}
        
        for n in grid_sizes:
            # グリッドを作成
            grid = Grid(n, x_range)
            
            # このクラスのインスタンスを新しいグリッドで作成
            tester = CCDTester(grid)
            
            # テストを実行
            result = tester.run_test_with_options(
                test_func, 
                use_dirichlet=use_dirichlet, 
                use_neumann=use_neumann,
                rehu_number=rehu_number
            )
            
            # 結果を保存
            results[n] = result["errors"]
        
        return results