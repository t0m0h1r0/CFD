"""
1次元ポアソン方程式セット（ディリクレ境界条件のみ）の定義を行うモジュール
"""

from core.base.base_equation_set import EquationSet
from equation.dim1.poisson import PoissonEquation
from equation.dim1.boundary import DirichletBoundaryEquation
from equation.dim1.compact_internal import (
    Internal1stDerivativeEquation,
    Internal2ndDerivativeEquation,
    Internal3rdDerivativeEquation
)
from equation.dim1.compact_left_boundary import (
    LeftBoundary1stDerivativeEquation,
    LeftBoundary2ndDerivativeEquation,
    LeftBoundary3rdDerivativeEquation
)
from equation.dim1.compact_right_boundary import (
    RightBoundary1stDerivativeEquation,
    RightBoundary2ndDerivativeEquation,
    RightBoundary3rdDerivativeEquation
)


class PoissonEquationSet1D2(EquationSet):
    """ディリクレ境界条件のみの1Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = False  # ノイマン境界条件を無効化
    
    def setup_equations(self, system, grid, test_func=None):
        """ディリクレ境界条件のみの1Dポアソン方程式セットをセットアップ"""
        if grid.is_2d:
            raise ValueError("1D方程式セットが2Dグリッドで使用されました")
        
        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation(grid=grid))
        
        # 内部点の方程式設定
        system.add_equation('interior', Internal1stDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal2ndDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal3rdDerivativeEquation(grid=grid))
        
        # 左境界の方程式設定
        system.add_equation('left', DirichletBoundaryEquation(grid=grid))
        system.add_equation('left', LeftBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('left', 
                          LeftBoundary1stDerivativeEquation(grid=grid)+
                          LeftBoundary3rdDerivativeEquation(grid=grid))
        
        # 右境界の方程式設定
        system.add_equation('right', DirichletBoundaryEquation(grid=grid))
        system.add_equation('right', RightBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('right', 
                          RightBoundary1stDerivativeEquation(grid=grid)+
                          RightBoundary3rdDerivativeEquation(grid=grid))
        
        return self.enable_dirichlet, self.enable_neumann