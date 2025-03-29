"""
2次元ポアソン方程式セットの定義を行うモジュール
"""

from core.base.base_equation_set import EquationSet
from equation.dim2.poisson import PoissonEquation2D
from equation.dim2.boundary import DirichletBoundaryEquation2D
from equation.dim1.boundary import NeumannBoundaryEquation
from equation.dim1.compact_internal import (
    Internal1stDerivativeEquation,
    Internal2ndDerivativeEquation,
    Internal3rdDerivativeEquation
)
from equation.dim1.compact_left_boundary import (
    LeftBoundary1stDerivativeEquation,
    LeftBoundary2ndDerivativeEquation
)
from equation.dim1.compact_right_boundary import (
    RightBoundary1stDerivativeEquation,
    RightBoundary2ndDerivativeEquation
)
from equation.converter import Equation1Dto2DConverter


class PoissonEquationSet2D(EquationSet):
    """ディリクレ・ノイマン混合境界条件の2Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = True
    
    def setup_equations(self, system, grid, test_func=None):
        """
        2次元ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (2D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if not grid.is_2d:
            raise ValueError("2D方程式セットが1Dグリッドで使用されました")
            
        # 変換器を作成
        converter = Equation1Dto2DConverter

        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation2D(grid=grid))
        
        # 内部点用の方程式
        system.add_equations('interior', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # 左境界用の方程式
        x_left_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ]
            
        system.add_equations('left', x_left_eqs)

        # 右境界用の方程式
        x_right_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('right', x_right_eqs)
        
        # 下境界用の方程式
        y_bottom_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
        
        system.add_equations('bottom', y_bottom_eqs)
            
        # 上境界用の方程式
        y_top_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
        
        system.add_equations('top', y_top_eqs)
        
        # 左下角 (i=0, j=0)
        left_bottom_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
                    
        system.add_equations('left_bottom', left_bottom_eqs)
        
        # 右上角 (i=nx-1, j=ny-1)
        right_top_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
            
        system.add_equations('right_top', right_top_eqs)
        
        # 左上角 (i=0, j=ny-1)
        left_top_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
            
        system.add_equations('left_top', left_top_eqs)
        
        # 右下角 (i=nx-1, j=0)
        right_bottom_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
                    
        system.add_equations('right_bottom', right_bottom_eqs)
        
        return True, True