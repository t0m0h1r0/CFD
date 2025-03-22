"""
2次元高階微分のための方程式セット定義モジュール
"""

from core.base.base_equation_set import EquationSet
from equation.dim2.original import OriginalEquation2D
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
from equation.converter import Equation1Dto2DConverter


class DerivativeEquationSet2D(EquationSet):
    """2次元高階微分のための方程式セット"""

    def __init__(self):
        """初期化"""
        super().__init__()
        # 微分方程式セットでは境界条件は無効
        self.enable_dirichlet = False
        self.enable_neumann = False
    
    def setup_equations(self, system, grid, test_func=None):
        """
        2次元導関数計算用の方程式システムを設定
        
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
        
        # 元の関数を全ての領域に追加
        system.add_dominant_equation(OriginalEquation2D(grid=grid))
        
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
        system.add_equations('left', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # 右境界用の方程式
        system.add_equations('right', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # 下境界用の方程式
        system.add_equations('bottom', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 上境界用の方程式
        system.add_equations('top', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 左下角用の方程式
        system.add_equations('left_bottom', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 右上角用の方程式
        system.add_equations('right_top', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 左上角用の方程式
        system.add_equations('left_top', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 右下角用の方程式
        system.add_equations('right_bottom', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False