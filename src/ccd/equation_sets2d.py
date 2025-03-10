import cupy as cp
from abc import ABC, abstractmethod

from equation.equation_converter import Equation1Dto2DConverter
from equation.poisson import PoissonEquation2D
from equation.original import OriginalEquation2D
from equation.compact_internal import Internal1stDerivativeEquation, Internal2ndDerivativeEquation, Internal3rdDerivativeEquation
from equation.compact_left_boundary import LeftBoundary1stDerivativeEquation, LeftBoundary2ndDerivativeEquation, LeftBoundary3rdDerivativeEquation
from equation.compact_right_boundary import RightBoundary1stDerivativeEquation, RightBoundary2ndDerivativeEquation, RightBoundary3rdDerivativeEquation
from equation.base2d import Equation2D
from equation.boundary import DirichletXBoundaryEquation2D, DirichletYBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
from grid2d import Grid2D

class CustomEquation2D(Equation2D):
    """2D用のカスタム方程式"""
    
    def __init__(self, f_func, coeffs, grid=None):
        """
        初期化
        
        Args:
            f_func: 関数 f(x, y)
            coeffs: 係数リスト [c0, c1, c2, c3, c4, c5, c6]
                    各係数は [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy] に対応
            grid: 計算格子オブジェクト（オプション）
        """
        super().__init__(grid)
        self.f_func = f_func
        self.coeffs = cp.array(coeffs)
    
    def get_stencil_coefficients(self, grid=None, i=None, j=None):
        """ステンシル係数を取得"""
        return {(0, 0): self.coeffs}
    
    def get_rhs(self, grid=None, i=None, j=None):
        """右辺値を取得"""
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
            
        x, y = using_grid.get_point(i, j)
        return self.f_func(x, y)
    
    def is_valid_at(self, grid=None, i=None, j=None):
        """指定点で有効かどうかを判定"""
        return True


class EquationSet2D(ABC):
    """方程式セットの抽象基底クラス (2D版)"""

    def __init__(self):
        """初期化"""
        # サブクラス用の共通初期化処理があれば追加
        pass

    @abstractmethod
    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        """方程式システムに方程式を設定する"""
        pass

    @classmethod
    def get_available_sets(cls):
        """利用可能な方程式セットを返す"""
        return {
            "poisson": Poisson2DEquationSet,
            "derivative": Derivative2DEquationSet,
        }

    @classmethod
    def create(cls, name):
        """方程式セットを名前から作成"""
        available_sets = cls.get_available_sets()
        
        name = name.strip()
        
        if name in available_sets:
            return available_sets[name]()
        else:
            print(f"警告: 方程式セット '{name}' は利用できません。")
            print(f"利用可能なセット: {list(available_sets.keys())}")
            print("デフォルトの 'poisson' を使用します。")
            return available_sets["poisson"]()


class Poisson2DEquationSet(EquationSet2D):
    """2次元ポアソン方程式のための方程式セット"""

    def setup_equations(self, system, grid: Grid2D, test_func, use_dirichlet=True, use_neumann=True):
        """
        2次元ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: 計算格子
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
        """
        # 変換器を作成
        converter = Equation1Dto2DConverter
        
        # ポアソン方程式: Δψ = f(x,y) - グリッドを渡す
        def poisson_source(x, y):
            return (test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y))
        poisson_eq = PoissonEquation2D(poisson_source, grid)
        system.add_equation(poisson_eq)

        # 境界値の計算（x, y方向）
        left_values = cp.array([test_func.f(grid.x_min, y) for y in grid.y])
        right_values = cp.array([test_func.f(grid.x_max, y) for y in grid.y])
        bottom_values = cp.array([test_func.f(x, grid.y_min) for x in grid.x])
        top_values = cp.array([test_func.f(x, grid.y_max) for x in grid.x])            

        # 境界の導関数値の計算（x, y方向）
        d_left_values = cp.array([test_func.df_dx(grid.x_min, y) for y in grid.y])
        d_right_values = cp.array([test_func.df_dx(grid.x_max, y) for y in grid.y])
        d_bottom_values = cp.array([test_func.df_dy(x, grid.y_min) for x in grid.x])
        d_top_values = cp.array([test_func.df_dy(x, grid.y_max) for x in grid.x])            
        
        # 内部点の方程式 - 1次元方程式を各方向に拡張（gridも渡す）
        # X方向
        system.add_interior_x_equation(converter.to_x(Internal1stDerivativeEquation(), grid=grid))
        system.add_interior_x_equation(converter.to_x(Internal2ndDerivativeEquation(), grid=grid))
        system.add_interior_x_equation(converter.to_x(Internal3rdDerivativeEquation(), grid=grid))
        
        # Y方向
        system.add_interior_y_equation(converter.to_y(Internal1stDerivativeEquation(), grid=grid))
        system.add_interior_y_equation(converter.to_y(Internal2ndDerivativeEquation(), grid=grid))
        system.add_interior_y_equation(converter.to_y(Internal3rdDerivativeEquation(), grid=grid))
        
        # 境界点の方程式 - gridを渡す
        # 左境界 (i=0)
        system.add_left_boundary_equation(DirichletXBoundaryEquation2D(value=left_values, grid=grid))
        system.add_left_boundary_equation(NeumannXBoundaryEquation2D(value=d_left_values, grid=grid))
        # 左境界の補助方程式（複合）
        left_combined = (
            LeftBoundary1stDerivativeEquation() +
            LeftBoundary2ndDerivativeEquation() +
            LeftBoundary3rdDerivativeEquation()
        )
        system.add_left_boundary_equation(converter.to_x(left_combined, grid=grid))
                
        # 右境界 (i=nx-1)
        system.add_right_boundary_equation(DirichletXBoundaryEquation2D(value=right_values, grid=grid))
        system.add_right_boundary_equation(NeumannXBoundaryEquation2D(value=d_right_values, grid=grid))
        # 右境界の補助方程式（複合）
        right_combined = (
            RightBoundary1stDerivativeEquation() +
            RightBoundary2ndDerivativeEquation() +
            RightBoundary3rdDerivativeEquation()
        )
        system.add_right_boundary_equation(converter.to_x(right_combined, grid=grid))
                
        # 下境界 (j=0)
        system.add_bottom_boundary_equation(DirichletYBoundaryEquation2D(value=bottom_values, grid=grid))
        system.add_bottom_boundary_equation(NeumannYBoundaryEquation2D(value=d_bottom_values, grid=grid))
        # 下境界の補助方程式（複合）
        bottom_combined = (
            LeftBoundary1stDerivativeEquation() +
            LeftBoundary2ndDerivativeEquation() +
            LeftBoundary3rdDerivativeEquation()
        )
        system.add_bottom_boundary_equation(converter.to_y(bottom_combined, grid=grid))
                
        # 上境界 (j=ny-1)
        system.add_top_boundary_equation(DirichletYBoundaryEquation2D(value=top_values, grid=grid))
        system.add_top_boundary_equation(NeumannYBoundaryEquation2D(value=d_top_values, grid=grid))
        # 上境界の補助方程式（複合）
        top_combined = (
            RightBoundary1stDerivativeEquation() +
            RightBoundary2ndDerivativeEquation() +
            RightBoundary3rdDerivativeEquation()
        )
        system.add_top_boundary_equation(converter.to_y(top_combined, grid=grid))
        
class Derivative2DEquationSet(EquationSet2D):
    """2次元高階微分のための方程式セット"""

    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        """
        2次元高階微分方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: 計算格子
            test_func: テスト関数
            use_dirichlet: 使用しない（互換性のため残す）
            use_neumann: 使用しない（互換性のため残す）
        """
        # 変換器を作成
        converter = Equation1Dto2DConverter
        
        # 内部点における偏導関数方程式 - gridを渡す
        # 関数値
        system.add_equation(OriginalEquation2D(
            lambda x, y: test_func.f(x, y),
            grid=grid
        ))
        
        # 内部点の方程式 - 1次元方程式を各方向に拡張（gridも渡す）
        # X方向
        system.add_interior_x_equation(converter.to_x(Internal1stDerivativeEquation(), grid=grid))
        system.add_interior_x_equation(converter.to_x(Internal2ndDerivativeEquation(), grid=grid))
        system.add_interior_x_equation(converter.to_x(Internal3rdDerivativeEquation(), grid=grid))
        
        # Y方向
        system.add_interior_y_equation(converter.to_y(Internal1stDerivativeEquation(), grid=grid))
        system.add_interior_y_equation(converter.to_y(Internal2ndDerivativeEquation(), grid=grid))
        system.add_interior_y_equation(converter.to_y(Internal3rdDerivativeEquation(), grid=grid))
        
        # 境界点の方程式 - 1次元方程式を各方向に拡張（gridも渡す）
        # 左境界 (i=0)
        system.add_left_boundary_equation(converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid))
        system.add_left_boundary_equation(converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid))
        system.add_left_boundary_equation(converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid))
        
        # 右境界 (i=nx-1)
        system.add_right_boundary_equation(converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid))
        system.add_right_boundary_equation(converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid))
        system.add_right_boundary_equation(converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid))
        
        # 下境界 (j=0)
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid))
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid))
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid))
                
        # 上境界 (j=ny-1)
        system.add_top_boundary_equation(converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid))
        system.add_top_boundary_equation(converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid))
        system.add_top_boundary_equation(converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid))