import cupy as cp
from abc import ABC, abstractmethod

from equation.equation_converter import Equation1Dto2DConverter
from equation.poisson2d import PoissonEquation2D
from equation.compact_internal import Internal1stDerivativeEquation, Internal2ndDerivativeEquation, Internal3rdDerivativeEquation
from equation.compact_left_boundary import LeftBoundary1stDerivativeEquation, LeftBoundary2ndDerivativeEquation, LeftBoundary3rdDerivativeEquation
from equation.compact_right_boundary import RightBoundary1stDerivativeEquation, RightBoundary2ndDerivativeEquation, RightBoundary3rdDerivativeEquation
from equation.base2d import Equation2D

class CustomEquation2D(Equation2D):
    """2D用のカスタム方程式"""
    
    def __init__(self, f_func, coeffs):
        """
        初期化
        
        Args:
            f_func: 関数 f(x, y)
            coeffs: 係数リスト [c0, c1, c2, c3, c4, c5, c6]
                    各係数は [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy] に対応
        """
        self.f_func = f_func
        self.coeffs = cp.array(coeffs)
    
    def get_stencil_coefficients(self, grid, i, j):
        """ステンシル係数を取得"""
        return {(0, 0): self.coeffs}
    
    def get_rhs(self, grid, i, j):
        """右辺値を取得"""
        x, y = grid.get_point(i, j)
        return self.f_func(x, y)
    
    def is_valid_at(self, grid, i, j):
        """指定点で有効かどうかを判定"""
        return True


class EquationSet2D(ABC):
    """方程式セットの抽象基底クラス (2D版)"""

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

    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        """2次元ポアソン方程式システムを設定"""
        # 変換器を作成
        converter = Equation1Dto2DConverter
        
        # ポアソン方程式: Δψ = f(x,y)
        def poisson_source(x, y):
            return -(test_func.d2f_dx2(x, y) + test_func.d2f_dy2(x, y))
        poisson_eq = PoissonEquation2D(poisson_source)
        system.add_interior_equation(poisson_eq)
        
        # 内部点の方程式 - 1次元方程式を各方向に拡張
        # X方向
        system.add_interior_equation(converter.to_x(Internal1stDerivativeEquation()))
        system.add_interior_equation(converter.to_x(Internal2ndDerivativeEquation()))
        system.add_interior_equation(converter.to_x(Internal3rdDerivativeEquation()))
        
        # Y方向
        system.add_interior_equation(converter.to_y(Internal1stDerivativeEquation()))
        system.add_interior_equation(converter.to_y(Internal2ndDerivativeEquation()))
        system.add_interior_equation(converter.to_y(Internal3rdDerivativeEquation()))
        
        # 境界点の方程式 - 1次元と同様の考え方
        # 左境界 (i=0)
        system.add_left_boundary_equation(converter.to_x(LeftBoundary1stDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_x(LeftBoundary2ndDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_x(LeftBoundary3rdDerivativeEquation()))
        
        # Y方向微分も左境界に必要
        system.add_left_boundary_equation(converter.to_y(Internal1stDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_y(Internal2ndDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_y(Internal3rdDerivativeEquation()))
        
        # 右境界 (i=nx-1)
        system.add_right_boundary_equation(converter.to_x(RightBoundary1stDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_x(RightBoundary2ndDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_x(RightBoundary3rdDerivativeEquation()))
        
        # Y方向微分も右境界に必要
        system.add_right_boundary_equation(converter.to_y(Internal1stDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_y(Internal2ndDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_y(Internal3rdDerivativeEquation()))
        
        # 下境界 (j=0)
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary1stDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary2ndDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary3rdDerivativeEquation()))
        
        # X方向微分も下境界に必要
        system.add_bottom_boundary_equation(converter.to_x(Internal1stDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_x(Internal2ndDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_x(Internal3rdDerivativeEquation()))
        
        # 上境界 (j=ny-1)
        system.add_top_boundary_equation(converter.to_y(RightBoundary1stDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_y(RightBoundary2ndDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_y(RightBoundary3rdDerivativeEquation()))
        
        # X方向微分も上境界に必要
        system.add_top_boundary_equation(converter.to_x(Internal1stDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_x(Internal2ndDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_x(Internal3rdDerivativeEquation()))

class Derivative2DEquationSet(EquationSet2D):
    """2次元高階微分のための方程式セット"""

    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        """2次元高階微分方程式システムを設定"""
        # 変換器を作成
        converter = Equation1Dto2DConverter
        
        # 内部点における偏導関数方程式
        # 関数値
        system.add_interior_equation(CustomEquation2D(
            lambda x, y: test_func.f(x, y),
            [1, 0, 0, 0, 0, 0, 0]
        ))
        
        # X方向微分
        # 内部点の方程式 - 1次元方程式を各方向に拡張
        # X方向
        system.add_interior_equation(converter.to_x(Internal1stDerivativeEquation()))
        system.add_interior_equation(converter.to_x(Internal2ndDerivativeEquation()))
        system.add_interior_equation(converter.to_x(Internal3rdDerivativeEquation()))
        
        # Y方向
        system.add_interior_equation(converter.to_y(Internal1stDerivativeEquation()))
        system.add_interior_equation(converter.to_y(Internal2ndDerivativeEquation()))
        system.add_interior_equation(converter.to_y(Internal3rdDerivativeEquation()))
        
        # 境界点の方程式 - 1次元と同様の考え方
        # 左境界 (i=0)
        system.add_left_boundary_equation(converter.to_x(LeftBoundary1stDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_x(LeftBoundary2ndDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_x(LeftBoundary3rdDerivativeEquation()))
        
        # Y方向微分も左境界に必要
        system.add_left_boundary_equation(converter.to_y(Internal1stDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_y(Internal2ndDerivativeEquation()))
        system.add_left_boundary_equation(converter.to_y(Internal3rdDerivativeEquation()))
        
        # 右境界 (i=nx-1)
        system.add_right_boundary_equation(converter.to_x(RightBoundary1stDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_x(RightBoundary2ndDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_x(RightBoundary3rdDerivativeEquation()))
        
        # Y方向微分も右境界に必要
        system.add_right_boundary_equation(converter.to_y(Internal1stDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_y(Internal2ndDerivativeEquation()))
        system.add_right_boundary_equation(converter.to_y(Internal3rdDerivativeEquation()))
        
        # 下境界 (j=0)
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary1stDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary2ndDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_y(LeftBoundary3rdDerivativeEquation()))
        
        # X方向微分も下境界に必要
        system.add_bottom_boundary_equation(converter.to_x(Internal1stDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_x(Internal2ndDerivativeEquation()))
        system.add_bottom_boundary_equation(converter.to_x(Internal3rdDerivativeEquation()))
        
        # 上境界 (j=ny-1)
        system.add_top_boundary_equation(converter.to_y(RightBoundary1stDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_y(RightBoundary2ndDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_y(RightBoundary3rdDerivativeEquation()))
        
        # X方向微分も上境界に必要
        system.add_top_boundary_equation(converter.to_x(Internal1stDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_x(Internal2ndDerivativeEquation()))
        system.add_top_boundary_equation(converter.to_x(Internal3rdDerivativeEquation()))