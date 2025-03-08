from abc import ABC, abstractmethod
from equation.poisson import PoissonEquation
from equation.custom import CustomEquation
from equation.boundary import DirichletBoundaryEquation, NeumannBoundaryEquation
from equation.compact_internal import Internal1stDerivativeEquation, Internal2ndDerivativeEquation, Internal3rdDerivativeEquation
from equation.compact_left_boundary import LeftBoundary1stDerivativeEquation, LeftBoundary2ndDerivativeEquation, LeftBoundary3rdDerivativeEquation
from equation.compact_right_boundary import RightBoundary1stDerivativeEquation, RightBoundary2ndDerivativeEquation, RightBoundary3rdDerivativeEquation


class EquationSet(ABC):
    """方程式セットの抽象基底クラス"""

    @abstractmethod
    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        """方程式システムに方程式を設定する"""
        pass

    @classmethod
    def get_available_sets(cls):
        return {
            "poisson": PoissonEquationSet,
            "derivative": DerivativeEquationSet,
        }

    @classmethod
    def create(cls, name):
        available_sets = cls.get_available_sets()
        if name not in available_sets:
            name = "poisson"
        return available_sets[name]()


class PoissonEquationSet(EquationSet):
    """ポアソン方程式のための方程式セット"""

    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        x_min = grid.x_min
        x_max = grid.x_max

        system.add_equation(PoissonEquation(test_func.d2f))

        system.add_interior_equation(Internal1stDerivativeEquation())
        system.add_interior_equation(Internal2ndDerivativeEquation())
        system.add_interior_equation(Internal3rdDerivativeEquation())

        if use_dirichlet:
            system.add_left_boundary_equation(DirichletBoundaryEquation(test_func.f(x_min)))
            system.add_right_boundary_equation(DirichletBoundaryEquation(test_func.f(x_max)))
        else:
            system.add_left_boundary_equation(LeftBoundary1stDerivativeEquation())
            system.add_right_boundary_equation(RightBoundary1stDerivativeEquation())
           
        if use_neumann:
            system.add_left_boundary_equation(NeumannBoundaryEquation(test_func.df(x_min)))
            system.add_right_boundary_equation(NeumannBoundaryEquation(test_func.df(x_max)))
        else:
            system.add_left_boundary_equation(LeftBoundary2ndDerivativeEquation())
            system.add_right_boundary_equation(RightBoundary2ndDerivativeEquation())
                        
        system.add_left_boundary_equation(LeftBoundary3rdDerivativeEquation())
        system.add_right_boundary_equation(RightBoundary3rdDerivativeEquation())


class DerivativeEquationSet(EquationSet):
    """高階微分のための方程式セット"""

    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        system.add_equation(CustomEquation(f_func=test_func.f, coeff=[1, 0, 0, 0]))

        system.add_interior_equation(Internal1stDerivativeEquation())
        system.add_interior_equation(Internal2ndDerivativeEquation())
        system.add_interior_equation(Internal3rdDerivativeEquation())

        system.add_left_boundary_equation(LeftBoundary1stDerivativeEquation())
        system.add_left_boundary_equation(LeftBoundary2ndDerivativeEquation())
        system.add_left_boundary_equation(LeftBoundary3rdDerivativeEquation())

        system.add_right_boundary_equation(RightBoundary1stDerivativeEquation())
        system.add_right_boundary_equation(RightBoundary2ndDerivativeEquation())
        system.add_right_boundary_equation(RightBoundary3rdDerivativeEquation())