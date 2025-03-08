# equation_sets.py
from abc import ABC, abstractmethod
from typing import Optional
from grid import Grid
from test_functions import TestFunction
from equation_system import EquationSystem
from equation.poisson import PoissonEquation
from equation.custom import CustomEquation
from equation.boundary import DirichletBoundaryEquation, NeumannBoundaryEquation
from equation.essential import EssentialEquation
from equation.compact_internal import (
    Internal1stDerivativeEquation,
    Internal2ndDerivativeEquation,
    Internal3rdDerivativeEquation,
)
from equation.compact_left_boundary import (
    LeftBoundary1stDerivativeEquation,
    LeftBoundary2ndDerivativeEquation,
    LeftBoundary3rdDerivativeEquation,
)
from equation.compact_right_boundary import (
    RightBoundary1stDerivativeEquation,
    RightBoundary2ndDerivativeEquation,
    RightBoundary3rdDerivativeEquation,
)


class EquationSet(ABC):
    """方程式セットの抽象基底クラス"""

    @abstractmethod
    def setup_equations(
        self, 
        system: EquationSystem, 
        grid: Grid, 
        test_func: TestFunction, 
        use_dirichlet: bool = True, 
        use_neumann: bool = True
    ) -> None:
        """
        方程式システムに方程式を設定する

        Args:
            system: 方程式システム
            grid: 計算格子
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
        """
        pass

    @classmethod
    def get_available_sets(cls) -> dict:
        """
        利用可能な方程式セットを取得する

        Returns:
            利用可能な方程式セットの辞書 {名前: クラス}
        """
        return {
            "poisson": PoissonEquationSet,
            "derivative": DerivativeEquationSet,
        }

    @classmethod
    def create(cls, name: str) -> 'EquationSet':
        """
        名前から方程式セットのインスタンスを作成する

        Args:
            name: 方程式セットの名前

        Returns:
            方程式セットのインスタンス
        """
        available_sets = cls.get_available_sets()
        if name not in available_sets:
            raise ValueError(
                f"方程式セット '{name}' は利用できません。"
                f"利用可能なセット: {list(available_sets.keys())}"
            )
        return available_sets[name]()


class PoissonEquationSet(EquationSet):
    """
    ポアソン方程式 (ψ''(x) = f(x)) のための方程式セット
    """

    def setup_equations(
        self, 
        system: EquationSystem, 
        grid: Grid, 
        test_func: TestFunction, 
        use_dirichlet: bool = True, 
        use_neumann: bool = True
    ) -> None:
        """
        ポアソン方程式システムを設定する

        Args:
            system: 方程式システム
            grid: 計算格子
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
        """
        # グリッド情報
        x_min = grid.x_min
        x_max = grid.x_max

        # ポアソン方程式: ψ''(x) = f(x)
        system.add_equation(PoissonEquation(test_func.d2f))

        # 内部点の方程式を設定
        system.add_interior_equation(Internal1stDerivativeEquation())
        system.add_interior_equation(Internal2ndDerivativeEquation())
        system.add_interior_equation(Internal3rdDerivativeEquation())

        # 左境界の方程式を設定
        if use_dirichlet:
            system.add_left_boundary_equation(DirichletBoundaryEquation(value=test_func.f(x_min)))
            system.add_right_boundary_equation(DirichletBoundaryEquation(value=test_func.f(x_max)))
        else:
            system.add_left_boundary_equation(LeftBoundary1stDerivativeEquation())
            system.add_right_boundary_equation(RightBoundary1stDerivativeEquation())
           
        if use_neumann:
            system.add_left_boundary_equation(NeumannBoundaryEquation(value=test_func.df(x_min)))
            system.add_right_boundary_equation(NeumannBoundaryEquation(value=test_func.df(x_max)))
        else:
            system.add_left_boundary_equation(LeftBoundary2ndDerivativeEquation())
            system.add_right_boundary_equation(RightBoundary2ndDerivativeEquation())
                        
        system.add_left_boundary_equation(
            LeftBoundary1stDerivativeEquation()
            + LeftBoundary2ndDerivativeEquation()
            + LeftBoundary3rdDerivativeEquation()
        )
        system.add_right_boundary_equation(
            RightBoundary1stDerivativeEquation()
            + RightBoundary2ndDerivativeEquation()
            + RightBoundary3rdDerivativeEquation()
        )


class DerivativeEquationSet(EquationSet):
    """
    高階微分のための方程式セット (カスタム方程式を使用)
    """

    def setup_equations(
        self, 
        system: EquationSystem, 
        grid: Grid, 
        test_func: TestFunction, 
        use_dirichlet: bool = True, 
        use_neumann: bool = True
    ) -> None:
        """
        高階微分の方程式システムを設定する

        Args:
            system: 方程式システム
            grid: 計算格子
            test_func: テスト関数
            use_dirichlet: 使用しない（インターフェース互換性のため維持）
            use_neumann: 使用しない（インターフェース互換性のため維持）
        """
        # カスタム方程式: ψ(x) = f(x) （最初の成分だけが1で他が0になる係数ベクトル）
        system.add_equation(CustomEquation(f_func=test_func.f, coeff=[1, 0, 0, 0]))

        # 内部点の方程式を設定
        system.add_interior_equation(Internal1stDerivativeEquation())
        system.add_interior_equation(Internal2ndDerivativeEquation())
        system.add_interior_equation(Internal3rdDerivativeEquation())

        # 左境界の方程式を設定
        system.add_left_boundary_equation(LeftBoundary1stDerivativeEquation())
        system.add_left_boundary_equation(LeftBoundary2ndDerivativeEquation())
        system.add_left_boundary_equation(LeftBoundary3rdDerivativeEquation())

        # 右境界の方程式を設定
        system.add_right_boundary_equation(RightBoundary1stDerivativeEquation())
        system.add_right_boundary_equation(RightBoundary2ndDerivativeEquation())
        system.add_right_boundary_equation(RightBoundary3rdDerivativeEquation())
