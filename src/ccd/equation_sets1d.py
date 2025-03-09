from abc import ABC, abstractmethod
from equation.poisson import PoissonEquation
from equation.original import OriginalEquation
from equation.custom import CustomEquation
from equation.boundary import DirichletBoundaryEquation, NeumannBoundaryEquation
from equation.compact_internal import Internal1stDerivativeEquation, Internal2ndDerivativeEquation, Internal3rdDerivativeEquation
from equation.compact_left_boundary import LeftBoundary1stDerivativeEquation, LeftBoundary2ndDerivativeEquation, LeftBoundary3rdDerivativeEquation
from equation.compact_right_boundary import RightBoundary1stDerivativeEquation, RightBoundary2ndDerivativeEquation, RightBoundary3rdDerivativeEquation


class EquationSet(ABC):
    """方程式セットの抽象基底クラス"""

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
            "poisson": PoissonEquationSet,
            "derivative": DerivativeEquationSet,
        }

    @classmethod
    def create(cls, name):
        """名前から方程式セットを作成"""
        available_sets = {
            "poisson": PoissonEquationSet,
            "derivative": DerivativeEquationSet,
        }
        
        # 文字列の前後の空白を削除
        name = name.strip()
        
        if name in available_sets:
            return available_sets[name]()
        else:
            print(f"警告: 方程式セット '{name}' は利用できません。")
            print(f"利用可能なセット: {list(available_sets.keys())}")
            print("デフォルトの 'poisson' を使用します。")
            return available_sets["poisson"]()


class PoissonEquationSet(EquationSet):
    """ポアソン方程式のための方程式セット"""

    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        """
        ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: 計算格子
            test_func: テスト関数
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
        """
        x_min = grid.x_min
        x_max = grid.x_max

        # ポアソン方程式本体 (psi''(x) = f(x)) - グリッドも渡す
        poisson_eq = PoissonEquation(test_func.d2f, grid)
        system.add_equation(poisson_eq)

        # 内部点の補助方程式 - グリッドも渡す
        system.add_interior_equation(Internal1stDerivativeEquation(grid))
        system.add_interior_equation(Internal2ndDerivativeEquation(grid))
        system.add_interior_equation(Internal3rdDerivativeEquation(grid))

        # 境界条件設定
        if use_dirichlet:
            # ディリクレ境界条件 (値固定) - グリッドも渡す
            system.add_left_boundary_equation(DirichletBoundaryEquation(test_func.f(x_min), grid))
            system.add_right_boundary_equation(DirichletBoundaryEquation(test_func.f(x_max), grid))
        else:
            # 代わりに1階導関数補助方程式 - グリッドも渡す
            system.add_left_boundary_equation(LeftBoundary1stDerivativeEquation(grid))
            system.add_right_boundary_equation(RightBoundary1stDerivativeEquation(grid))
           
        if use_neumann:
            # ノイマン境界条件 (導関数固定) - グリッドも渡す
            system.add_left_boundary_equation(NeumannBoundaryEquation(test_func.df(x_min), grid))
            system.add_right_boundary_equation(NeumannBoundaryEquation(test_func.df(x_max), grid))
        else:
            # 代わりに2階導関数補助方程式 - グリッドも渡す
            system.add_left_boundary_equation(LeftBoundary2ndDerivativeEquation(grid))
            system.add_right_boundary_equation(RightBoundary2ndDerivativeEquation(grid))
                        
        # 3階導関数補助方程式 - グリッドも渡す
        system.add_left_boundary_equation(LeftBoundary3rdDerivativeEquation(grid))
        system.add_right_boundary_equation(RightBoundary3rdDerivativeEquation(grid))


class DerivativeEquationSet(EquationSet):
    """高階微分のための方程式セット"""

    def setup_equations(self, system, grid, test_func, use_dirichlet=True, use_neumann=True):
        """
        導関数計算用の方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: 計算格子
            test_func: テスト関数
            use_dirichlet: 使用しない（互換性のため残す）
            use_neumann: 使用しない（互換性のため残す）
        """
        # 元の関数を使用する方程式 - グリッドも渡す
        system.add_equation(OriginalEquation(f_func=test_func.f, grid=grid))

        # 内部点の補助方程式 - グリッドも渡す
        system.add_interior_equation(Internal1stDerivativeEquation(grid))
        system.add_interior_equation(Internal2ndDerivativeEquation(grid))
        system.add_interior_equation(Internal3rdDerivativeEquation(grid))

        # 左境界点の補助方程式 - グリッドも渡す
        system.add_left_boundary_equation(LeftBoundary1stDerivativeEquation(grid))
        system.add_left_boundary_equation(LeftBoundary2ndDerivativeEquation(grid))
        system.add_left_boundary_equation(LeftBoundary3rdDerivativeEquation(grid))

        # 右境界点の補助方程式 - グリッドも渡す
        system.add_right_boundary_equation(RightBoundary1stDerivativeEquation(grid))
        system.add_right_boundary_equation(RightBoundary2ndDerivativeEquation(grid))
        system.add_right_boundary_equation(RightBoundary3rdDerivativeEquation(grid))