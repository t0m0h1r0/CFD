from abc import ABC, abstractmethod
import cupy as cp

# 共通の方程式
from equation.poisson import PoissonEquation, PoissonEquation2D
from equation.original import OriginalEquation, OriginalEquation2D
from equation.boundary import (DirichletBoundaryEquation, NeumannBoundaryEquation,
                               DirichletXBoundaryEquation2D, DirichletYBoundaryEquation2D,
                               NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D)
from equation.compact_internal import (Internal1stDerivativeEquation,
                                      Internal2ndDerivativeEquation,
                                      Internal3rdDerivativeEquation)
from equation.compact_left_boundary import (LeftBoundary1stDerivativeEquation,
                                           LeftBoundary2ndDerivativeEquation,
                                           LeftBoundary3rdDerivativeEquation)
from equation.compact_right_boundary import (RightBoundary1stDerivativeEquation,
                                           RightBoundary2ndDerivativeEquation,
                                           RightBoundary3rdDerivativeEquation)
from equation.equation_converter import Equation1Dto2DConverter


class EquationSet(ABC):
    """統合された方程式セットの抽象基底クラス (1D/2D両対応)"""

    def __init__(self):
        """初期化"""
        # 境界条件の有効フラグ（サブクラスで変更可能）
        self.enable_dirichlet = True
        self.enable_neumann = True

    @abstractmethod
    def setup_equations(self, system, grid, test_func=None):
        """
        方程式システムに方程式を設定する
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (1D/2D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        pass

    def get_boundary_settings(self):
        """
        境界条件の設定を取得
        
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        return self.enable_dirichlet, self.enable_neumann

    def set_boundary_options(self, use_dirichlet=True, use_neumann=True):
        """
        境界条件のオプションを設定する
        
        Args:
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
            
        Returns:
            self: メソッドチェーン用
        """
        self.enable_dirichlet = use_dirichlet
        self.enable_neumann = use_neumann
        return self

    @classmethod
    def get_available_sets(cls, dimension=None):
        """
        利用可能な方程式セットを返す
        
        Args:
            dimension: 1または2 (Noneの場合は両方)
            
        Returns:
            利用可能な方程式セットの辞書
        """
        all_sets = {
            "poisson": {"1d": PoissonEquationSet1D, "2d": PoissonEquationSet2D},
            "derivative": {"1d": DerivativeEquationSet1D, "2d": DerivativeEquationSet2D},
        }
        
        if dimension == 1:
            return {key: value["1d"] for key, value in all_sets.items()}
        elif dimension == 2:
            return {key: value["2d"] for key, value in all_sets.items()}
        else:
            # 統合されたセットを返す（どちらのバージョンも持つ辞書）
            return all_sets

    @classmethod
    def create(cls, name, dimension=None):
        """
        名前から方程式セットを作成
        
        Args:
            name: 方程式セット名
            dimension: 1または2 (Noneの場合はgridの次元から判断)
            
        Returns:
            方程式セットのインスタンス
        """
        available_sets = cls.get_available_sets(dimension)
        
        # 文字列の前後の空白を削除
        name = name.strip()
        
        if name in available_sets:
            if dimension is None:
                # 次元が指定されていない場合
                if isinstance(available_sets[name], dict):
                    # デフォルトとして1Dを返す（実際の使用時にはgridから自動判別）
                    return DimensionalEquationSetWrapper(name, available_sets[name])
                else:
                    return available_sets[name]()
            else:
                # 1Dまたは2Dが明示的に指定されている場合
                return available_sets[name]() if not isinstance(available_sets[name], dict) else available_sets[name][f"{dimension}d"]()
        else:
            print(f"警告: 方程式セット '{name}' は利用できません。")
            print(f"利用可能なセット: {list(available_sets.keys())}")
            print("デフォルトの 'poisson' を使用します。")
            
            # デフォルトとしてpoissonを返す
            if dimension is None:
                if isinstance(available_sets["poisson"], dict):
                    return DimensionalEquationSetWrapper("poisson", available_sets["poisson"])
                else:
                    return available_sets["poisson"]()
            else:
                return available_sets["poisson"]() if not isinstance(available_sets["poisson"], dict) else available_sets["poisson"][f"{dimension}d"]()


class DimensionalEquationSetWrapper(EquationSet):
    """
    次元に基づいて適切な方程式セットを選択するラッパークラス
    """
    
    def __init__(self, name, dimension_sets):
        """
        初期化
        
        Args:
            name: 方程式セット名
            dimension_sets: 次元ごとの方程式セットクラス {"1d": Class1D, "2d": Class2D}
        """
        super().__init__()
        self.name = name
        self.dimension_sets = dimension_sets
        self._1d_instance = None
        self._2d_instance = None
    
    def setup_equations(self, system, grid, test_func=None):
        """
        方程式システムに方程式を設定する
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (1D/2D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        # Gridの次元に基づいて適切なインスタンスを使用
        if grid.is_2d:
            if self._2d_instance is None:
                self._2d_instance = self.dimension_sets["2d"]()
                
            # 境界条件の設定を継承
            self._2d_instance.set_boundary_options(self.enable_dirichlet, self.enable_neumann)
            
            # 方程式をセットアップして境界条件フラグを返す
            return self._2d_instance.setup_equations(system, grid, test_func)
        else:
            if self._1d_instance is None:
                self._1d_instance = self.dimension_sets["1d"]()
            
            # 境界条件の設定を継承
            self._1d_instance.set_boundary_options(self.enable_dirichlet, self.enable_neumann)
            
            # 方程式をセットアップして境界条件フラグを返す
            return self._1d_instance.setup_equations(system, grid, test_func)
    
    def set_boundary_options(self, use_dirichlet=True, use_neumann=True):
        """
        境界条件のオプションを設定する
        
        Args:
            use_dirichlet: ディリクレ境界条件を使用するかどうか
            use_neumann: ノイマン境界条件を使用するかどうか
            
        Returns:
            self: メソッドチェーン用
        """
        super().set_boundary_options(use_dirichlet, use_neumann)
        
        # 既存のインスタンスがあれば、そちらにも設定を反映
        if self._1d_instance is not None:
            self._1d_instance.set_boundary_options(use_dirichlet, use_neumann)
        if self._2d_instance is not None:
            self._2d_instance.set_boundary_options(use_dirichlet, use_neumann)
            
        return self


# ===== 1D 方程式セット =====

class PoissonEquationSet1D(EquationSet):
    """1次元ポアソン方程式のための方程式セット"""

    def setup_equations(self, system, grid, test_func=None):
        """
        ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if grid.is_2d:
            raise ValueError("1D方程式セットが2Dグリッドで使用されました")
            
        x_min = grid.x_min
        x_max = grid.x_max

        # ポアソン方程式本体 (psi''(x) = f(x)) - グリッドも渡す
        poisson_eq = PoissonEquation(grid=grid)
        system.add_equation(poisson_eq)

        # 内部点の補助方程式 - グリッドも渡す
        system.add_interior_equation(Internal1stDerivativeEquation(grid))
        system.add_interior_equation(Internal2ndDerivativeEquation(grid))
        system.add_interior_equation(Internal3rdDerivativeEquation(grid))

        # 境界条件設定 - 常に両方のタイプの方程式を追加
        # ディリクレ境界条件 (値固定) - グリッドも渡す
        dirichlet_left = DirichletBoundaryEquation(grid=grid)
        dirichlet_right = DirichletBoundaryEquation(grid=grid)
        system.add_left_boundary_equation(dirichlet_left)
        system.add_right_boundary_equation(dirichlet_right)

        # ノイマン境界条件 (導関数固定) - グリッドも渡す
        neumann_left = NeumannBoundaryEquation(grid=grid)
        neumann_right = NeumannBoundaryEquation(grid=grid)
        system.add_left_boundary_equation(neumann_left)
        system.add_right_boundary_equation(neumann_right)
                        
        # 3階導関数補助方程式 - グリッドも渡す
        system.add_left_boundary_equation(LeftBoundary3rdDerivativeEquation(grid))
        system.add_right_boundary_equation(RightBoundary3rdDerivativeEquation(grid))

        return self.enable_dirichlet, self.enable_neumann


class DerivativeEquationSet1D(EquationSet):
    """1次元高階微分のための方程式セット"""

    def setup_equations(self, system, grid, test_func=None):
        """
        導関数計算用の方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if grid.is_2d:
            raise ValueError("1D方程式セットが2Dグリッドで使用されました")
            
        # 元の関数を使用する方程式 - グリッドも渡す
        system.add_equation(OriginalEquation(grid=grid))

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

        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False


# ===== 2D 方程式セット =====

class PoissonEquationSet2D(EquationSet):
    """2次元ポアソン方程式のための方程式セット"""

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
        
        # ポアソン方程式: Δψ = f(x,y) - グリッドを渡す
        poisson_eq = PoissonEquation2D(grid=grid)
        system.add_equation(poisson_eq)

        # 内部点の方程式 - 1次元方程式を各方向に拡張（gridも渡す）
        # X方向
        system.add_interior_x_equation(converter.to_x(Internal1stDerivativeEquation(), grid=grid))
        system.add_interior_x_equation(converter.to_x(Internal2ndDerivativeEquation(), grid=grid))
        system.add_interior_x_equation(converter.to_x(Internal3rdDerivativeEquation(), grid=grid))
        
        # Y方向
        system.add_interior_y_equation(converter.to_y(Internal1stDerivativeEquation(), grid=grid))
        system.add_interior_y_equation(converter.to_y(Internal2ndDerivativeEquation(), grid=grid))
        system.add_interior_y_equation(converter.to_y(Internal3rdDerivativeEquation(), grid=grid))
        
        # 境界点の方程式 - ディリクレ境界条件
        # 左境界 (i=0)
        dirichlet_left = DirichletXBoundaryEquation2D(grid=grid)
        system.add_left_boundary_equation(dirichlet_left)
        # 右境界 (i=nx-1)
        dirichlet_right = DirichletXBoundaryEquation2D(grid=grid)
        system.add_right_boundary_equation(dirichlet_right)
        # 下境界 (j=0)
        dirichlet_bottom = DirichletYBoundaryEquation2D(grid=grid)
        system.add_bottom_boundary_equation(dirichlet_bottom)
        # 上境界 (j=ny-1)
        dirichlet_top = DirichletYBoundaryEquation2D(grid=grid)
        system.add_top_boundary_equation(dirichlet_top)
        
        # 境界点の方程式 - ノイマン境界条件
        # 左境界 (i=0)
        neumann_left = NeumannXBoundaryEquation2D(grid=grid)
        system.add_left_boundary_equation(neumann_left)
        # 右境界 (i=nx-1)
        neumann_right = NeumannXBoundaryEquation2D(grid=grid)
        system.add_right_boundary_equation(neumann_right)
        # 下境界 (j=0)
        neumann_bottom = NeumannYBoundaryEquation2D(grid=grid)
        system.add_bottom_boundary_equation(neumann_bottom)
        # 上境界 (j=ny-1)
        neumann_top = NeumannYBoundaryEquation2D(grid=grid)
        system.add_top_boundary_equation(neumann_top)
        
        # 左右境界の補助方程式
        # 左境界の補助方程式
        left_combined = converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid)
        system.add_left_boundary_equation(left_combined)
                
        # 右境界の補助方程式
        right_combined = converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid)
        system.add_right_boundary_equation(right_combined)
                
        # 上下境界の補助方程式
        # 下境界の補助方程式
        bottom_combined = converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid)
        system.add_bottom_boundary_equation(bottom_combined)
                
        # 上境界の補助方程式
        top_combined = converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid)
        system.add_top_boundary_equation(top_combined)
        
        return self.enable_dirichlet, self.enable_neumann
        
class DerivativeEquationSet2D(EquationSet):
    """2次元高階微分のための方程式セット"""

    def setup_equations(self, system, grid, test_func=None):
        """
        2次元高階微分方程式システムを設定
        
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
        
        # 内部点における偏導関数方程式 - gridを渡す
        # 関数値
        system.add_equation(OriginalEquation2D(grid=grid))
        
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

        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False