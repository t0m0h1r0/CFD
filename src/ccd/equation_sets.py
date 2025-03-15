"""
方程式セットの定義と管理を行うモジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
様々な種類の方程式セットを定義し、次元に応じて適切に管理します。
各方程式セットは境界条件タイプ（ディリクレ+ノイマン、ディリクレのみ、境界条件なし）ごとに実装されています。
"""

from abc import ABC, abstractmethod
import cupy as cp

# 共通の方程式をインポート
from equation.poisson import PoissonEquation, PoissonEquation2D
from equation.original import OriginalEquation, OriginalEquation2D
from equation.boundary import (
    DirichletBoundaryEquation, NeumannBoundaryEquation,
    DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
)
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
        name = name.strip()
        
        if name in available_sets:
            if dimension is None:
                # 次元が指定されていない場合
                if isinstance(available_sets[name], dict):
                    return DimensionalEquationSetWrapper(name, available_sets[name])
                else:
                    return available_sets[name]()
            else:
                # 次元が明示的に指定されている場合
                if isinstance(available_sets[name], dict):
                    return available_sets[name][f"{dimension}d"]()
                else:
                    return available_sets[name]()
        else:
            print(f"警告: 方程式セット '{name}' は利用できません。デフォルトの 'poisson' を使用します。")
            print(f"利用可能なセット: {list(available_sets.keys())}")
            
            # デフォルトとしてpoissonを返す
            if dimension is None:
                if isinstance(available_sets["poisson"], dict):
                    return DimensionalEquationSetWrapper("poisson", available_sets["poisson"])
                else:
                    return available_sets["poisson"]()
            else:
                if isinstance(available_sets["poisson"], dict):
                    return available_sets["poisson"][f"{dimension}d"]()
                else:
                    return available_sets["poisson"]()


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
class DerivativeEquationSet1D(EquationSet):
    """1次元高階微分のための方程式セット"""

    def __init__(self):
        """初期化"""
        super().__init__()
        # 微分方程式セットでは境界条件は無効
        self.enable_dirichlet = False
        self.enable_neumann = False
    
    def setup_equations(self, system, grid, test_func=None):
        """
        1次元導関数計算用の方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (1D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if grid.is_2d:
            raise ValueError("1D方程式セットが2Dグリッドで使用されました")
        
        # 元の関数を全ての領域に追加
        system.add_dominant_equation(OriginalEquation(grid=grid))
        
        # 内部点の方程式設定
        system.add_equation('interior', Internal1stDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal2ndDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal3rdDerivativeEquation(grid=grid))
        
        # 左境界の方程式設定
        system.add_equation('left', LeftBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('left', LeftBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('left', LeftBoundary3rdDerivativeEquation(grid=grid))
        
        # 右境界の方程式設定
        system.add_equation('right', RightBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('right', RightBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('right', RightBoundary3rdDerivativeEquation(grid=grid))
        
        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False


# 境界条件タイプ別の具象クラス (1D)
class PoissonEquationSet1D(EquationSet):
    """ディリクレ・ノイマン混合境界条件の1Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = True
    
    def setup_equations(self, system, grid):
        """
        ディリクレ・ノイマン混合境界条件の1Dポアソン方程式セットをセットアップ
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if grid.is_2d:
            raise ValueError("1D方程式セットが2Dグリッドで使用されました")
        
        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation(grid=grid))
        
        # 境界条件を設定
        system.add_equation('left', DirichletBoundaryEquation(grid=grid))
        system.add_equation('right', DirichletBoundaryEquation(grid=grid))
        system.add_equation('left', NeumannBoundaryEquation(grid=grid))
        system.add_equation('right', NeumannBoundaryEquation(grid=grid))
        
        # 内部点の方程式設定
        system.add_equation('interior', Internal1stDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal2ndDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal3rdDerivativeEquation(grid=grid))
        
        # 左境界の方程式設定
        system.add_equation('left', LeftBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('left', LeftBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('left', LeftBoundary3rdDerivativeEquation(grid=grid))
        
        # 右境界の方程式設定
        system.add_equation('right', RightBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('right', RightBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('right', RightBoundary3rdDerivativeEquation(grid=grid))
        
        return True, True

# ===== 2D 方程式セット =====
class DerivativeEquationSet2D(EquationSet):
    """2次元高階微分のための方程式セット"""

    def __init__(self):
        """初期化"""
        super().__init__()
        # 微分方程式セットでは境界条件は無効
        self.enable_dirichlet = False
        self.enable_neumann = False
    
    def setup_equations(self, system, grid):
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


# 境界条件タイプ別の具象クラス (2D)
class PoissonEquationSet2D(EquationSet):
    """ディリクレ・ノイマン混合境界条件の2Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = True
    
    def setup_equations(self, system, grid):
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
            PoissonEquation2D(grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # 左境界用の方程式
        system.add_equations('left', [
            PoissonEquation2D(grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary2ndDerivativeEquation()+
                LeftBoundary3rdDerivativeEquation(),
                grid=grid
            ),
            converter.to_y(Internal1stDerivativeEquation(),grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(),grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(),grid=grid),
        ])

        # 右境界用の方程式
        system.add_equations('right', [
            PoissonEquation2D(grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation()+
                RightBoundary2ndDerivativeEquation()+
                RightBoundary3rdDerivativeEquation(),
                grid=grid
            ),
            converter.to_y(Internal1stDerivativeEquation(),grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(),grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(),grid=grid),
        ])
        
        # 下境界用の方程式
        system.add_equations('bottom', [
            PoissonEquation2D(grid=grid),
            converter.to_x(Internal1stDerivativeEquation(),grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(),grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(),grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary2ndDerivativeEquation()+
                LeftBoundary3rdDerivativeEquation(),
                grid=grid
            )
        ])

        # 上境界用の方程式
        system.add_equations('top', [
            PoissonEquation2D(grid=grid),
            converter.to_x(Internal1stDerivativeEquation(),grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(),grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(),grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation()+
                RightBoundary2ndDerivativeEquation()+
                RightBoundary3rdDerivativeEquation(),
                grid=grid
            )
        ])
        
        # 左下角 (i=0, j=0) - x方向マージ
        system.add_equations('left_bottom', [
            PoissonEquation2D(grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation() +
                LeftBoundary2ndDerivativeEquation() +
                LeftBoundary3rdDerivativeEquation(),
                grid=grid
            ),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                grid=grid
            ),
            converter.to_y(
                LeftBoundary2ndDerivativeEquation() +
                LeftBoundary3rdDerivativeEquation(),
                grid=grid
            )
        ])
        
        # 右上角 (i=nx-1, j=ny-1) - x方向マージ
        system.add_equations('right_top', [
            PoissonEquation2D(grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation() +
                RightBoundary2ndDerivativeEquation() +
                RightBoundary3rdDerivativeEquation(),
                grid=grid
            ),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                grid=grid
            ),
            converter.to_y(
                RightBoundary2ndDerivativeEquation()+
                RightBoundary3rdDerivativeEquation(),
                grid=grid
            )
        ])
        
        # 左上角 (i=0, j=ny-1) - y方向マージ
        system.add_equations('left_top', [
            PoissonEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                grid=grid
            ),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary2ndDerivativeEquation() +
                LeftBoundary3rdDerivativeEquation(),
                grid=grid
            ),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation() +
                RightBoundary2ndDerivativeEquation() +
                RightBoundary3rdDerivativeEquation(),
                grid=grid
            )
        ])
        
        # 右下角 (i=nx-1, j=0) - y方向マージ
        system.add_equations('right_bottom', [
            PoissonEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                grid=grid
            ),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary2ndDerivativeEquation() +
                RightBoundary3rdDerivativeEquation(),
                grid=grid
            ),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation() +
                LeftBoundary2ndDerivativeEquation() +
                LeftBoundary3rdDerivativeEquation(),
                grid=grid
            )
        ])
        
        return True, True