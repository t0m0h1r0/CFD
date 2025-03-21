"""
方程式セットの定義と管理を行うモジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
様々な種類の方程式セットを定義し、次元に応じて適切に管理します。
各方程式セットは境界条件タイプ（ディリクレ+ノイマン、ディリクレのみ、境界条件なし）ごとに実装されています。
"""

from abc import ABC, abstractmethod

# 共通の方程式をインポート
from equation.poisson import PoissonEquation, PoissonEquation2D, PoissonEquation3D
from equation.original import OriginalEquation, OriginalEquation2D, OriginalEquation3D
from equation.boundary import (
    DirichletBoundaryEquation, NeumannBoundaryEquation,
    DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D,
    DirichletBoundaryEquation3D, NeumannXBoundaryEquation3D, NeumannYBoundaryEquation3D, NeumannZBoundaryEquation3D
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
from equation.equation_converter import (
    Equation1Dto2DConverter,
    Equation1Dto3DConverter
)


class EquationSet(ABC):
    """統合された方程式セットの抽象基底クラス (1D/2D/3D対応)"""

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
            grid: Grid オブジェクト (1D/2D/3D)
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
            dimension: 1, 2, または3 (Noneの場合は全て)
            
        Returns:
            利用可能な方程式セットの辞書
        """
        all_sets = {
            "poisson": {"1d": PoissonEquationSet1D, "2d": PoissonEquationSet2D, "3d": PoissonEquationSet3D},
            "poisson2": {"1d": PoissonEquationSet1D2, "2d": PoissonEquationSet2D2, "3d": PoissonEquationSet3D2},
            "derivative": {"1d": DerivativeEquationSet1D, "2d": DerivativeEquationSet2D, "3d": DerivativeEquationSet3D},
        }
        
        if dimension == 1:
            return {key: value["1d"] for key, value in all_sets.items()}
        elif dimension == 2:
            return {key: value["2d"] for key, value in all_sets.items()}
        elif dimension == 3:
            return {key: value["3d"] for key, value in all_sets.items()}
        else:
            return all_sets

    @classmethod
    def create(cls, name, dimension=None):
        """
        名前から方程式セットを作成
        
        Args:
            name: 方程式セット名
            dimension: 1, 2, または3 (Noneの場合はgridの次元から判断)
            
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
            dimension_sets: 次元ごとの方程式セットクラス {"1d": Class1D, "2d": Class2D, "3d": Class3D}
        """
        super().__init__()
        self.name = name
        self.dimension_sets = dimension_sets
        self._1d_instance = None
        self._2d_instance = None
        self._3d_instance = None
    
    def setup_equations(self, system, grid, test_func=None):
        """
        方程式システムに方程式を設定する
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (1D/2D/3D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        # Gridの次元に基づいて適切なインスタンスを使用
        if grid.is_3d:
            if self._3d_instance is None:
                self._3d_instance = self.dimension_sets["3d"]()
                
            # 境界条件の設定を継承
            self._3d_instance.set_boundary_options(self.enable_dirichlet, self.enable_neumann)
            
            # 方程式をセットアップして境界条件フラグを返す
            return self._3d_instance.setup_equations(system, grid, test_func)
        elif grid.is_2d:
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
        if self._3d_instance is not None:
            self._3d_instance.set_boundary_options(use_dirichlet, use_neumann)
            
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
    
    def setup_equations(self, system, grid, test_func=None):
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
        
        # 内部点の方程式設定
        system.add_equation('interior', Internal1stDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal2ndDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal3rdDerivativeEquation(grid=grid))
        
        # 左境界の方程式設定
        if self.enable_dirichlet:
            system.add_equation('left', DirichletBoundaryEquation(grid=grid))
        if self.enable_neumann:
            system.add_equation('left', NeumannBoundaryEquation(grid=grid))
            
        system.add_equation('left', LeftBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('left', LeftBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('left', LeftBoundary3rdDerivativeEquation(grid=grid))
        
        # 右境界の方程式設定
        if self.enable_dirichlet:
            system.add_equation('right', DirichletBoundaryEquation(grid=grid))
        if self.enable_neumann:
            system.add_equation('right', NeumannBoundaryEquation(grid=grid))
            
        system.add_equation('right', RightBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('right', RightBoundary2ndDerivativeEquation(grid=grid))
        system.add_equation('right', RightBoundary3rdDerivativeEquation(grid=grid))
        
        return self.enable_dirichlet, self.enable_neumann

# ===== 2D 方程式セット =====
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


# 境界条件タイプ別の具象クラス (2D)
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
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ]
            
        system.add_equations('left', x_left_eqs)

        # 右境界用の方程式
        x_right_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation()+
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
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
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
        
        system.add_equations('bottom', y_bottom_eqs)
            
        # 上境界用の方程式
        y_top_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation()+
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
        
        system.add_equations('top', y_top_eqs)
        
        # 左下角 (i=0, j=0)
        left_bottom_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
                    
        system.add_equations('left_bottom', left_bottom_eqs)
        
        # 右上角 (i=nx-1, j=ny-1)
        right_top_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation()+
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation()+
                RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
            
        system.add_equations('right_top', right_top_eqs)
        
        # 左上角 (i=0, j=ny-1)
        left_top_eqs = [
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation()+
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
            
        system.add_equations('left_top', left_top_eqs)
        
        # 右下角 (i=nx-1, j=0)
        right_bottom_eqs = [
            NeumannXBoundaryEquation2D(grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation()+
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            NeumannYBoundaryEquation2D(grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation()+
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
                    
        system.add_equations('right_bottom', right_bottom_eqs)
        
        return True, True
    
# 境界条件タイプ別の具象クラス (1D)
class PoissonEquationSet1D2(EquationSet):
    """ディリクレ境界条件のみの1Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = False  # ノイマン境界条件を無効化
    
    def setup_equations(self, system, grid, test_func=None):
        """
        ディリクレ境界条件のみの1Dポアソン方程式セットをセットアップ
        
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
        
        # 内部点の方程式設定
        system.add_equation('interior', Internal1stDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal2ndDerivativeEquation(grid=grid))
        system.add_equation('interior', Internal3rdDerivativeEquation(grid=grid))
        
        # 左境界の方程式設定
        system.add_equation('left', DirichletBoundaryEquation(grid=grid))
        system.add_equation('left', LeftBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('left', 
                            LeftBoundary2ndDerivativeEquation(grid=grid)+
                            LeftBoundary3rdDerivativeEquation(grid=grid))
        
        # 右境界の方程式設定
        system.add_equation('right', DirichletBoundaryEquation(grid=grid))
        system.add_equation('right', RightBoundary1stDerivativeEquation(grid=grid))
        system.add_equation('right', 
                            RightBoundary2ndDerivativeEquation(grid=grid)+
                            RightBoundary3rdDerivativeEquation(grid=grid))
        
        return self.enable_dirichlet, self.enable_neumann


# 境界条件タイプ別の具象クラス (2D)
class PoissonEquationSet2D2(EquationSet):
    """ディリクレ境界条件のみの2Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = False  # ノイマン境界条件を無効化
    
    def setup_equations(self, system, grid, test_func=None):
        """
        ディリクレ境界条件のみの2次元ポアソン方程式システムを設定
        
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
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_x(
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid)
        ]
            
        system.add_equations('left', x_left_eqs)

        # 右境界用の方程式
        x_right_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_x(
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
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
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_y(
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
        
        system.add_equations('bottom', y_bottom_eqs)
            
        # 上境界用の方程式
        y_top_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_y(
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
        
        system.add_equations('top', y_top_eqs)
        
        # 左下角 (i=0, j=0)
        left_bottom_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_x(
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_y(
                LeftBoundary2ndDerivativeEquation(), 
                grid=grid),
            converter.to_y(
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
                    
        system.add_equations('left_bottom', left_bottom_eqs)
        
        # 右上角 (i=nx-1, j=ny-1)
        right_top_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_x(
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_y(
                RightBoundary2ndDerivativeEquation(), 
                grid=grid),
            converter.to_y(
                RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
            
        system.add_equations('right_top', right_top_eqs)
        
        # 左上角 (i=0, j=ny-1)
        left_top_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_x(
                LeftBoundary2ndDerivativeEquation(), 
                grid=grid),
            converter.to_x(
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_y(
                RightBoundary2ndDerivativeEquation()+ 
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
            
        system.add_equations('left_top', left_top_eqs)
        
        # 右下角 (i=nx-1, j=0)
        right_bottom_eqs = [
            DirichletBoundaryEquation2D(grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_x(
                RightBoundary2ndDerivativeEquation(), 
                grid=grid),
            converter.to_x(
                RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                grid=grid),
            converter.to_y(
                LeftBoundary2ndDerivativeEquation()+ 
                LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
        ]
                    
        system.add_equations('right_bottom', right_bottom_eqs)
        
        return True, False  # ディリクレト境界条件のみ有効

# ===== 3D 方程式セット =====
class DerivativeEquationSet3D(EquationSet):
    """3次元高階微分のための方程式セット"""

    def __init__(self):
        """初期化"""
        super().__init__()
        # 微分方程式セットでは境界条件は無効
        self.enable_dirichlet = False
        self.enable_neumann = False
    
    def setup_equations(self, system, grid, test_func=None):
        """
        3次元導関数計算用の方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (3D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if not grid.is_3d:
            raise ValueError("3D方程式セットが非3Dグリッドで使用されました")
        
        # 変換器を作成
        converter = Equation1Dto3DConverter
        
        # 元の関数を全ての領域に追加
        system.add_dominant_equation(OriginalEquation3D(grid=grid))
        
        # 内部点用の方程式
        system.add_equations('interior', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x面境界 (左右境界)
        for region in ['left', 'right']:
            system.add_equations(region, [
                converter.to_x(LeftBoundary1stDerivativeEquation() if region == 'left' else 
                             RightBoundary1stDerivativeEquation(), grid=grid),
                converter.to_x(LeftBoundary2ndDerivativeEquation() if region == 'left' else 
                             RightBoundary2ndDerivativeEquation(), grid=grid),
                converter.to_x(LeftBoundary3rdDerivativeEquation() if region == 'left' else 
                             RightBoundary3rdDerivativeEquation(), grid=grid),
                converter.to_y(Internal1stDerivativeEquation(), grid=grid),
                converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_z(Internal1stDerivativeEquation(), grid=grid),
                converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
            ])
        
        # y面境界 (下上境界)
        for region in ['bottom', 'top']:
            system.add_equations(region, [
                converter.to_x(Internal1stDerivativeEquation(), grid=grid),
                converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_y(LeftBoundary1stDerivativeEquation() if region == 'bottom' else 
                             RightBoundary1stDerivativeEquation(), grid=grid),
                converter.to_y(LeftBoundary2ndDerivativeEquation() if region == 'bottom' else 
                             RightBoundary2ndDerivativeEquation(), grid=grid),
                converter.to_y(LeftBoundary3rdDerivativeEquation() if region == 'bottom' else 
                             RightBoundary3rdDerivativeEquation(), grid=grid),
                converter.to_z(Internal1stDerivativeEquation(), grid=grid),
                converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
            ])
        
        # z面境界 (前後境界)
        for region in ['front', 'back']:
            system.add_equations(region, [
                converter.to_x(Internal1stDerivativeEquation(), grid=grid),
                converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_y(Internal1stDerivativeEquation(), grid=grid),
                converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_z(LeftBoundary1stDerivativeEquation() if region == 'front' else 
                             RightBoundary1stDerivativeEquation(), grid=grid),
                converter.to_z(LeftBoundary2ndDerivativeEquation() if region == 'front' else 
                             RightBoundary2ndDerivativeEquation(), grid=grid),
                converter.to_z(LeftBoundary3rdDerivativeEquation() if region == 'front' else 
                             RightBoundary3rdDerivativeEquation(), grid=grid)
            ])
        
        # エッジ領域 (12本)
        edge_regions = [
            # x軸に沿ったエッジ
            ('bottom_front', 'y', 'bottom', 'z', 'front'),
            ('bottom_back', 'y', 'bottom', 'z', 'back'),
            ('top_front', 'y', 'top', 'z', 'front'),
            ('top_back', 'y', 'top', 'z', 'back'),
            # y軸に沿ったエッジ
            ('left_front', 'x', 'left', 'z', 'front'),
            ('left_back', 'x', 'left', 'z', 'back'),
            ('right_front', 'x', 'right', 'z', 'front'),
            ('right_back', 'x', 'right', 'z', 'back'),
            # z軸に沿ったエッジ
            ('left_bottom', 'x', 'left', 'y', 'bottom'),
            ('left_top', 'x', 'left', 'y', 'top'),
            ('right_bottom', 'x', 'right', 'y', 'bottom'),
            ('right_top', 'x', 'right', 'y', 'top')
        ]
        
        for region, dir1, bound1, dir2, bound2 in edge_regions:
            system.add_equations(region, self._create_edge_equations(converter, grid, dir1, bound1, dir2, bound2))
        
        # 頂点領域 (8個)
        corner_regions = [
            ('left_bottom_front', 'x', 'left', 'y', 'bottom', 'z', 'front'),
            ('left_bottom_back', 'x', 'left', 'y', 'bottom', 'z', 'back'),
            ('left_top_front', 'x', 'left', 'y', 'top', 'z', 'front'),
            ('left_top_back', 'x', 'left', 'y', 'top', 'z', 'back'),
            ('right_bottom_front', 'x', 'right', 'y', 'bottom', 'z', 'front'),
            ('right_bottom_back', 'x', 'right', 'y', 'bottom', 'z', 'back'),
            ('right_top_front', 'x', 'right', 'y', 'top', 'z', 'front'),
            ('right_top_back', 'x', 'right', 'y', 'top', 'z', 'back')
        ]
        
        for region, dir1, bound1, dir2, bound2, dir3, bound3 in corner_regions:
            system.add_equations(region, self._create_corner_equations(converter, grid, 
                                                                    dir1, bound1, 
                                                                    dir2, bound2, 
                                                                    dir3, bound3))
        
        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False
    
    def _create_edge_equations(self, converter, grid, dir1, bound1, dir2, bound2):
        """エッジ領域用の方程式セットを作成"""
        # 左右境界用の1階・2階・3階微分方程式を取得
        def get_bound_deriv(direction, bound_type):
            if direction in ['x', 'y', 'z']:
                if bound_type in ['left', 'bottom', 'front']:
                    return [
                        LeftBoundary1stDerivativeEquation(),
                        LeftBoundary2ndDerivativeEquation(),
                        LeftBoundary3rdDerivativeEquation()
                    ]
                elif bound_type in ['right', 'top', 'back']:
                    return [
                        RightBoundary1stDerivativeEquation(),
                        RightBoundary2ndDerivativeEquation(),
                        RightBoundary3rdDerivativeEquation()
                    ]
            return []
        
        # 方向1の境界方程式
        dir1_eqs = get_bound_deriv(dir1, bound1)
        
        # 方向2の境界方程式
        dir2_eqs = get_bound_deriv(dir2, bound2)
        
        # 方向1の方程式変換
        dir1_converted = []
        for eq in dir1_eqs:
            dir1_converted.append(getattr(converter, f"to_{dir1}")(eq, grid=grid))
        
        # 方向2の方程式変換
        dir2_converted = []
        for eq in dir2_eqs:
            dir2_converted.append(getattr(converter, f"to_{dir2}")(eq, grid=grid))
        
        # 残りの方向
        dirs = ['x', 'y', 'z']
        dirs.remove(dir1)
        dirs.remove(dir2)
        dir3 = dirs[0]
        
        # 方向3の内部方程式
        dir3_converted = [
            getattr(converter, f"to_{dir3}")(Internal1stDerivativeEquation(), grid=grid),
            getattr(converter, f"to_{dir3}")(Internal2ndDerivativeEquation(), grid=grid),
            getattr(converter, f"to_{dir3}")(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        # 全ての方程式を結合
        return dir1_converted + dir2_converted + dir3_converted
    
    def _create_corner_equations(self, converter, grid, dir1, bound1, dir2, bound2, dir3, bound3):
        """頂点領域用の方程式セットを作成"""
        # 左右境界用の1階・2階・3階微分方程式を取得
        def get_bound_deriv(direction, bound_type):
            if direction in ['x', 'y', 'z']:
                if bound_type in ['left', 'bottom', 'front']:
                    return [
                        LeftBoundary1stDerivativeEquation(),
                        LeftBoundary2ndDerivativeEquation(),
                        LeftBoundary3rdDerivativeEquation()
                    ]
                elif bound_type in ['right', 'top', 'back']:
                    return [
                        RightBoundary1stDerivativeEquation(),
                        RightBoundary2ndDerivativeEquation(),
                        RightBoundary3rdDerivativeEquation()
                    ]
            return []
        
        # 各方向の境界方程式
        dir1_eqs = get_bound_deriv(dir1, bound1)
        dir2_eqs = get_bound_deriv(dir2, bound2)
        dir3_eqs = get_bound_deriv(dir3, bound3)
        
        # 各方向の方程式変換
        equations = []
        
        for eq in dir1_eqs:
            equations.append(getattr(converter, f"to_{dir1}")(eq, grid=grid))
        
        for eq in dir2_eqs:
            equations.append(getattr(converter, f"to_{dir2}")(eq, grid=grid))
        
        for eq in dir3_eqs:
            equations.append(getattr(converter, f"to_{dir3}")(eq, grid=grid))
        
        return equations


class PoissonEquationSet3D(EquationSet):
    """ディリクレ・ノイマン混合境界条件の3Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = True
    
    def setup_equations(self, system, grid, test_func=None):
        """
        3次元ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (3D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if not grid.is_3d:
            raise ValueError("3D方程式セットが非3Dグリッドで使用されました")
            
        # 変換器を作成
        converter = Equation1Dto3DConverter

        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation3D(grid=grid))
        
        # 内部点用の方程式
        system.add_equations('interior', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x面境界 (左右境界)
        for region in ['left', 'right']:
            system.add_equations(region, [
                DirichletBoundaryEquation3D(grid=grid),
                NeumannXBoundaryEquation3D(grid=grid),
                converter.to_x(
                    (LeftBoundary1stDerivativeEquation() if region == 'left' else RightBoundary1stDerivativeEquation())+
                    (LeftBoundary2ndDerivativeEquation() if region == 'left' else RightBoundary2ndDerivativeEquation())+ 
                    (LeftBoundary3rdDerivativeEquation() if region == 'left' else RightBoundary3rdDerivativeEquation()), 
                    grid=grid),
                converter.to_y(Internal1stDerivativeEquation(), grid=grid),
                converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_z(Internal1stDerivativeEquation(), grid=grid),
                converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
            ])
        
        # y面境界 (下上境界)
        for region in ['bottom', 'top']:
            system.add_equations(region, [
                converter.to_x(Internal1stDerivativeEquation(), grid=grid),
                converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
                DirichletBoundaryEquation3D(grid=grid),
                NeumannYBoundaryEquation3D(grid=grid),
                converter.to_y(
                    (LeftBoundary1stDerivativeEquation() if region == 'bottom' else RightBoundary1stDerivativeEquation())+
                    (LeftBoundary2ndDerivativeEquation() if region == 'bottom' else RightBoundary2ndDerivativeEquation())+ 
                    (LeftBoundary3rdDerivativeEquation() if region == 'bottom' else RightBoundary3rdDerivativeEquation()), 
                    grid=grid),
                converter.to_z(Internal1stDerivativeEquation(), grid=grid),
                converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
            ])
        
        # z面境界 (前後境界)
        for region in ['front', 'back']:
            system.add_equations(region, [
                converter.to_x(Internal1stDerivativeEquation(), grid=grid),
                converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_y(Internal1stDerivativeEquation(), grid=grid),
                converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
                DirichletBoundaryEquation3D(grid=grid),
                NeumannZBoundaryEquation3D(grid=grid),
                converter.to_z(
                    (LeftBoundary1stDerivativeEquation() if region == 'front' else RightBoundary1stDerivativeEquation())+
                    (LeftBoundary2ndDerivativeEquation() if region == 'front' else RightBoundary2ndDerivativeEquation())+ 
                    (LeftBoundary3rdDerivativeEquation() if region == 'front' else RightBoundary3rdDerivativeEquation()), 
                    grid=grid)
            ])
        
        # エッジ領域のセットアップ
        # 各エッジに適切な方程式を設定
        edge_configs = [
            # x軸エッジ (y/z面の交差)
            ('bottom_front', ['y', 'bottom'], ['z', 'front']),
            ('bottom_back', ['y', 'bottom'], ['z', 'back']),
            ('top_front', ['y', 'top'], ['z', 'front']),
            ('top_back', ['y', 'top'], ['z', 'back']),
            # y軸エッジ (x/z面の交差)
            ('left_front', ['x', 'left'], ['z', 'front']),
            ('left_back', ['x', 'left'], ['z', 'back']),
            ('right_front', ['x', 'right'], ['z', 'front']),
            ('right_back', ['x', 'right'], ['z', 'back']),
            # z軸エッジ (x/y面の交差)
            ('left_bottom', ['x', 'left'], ['y', 'bottom']),
            ('left_top', ['x', 'left'], ['y', 'top']),
            ('right_bottom', ['x', 'right'], ['y', 'bottom']),
            ('right_top', ['x', 'right'], ['y', 'top'])
        ]
        
        for region, dir1_info, dir2_info in edge_configs:
            system.add_equations(region, self._create_edge_equations_poisson(
                grid, converter, dir1_info, dir2_info
            ))
        
        # 頂点領域のセットアップ
        corner_configs = [
            ('left_bottom_front', ['x', 'left'], ['y', 'bottom'], ['z', 'front']),
            ('left_bottom_back', ['x', 'left'], ['y', 'bottom'], ['z', 'back']),
            ('left_top_front', ['x', 'left'], ['y', 'top'], ['z', 'front']),
            ('left_top_back', ['x', 'left'], ['y', 'top'], ['z', 'back']),
            ('right_bottom_front', ['x', 'right'], ['y', 'bottom'], ['z', 'front']),
            ('right_bottom_back', ['x', 'right'], ['y', 'bottom'], ['z', 'back']),
            ('right_top_front', ['x', 'right'], ['y', 'top'], ['z', 'front']),
            ('right_top_back', ['x', 'right'], ['y', 'top'], ['z', 'back'])
        ]
        
        for region, dir1_info, dir2_info, dir3_info in corner_configs:
            system.add_equations(region, self._create_corner_equations_poisson(
                grid, converter, dir1_info, dir2_info, dir3_info
            ))
        
        return True, True
    
    def _create_edge_equations_poisson(self, grid, converter, dir1_info, dir2_info):
        """ポアソン方程式用のエッジ方程式を構築"""
        dir1, bound1 = dir1_info
        dir2, bound2 = dir2_info
        
        # 残りの方向
        remaining_dirs = ['x', 'y', 'z']
        remaining_dirs.remove(dir1)
        remaining_dirs.remove(dir2)
        dir3 = remaining_dirs[0]
        
        # ディリクレット境界条件
        equations = [DirichletBoundaryEquation3D(grid=grid)]
        
        # 各方向のノイマン境界条件
        if dir1 == 'x':
            equations.append(NeumannXBoundaryEquation3D(grid=grid))
        elif dir1 == 'y':
            equations.append(NeumannYBoundaryEquation3D(grid=grid))
        elif dir1 == 'z':
            equations.append(NeumannZBoundaryEquation3D(grid=grid))
            
        if dir2 == 'x':
            equations.append(NeumannXBoundaryEquation3D(grid=grid))
        elif dir2 == 'y':
            equations.append(NeumannYBoundaryEquation3D(grid=grid))
        elif dir2 == 'z':
            equations.append(NeumannZBoundaryEquation3D(grid=grid))
        
        # 方向1の境界微分方程式
        eq1 = None
        if bound1 in ['left', 'bottom', 'front']:
            eq1 = LeftBoundary1stDerivativeEquation() + LeftBoundary2ndDerivativeEquation() + LeftBoundary3rdDerivativeEquation()
        else:
            eq1 = RightBoundary1stDerivativeEquation() + RightBoundary2ndDerivativeEquation() + RightBoundary3rdDerivativeEquation()
        
        equations.append(getattr(converter, f"to_{dir1}")(eq1, grid=grid))
        
        # 方向2の境界微分方程式
        eq2 = None
        if bound2 in ['left', 'bottom', 'front']:
            eq2 = LeftBoundary1stDerivativeEquation() + LeftBoundary2ndDerivativeEquation() + LeftBoundary3rdDerivativeEquation()
        else:
            eq2 = RightBoundary1stDerivativeEquation() + RightBoundary2ndDerivativeEquation() + RightBoundary3rdDerivativeEquation()
        
        equations.append(getattr(converter, f"to_{dir2}")(eq2, grid=grid))
        
        # 方向3の内部微分方程式
        equations.append(getattr(converter, f"to_{dir3}")(Internal1stDerivativeEquation(), grid=grid))
        equations.append(getattr(converter, f"to_{dir3}")(Internal2ndDerivativeEquation(), grid=grid))
        equations.append(getattr(converter, f"to_{dir3}")(Internal3rdDerivativeEquation(), grid=grid))
        
        return equations
    
    def _create_corner_equations_poisson(self, grid, converter, dir1_info, dir2_info, dir3_info):
        """ポアソン方程式用の頂点方程式を構築"""
        dir1, bound1 = dir1_info
        dir2, bound2 = dir2_info
        dir3, bound3 = dir3_info
        
        # ディリクレット境界条件
        equations = [DirichletBoundaryEquation3D(grid=grid)]
        
        # 各方向のノイマン境界条件
        if dir1 == 'x':
            equations.append(NeumannXBoundaryEquation3D(grid=grid))
        elif dir1 == 'y':
            equations.append(NeumannYBoundaryEquation3D(grid=grid))
        elif dir1 == 'z':
            equations.append(NeumannZBoundaryEquation3D(grid=grid))
            
        if dir2 == 'x':
            equations.append(NeumannXBoundaryEquation3D(grid=grid))
        elif dir2 == 'y':
            equations.append(NeumannYBoundaryEquation3D(grid=grid))
        elif dir2 == 'z':
            equations.append(NeumannZBoundaryEquation3D(grid=grid))
            
        if dir3 == 'x':
            equations.append(NeumannXBoundaryEquation3D(grid=grid))
        elif dir3 == 'y':
            equations.append(NeumannYBoundaryEquation3D(grid=grid))
        elif dir3 == 'z':
            equations.append(NeumannZBoundaryEquation3D(grid=grid))
        
        # 各方向の境界微分方程式
        for dir_idx, (dir_name, bound_type) in enumerate([(dir1, bound1), (dir2, bound2), (dir3, bound3)]):
            # 1階・2階・3階微分方程式を分けて適用
            if bound_type in ['left', 'bottom', 'front']:
                eqs = [LeftBoundary1stDerivativeEquation(), 
                      LeftBoundary2ndDerivativeEquation(), 
                      LeftBoundary3rdDerivativeEquation()]
            else:  # ['right', 'top', 'back']
                eqs = [RightBoundary1stDerivativeEquation(), 
                      RightBoundary2ndDerivativeEquation(), 
                      RightBoundary3rdDerivativeEquation()]
            
            # すでに十分な方程式がある場合は、少し減らす
            if dir_idx < 2:
                # 最初の2方向は全ての階数の微分を使用
                for eq in eqs:
                    equations.append(getattr(converter, f"to_{dir_name}")(eq, grid=grid))
            else:
                # 最後の方向は最低限の微分だけ使用（残りはディリクレット、ノイマンで代替）
                equations.append(getattr(converter, f"to_{dir_name}")(eqs[1], grid=grid))  # 2階微分のみ
        
        return equations


class PoissonEquationSet3D2(EquationSet):
    """ディリクレ境界条件のみの3Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = False  # ノイマン境界条件を無効化
    
    def setup_equations(self, system, grid, test_func=None):
        """
        ディリクレ境界条件のみの3次元ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (3D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if not grid.is_3d:
            raise ValueError("3D方程式セットが非3Dグリッドで使用されました")
            
        # 変換器を作成
        converter = Equation1Dto3DConverter

        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation3D(grid=grid))
        
        # 内部点用の方程式
        system.add_equations('interior', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x面境界 (左右境界)
        for region in ['left', 'right']:
            system.add_equations(region, [
                DirichletBoundaryEquation3D(grid=grid),
                converter.to_x(
                    (LeftBoundary1stDerivativeEquation() if region == 'left' else RightBoundary1stDerivativeEquation()),
                    grid=grid),
                converter.to_x(
                    (LeftBoundary2ndDerivativeEquation() if region == 'left' else RightBoundary2ndDerivativeEquation())+ 
                    (LeftBoundary3rdDerivativeEquation() if region == 'left' else RightBoundary3rdDerivativeEquation()), 
                    grid=grid),
                converter.to_y(Internal1stDerivativeEquation(), grid=grid),
                converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_z(Internal1stDerivativeEquation(), grid=grid),
                converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
            ])
        
        # y面境界 (下上境界)
        for region in ['bottom', 'top']:
            system.add_equations(region, [
                converter.to_x(Internal1stDerivativeEquation(), grid=grid),
                converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
                DirichletBoundaryEquation3D(grid=grid),
                converter.to_y(
                    (LeftBoundary1stDerivativeEquation() if region == 'bottom' else RightBoundary1stDerivativeEquation()),
                    grid=grid),
                converter.to_y(
                    (LeftBoundary2ndDerivativeEquation() if region == 'bottom' else RightBoundary2ndDerivativeEquation())+ 
                    (LeftBoundary3rdDerivativeEquation() if region == 'bottom' else RightBoundary3rdDerivativeEquation()), 
                    grid=grid),
                converter.to_z(Internal1stDerivativeEquation(), grid=grid),
                converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
            ])
        
        # z面境界 (前後境界)
        for region in ['front', 'back']:
            system.add_equations(region, [
                converter.to_x(Internal1stDerivativeEquation(), grid=grid),
                converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_y(Internal1stDerivativeEquation(), grid=grid),
                converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
                DirichletBoundaryEquation3D(grid=grid),
                converter.to_z(
                    (LeftBoundary1stDerivativeEquation() if region == 'front' else RightBoundary1stDerivativeEquation()),
                    grid=grid),
                converter.to_z(
                    (LeftBoundary2ndDerivativeEquation() if region == 'front' else RightBoundary2ndDerivativeEquation())+ 
                    (LeftBoundary3rdDerivativeEquation() if region == 'front' else RightBoundary3rdDerivativeEquation()), 
                    grid=grid)
            ])
        
        # エッジ領域のセットアップ
        edge_configs = [
            # x軸エッジ (y/z面の交差)
            ('bottom_front', ['y', 'bottom'], ['z', 'front']),
            ('bottom_back', ['y', 'bottom'], ['z', 'back']),
            ('top_front', ['y', 'top'], ['z', 'front']),
            ('top_back', ['y', 'top'], ['z', 'back']),
            # y軸エッジ (x/z面の交差)
            ('left_front', ['x', 'left'], ['z', 'front']),
            ('left_back', ['x', 'left'], ['z', 'back']),
            ('right_front', ['x', 'right'], ['z', 'front']),
            ('right_back', ['x', 'right'], ['z', 'back']),
            # z軸エッジ (x/y面の交差)
            ('left_bottom', ['x', 'left'], ['y', 'bottom']),
            ('left_top', ['x', 'left'], ['y', 'top']),
            ('right_bottom', ['x', 'right'], ['y', 'bottom']),
            ('right_top', ['x', 'right'], ['y', 'top'])
        ]
        
        for region, dir1_info, dir2_info in edge_configs:
            system.add_equations(region, self._create_edge_equations_poisson2(
                grid, converter, dir1_info, dir2_info
            ))
        
        # 頂点領域のセットアップ
        corner_configs = [
            ('left_bottom_front', ['x', 'left'], ['y', 'bottom'], ['z', 'front']),
            ('left_bottom_back', ['x', 'left'], ['y', 'bottom'], ['z', 'back']),
            ('left_top_front', ['x', 'left'], ['y', 'top'], ['z', 'front']),
            ('left_top_back', ['x', 'left'], ['y', 'top'], ['z', 'back']),
            ('right_bottom_front', ['x', 'right'], ['y', 'bottom'], ['z', 'front']),
            ('right_bottom_back', ['x', 'right'], ['y', 'bottom'], ['z', 'back']),
            ('right_top_front', ['x', 'right'], ['y', 'top'], ['z', 'front']),
            ('right_top_back', ['x', 'right'], ['y', 'top'], ['z', 'back'])
        ]
        
        for region, dir1_info, dir2_info, dir3_info in corner_configs:
            system.add_equations(region, self._create_corner_equations_poisson2(
                grid, converter, dir1_info, dir2_info, dir3_info
            ))
        
        return True, False  # ディリクレト境界条件のみ有効
    
    def _create_edge_equations_poisson2(self, grid, converter, dir1_info, dir2_info):
        """ポアソン方程式用のエッジ方程式を構築 (ディリクレ境界条件のみ)"""
        dir1, bound1 = dir1_info
        dir2, bound2 = dir2_info
        
        # 残りの方向
        remaining_dirs = ['x', 'y', 'z']
        remaining_dirs.remove(dir1)
        remaining_dirs.remove(dir2)
        dir3 = remaining_dirs[0]
        
        # ディリクレット境界条件
        equations = [DirichletBoundaryEquation3D(grid=grid)]
        
        # 方向1の境界微分方程式
        eq1_1st = None
        eq1_rest = None
        if bound1 in ['left', 'bottom', 'front']:
            eq1_1st = LeftBoundary1stDerivativeEquation()
            eq1_rest = LeftBoundary2ndDerivativeEquation() + LeftBoundary3rdDerivativeEquation()
        else:
            eq1_1st = RightBoundary1stDerivativeEquation()
            eq1_rest = RightBoundary2ndDerivativeEquation() + RightBoundary3rdDerivativeEquation()
        
        equations.append(getattr(converter, f"to_{dir1}")(eq1_1st, grid=grid))
        equations.append(getattr(converter, f"to_{dir1}")(eq1_rest, grid=grid))
        
        # 方向2の境界微分方程式
        eq2_1st = None
        eq2_rest = None
        if bound2 in ['left', 'bottom', 'front']:
            eq2_1st = LeftBoundary1stDerivativeEquation()
            eq2_rest = LeftBoundary2ndDerivativeEquation() + LeftBoundary3rdDerivativeEquation()
        else:
            eq2_1st = RightBoundary1stDerivativeEquation()
            eq2_rest = RightBoundary2ndDerivativeEquation() + RightBoundary3rdDerivativeEquation()
        
        equations.append(getattr(converter, f"to_{dir2}")(eq2_1st, grid=grid))
        equations.append(getattr(converter, f"to_{dir2}")(eq2_rest, grid=grid))
        
        # 方向3の内部微分方程式
        equations.append(getattr(converter, f"to_{dir3}")(Internal1stDerivativeEquation(), grid=grid))
        equations.append(getattr(converter, f"to_{dir3}")(Internal2ndDerivativeEquation(), grid=grid))
        equations.append(getattr(converter, f"to_{dir3}")(Internal3rdDerivativeEquation(), grid=grid))
        
        return equations
    
    def _create_corner_equations_poisson2(self, grid, converter, dir1_info, dir2_info, dir3_info):
        """ポアソン方程式用の頂点方程式を構築 (ディリクレ境界条件のみ)"""
        dir1, bound1 = dir1_info
        dir2, bound2 = dir2_info
        dir3, bound3 = dir3_info
        
        # ディリクレット境界条件
        equations = [DirichletBoundaryEquation3D(grid=grid)]
        
        # 各方向の境界微分方程式を追加
        directions = [(dir1, bound1), (dir2, bound2), (dir3, bound3)]
        
        for dir_name, bound_type in directions:
            if bound_type in ['left', 'bottom', 'front']:
                eq1st = LeftBoundary1stDerivativeEquation()
                eq2nd = LeftBoundary2ndDerivativeEquation()
                eq3rd = LeftBoundary3rdDerivativeEquation()
            else:  # ['right', 'top', 'back']
                eq1st = RightBoundary1stDerivativeEquation()
                eq2nd = RightBoundary2ndDerivativeEquation()
                eq3rd = RightBoundary3rdDerivativeEquation()
            
            # 各方向の微分方程式を個別に追加
            equations.append(getattr(converter, f"to_{dir_name}")(eq1st, grid=grid))
            equations.append(getattr(converter, f"to_{dir_name}")(eq2nd, grid=grid))
            equations.append(getattr(converter, f"to_{dir_name}")(eq3rd, grid=grid))
        
        return equations