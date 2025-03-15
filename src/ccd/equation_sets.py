"""
方程式セットの定義と管理を行うモジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
様々な種類の方程式セットを定義し、次元に応じて適切に管理します。
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

# ヘルパー関数をインポート
from equation_helpers import (
    setup_region_equations,
    setup_derivative_equations_1d,
    setup_derivative_equations_2d,
    setup_poisson_equations_1d,
    setup_poisson_equations_2d
)


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
            "poisson_dn": {"1d": PoissonEquationSet1D_DN, "2d": PoissonEquationSet2D_DN},
            "poisson_d": {"1d": PoissonEquationSet1D_D, "2d": PoissonEquationSet2D_D},
            "poisson_n": {"1d": PoissonEquationSet1D_N, "2d": PoissonEquationSet2D_N},
            "poisson_none": {"1d": PoissonEquationSet1D_NONE, "2d": PoissonEquationSet2D_NONE},
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

class PoissonEquationSet1D(EquationSet):
    """1次元ポアソン方程式のための統一された方程式セット"""

    def __init__(self, boundary_type='DN', merge_derivatives=False):
        """
        初期化
        
        Args:
            boundary_type: 境界条件のタイプ
                'DN' - ディリクレとノイマン両方
                'D'  - ディリクレのみ
                'N'  - ノイマンのみ
                'none' - 境界条件なし
            merge_derivatives: 高階微分方程式を結合するか
        """
        super().__init__()
        self.boundary_type = boundary_type
        self.merge_derivatives = merge_derivatives
        
        # 境界条件の設定
        if boundary_type == 'DN':
            self.use_dirichlet = True
            self.use_neumann = True
        elif boundary_type == 'D':
            self.use_dirichlet = True
            self.use_neumann = False
        elif boundary_type == 'N':
            self.use_dirichlet = False
            self.use_neumann = True
        else:  # 'none'
            self.use_dirichlet = False
            self.use_neumann = False
    
    def setup_equations(self, system, grid, test_func=None):
        """
        1次元ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (1D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if grid.is_2d:
            raise ValueError("1D方程式セットが2Dグリッドで使用されました")
        
        # 統一されたヘルパー関数を使用して方程式を設定
        setup_poisson_equations_1d(
            system, grid, 
            use_dirichlet=self.enable_dirichlet, 
            use_neumann=self.enable_neumann,
            merge_derivatives=self.merge_derivatives
        )
        
        return self.enable_dirichlet, self.enable_neumann


class DerivativeEquationSet1D(EquationSet):
    """1次元高階微分のための統一された方程式セット"""

    def __init__(self, merge_derivatives=False):
        """
        初期化
        
        Args:
            merge_derivatives: 高階微分方程式を結合するか
        """
        super().__init__()
        self.merge_derivatives = merge_derivatives
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
        
        # 統一されたヘルパー関数を使用して方程式を設定
        setup_derivative_equations_1d(
            system, grid, 
            merge_derivatives=self.merge_derivatives
        )
        
        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False


# 特定の境界条件の組み合わせ用のクラス (1D)
class PoissonEquationSet1D_DN(PoissonEquationSet1D):
    """ディリクレ・ノイマン混合境界条件の1Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('DN', False)

class PoissonEquationSet1D_D(PoissonEquationSet1D):
    """ディリクレ境界条件のみの1Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('D', True)

class PoissonEquationSet1D_N(PoissonEquationSet1D):
    """ノイマン境界条件のみの1Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('N', True)

class PoissonEquationSet1D_NONE(PoissonEquationSet1D):
    """境界条件なしの1Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('none', False)


# ===== 2D 方程式セット =====

class PoissonEquationSet2D(EquationSet):
    """2次元ポアソン方程式のための統一された方程式セット"""

    def __init__(self, boundary_type='DN', merge_derivatives=False):
        """
        初期化
        
        Args:
            boundary_type: 境界条件のタイプ
                'DN' - ディリクレとノイマン両方
                'D'  - ディリクレのみ
                'N'  - ノイマンのみ
                'none' - 境界条件なし
            merge_derivatives: 高階微分方程式を結合するか
        """
        super().__init__()
        self.boundary_type = boundary_type
        self.merge_derivatives = merge_derivatives
        
        # 境界条件の設定
        if boundary_type == 'DN':
            self.enable_dirichlet = True
            self.enable_neumann = True
        elif boundary_type == 'D':
            self.enable_dirichlet = True
            self.enable_neumann = False
        elif boundary_type == 'N':
            self.enable_dirichlet = False
            self.enable_neumann = True
        else:  # 'none'
            self.enable_dirichlet = False
            self.enable_neumann = False
    
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
        
        # 統一されたヘルパー関数を使用して方程式を設定
        setup_poisson_equations_2d(
            system, grid, 
            use_dirichlet=self.enable_dirichlet, 
            use_neumann=self.enable_neumann,
            merge_derivatives=self.merge_derivatives
        )
        
        return self.enable_dirichlet, self.enable_neumann


class DerivativeEquationSet2D(EquationSet):
    """2次元高階微分のための統一された方程式セット"""

    def __init__(self, merge_derivatives=False):
        """
        初期化
        
        Args:
            merge_derivatives: 高階微分方程式を結合するか
        """
        super().__init__()
        self.merge_derivatives = merge_derivatives
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
        
        # 統一されたヘルパー関数を使用して方程式を設定
        setup_derivative_equations_2d(
            system, grid, 
            merge_derivatives=self.merge_derivatives
        )
        
        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False


# 特定の境界条件の組み合わせ用のクラス (2D)
class PoissonEquationSet2D_DN(PoissonEquationSet2D):
    """ディリクレ・ノイマン混合境界条件の2Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('DN', False)

class PoissonEquationSet2D_D(PoissonEquationSet2D):
    """ディリクレ境界条件のみの2Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('D', True)

class PoissonEquationSet2D_N(PoissonEquationSet2D):
    """ノイマン境界条件のみの2Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('N', True)

class PoissonEquationSet2D_NONE(PoissonEquationSet2D):
    """境界条件なしの2Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__('none', False)


# ユーティリティ関数
def create_poisson_equation_set(boundary_type='DN', merge_derivatives=False, dimension=None):
    """
    ポアソン方程式セットを作成するファクトリーメソッド
    
    Args:
        boundary_type: 境界条件タイプ ('DN', 'D', 'N', 'none')
        merge_derivatives: 高階微分方程式を結合するか
        dimension: 次元 (1または2、Noneの場合はGridの次元から判断)
    
    Returns:
        PoissonEquationSet インスタンス
    """
    if boundary_type not in ['DN', 'D', 'N', 'none']:
        print(f"警告: 不明な境界条件タイプ '{boundary_type}' です。'DN'を使用します。")
        boundary_type = 'DN'
    
    if dimension == 1:
        return PoissonEquationSet1D(boundary_type, merge_derivatives)
    elif dimension == 2:
        return PoissonEquationSet2D(boundary_type, merge_derivatives)
    else:
        # 次元が指定されていない場合はグリッドの次元から判断できるようにラッパーを返す
        return DimensionalEquationSetWrapper(
            "poisson",
            {
                "1d": lambda: PoissonEquationSet1D(boundary_type, merge_derivatives),
                "2d": lambda: PoissonEquationSet2D(boundary_type, merge_derivatives)
            }
        )

def create_derivative_equation_set(merge_derivatives=False, dimension=None):
    """
    導関数方程式セットを作成するファクトリーメソッド
    
    Args:
        merge_derivatives: 高階微分方程式を結合するか
        dimension: 次元 (1または2、Noneの場合はGridの次元から判断)
    
    Returns:
        DerivativeEquationSet インスタンス
    """
    if dimension == 1:
        return DerivativeEquationSet1D(merge_derivatives)
    elif dimension == 2:
        return DerivativeEquationSet2D(merge_derivatives)
    else:
        # 次元が指定されていない場合はグリッドの次元から判断できるようにラッパーを返す
        return DimensionalEquationSetWrapper(
            "derivative",
            {
                "1d": lambda: DerivativeEquationSet1D(merge_derivatives),
                "2d": lambda: DerivativeEquationSet2D(merge_derivatives)
            }
        )