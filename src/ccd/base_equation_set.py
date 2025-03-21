"""
方程式セットの基底クラス定義モジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
方程式セットの基底クラスとディメンションに応じたラッパークラスを定義します。
"""

from abc import ABC, abstractmethod

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
        # 注: 実際の実装はequation_sets.pyでオーバーライドされます
        # ここではプレースホルダーのみを提供
        all_sets = {
            "poisson": {"1d": None, "2d": None},
            "poisson2": {"1d": None, "2d": None},
            "derivative": {"1d": None, "2d": None},
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
