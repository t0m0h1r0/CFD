"""
1次元方程式セットの定義を行うモジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
1次元の方程式セットを定義します。
"""

from core.base.base_equation_set import EquationSet

# 共通の方程式をインポート
from equation.poisson import PoissonEquation
from equation.original import OriginalEquation
from equation.boundary import (
    DirichletBoundaryEquation, NeumannBoundaryEquation
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
