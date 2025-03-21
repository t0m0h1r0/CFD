"""
方程式システムの定義と管理を行うモジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
格子点の方程式をシステム全体として効率的に管理し、最終的に線形方程式系の
行列を構築するための機能を提供します。
後方互換性のために、リファクタリングされたクラスからのインポートを提供します。
"""

from base_equation_system import BaseEquationSystem
from equation_system1d import EquationSystem1D
from equation_system2d import EquationSystem2D

class EquationSystem:
    """1D/2D両対応の方程式システムを管理するクラス（後方互換性用）"""

    def __init__(self, grid):
        """
        方程式システムを初期化
        
        Args:
            grid: Grid (1D/2D) オブジェクト
        """
        self.grid = grid
        self.is_2d = grid.is_2d
        
        # 次元に応じた適切なシステムを生成
        if self.is_2d:
            self._system = EquationSystem2D(grid)
        else:
            self._system = EquationSystem1D(grid)
            
        # 内部システムへの参照を保持
        self.equations = self._system.equations
    
    def add_equation(self, region: str, equation) -> None:
        """
        指定された領域に方程式を追加
        
        Args:
            region: 領域識別子 ('interior', 'left', 'right', 等)
            equation: 追加する方程式オブジェクト
        """
        self._system.add_equation(region, equation)
    
    def add_equations(self, region: str, equations: list) -> None:
        """
        指定された領域に複数の方程式を一度に追加
        
        Args:
            region: 領域識別子
            equations: 追加する方程式のリスト
        """
        self._system.add_equations(region, equations)
    
    def add_dominant_equation(self, equation) -> None:
        """
        支配方程式をすべての領域に追加
        
        Args:
            equation: 支配方程式
        """
        self._system.add_dominant_equation(equation)
    
    def build_matrix_system(self):
        """
        行列システムを構築
        
        Returns:
            システム行列 (CSR形式、CPU)
        """
        return self._system.build_matrix_system()
    
    def _get_point_location(self, i: int, j: int = None) -> str:
        """
        格子点の位置タイプを判定（互換性用）
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス (2Dのみ)
            
        Returns:
            位置を表す文字列 ('interior', 'left', 等)
        """
        return self._system._get_point_location(i, j)
    
    def _identify_equation_type(self, equation, i: int, j: int = None) -> str:
        """
        方程式の種類を識別（互換性用）
        
        Args:
            equation: 対象の方程式
            i: x方向のインデックス
            j: y方向のインデックス (2Dのみ)
            
        Returns:
            方程式の種類 ('governing', 'dirichlet', 等) または None (無効な場合)
        """
        return self._system._identify_equation_type(equation, i, j)
    
    def _assign_equations_2d(self, eq_by_type, i: int, j: int) -> list:
        """
        2D格子点における方程式の割り当てを決定（互換性用）
        
        Args:
            eq_by_type: 種類別に分類された方程式
            i: x方向のインデックス
            j: y方向のインデックス
            
        Returns:
            各行に割り当てる方程式のリスト [0行目, 1行目, ..., 6行目]
        """
        if self.is_2d:
            return self._system._assign_equations_2d(eq_by_type, i, j)
        raise NotImplementedError("1D格子では _assign_equations_2d は使用できません")
    
    def analyze_sparsity(self):
        """
        行列の疎性を分析
        
        Returns:
            疎性分析結果の辞書
        """
        return self._system.analyze_sparsity()

# 後方互換性のためにエクスポート
__all__ = [
    'BaseEquationSystem',
    'EquationSystem1D',
    'EquationSystem2D',
    'EquationSystem'
]