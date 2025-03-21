"""
方程式システムの基底クラスを定義するモジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
格子点の方程式をシステム全体として管理するための基底クラスと共通機能を提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseEquationSystem(ABC):
    """方程式システムの抽象基底クラス"""

    def __init__(self, grid):
        """
        方程式システムを初期化
        
        Args:
            grid: 計算格子オブジェクト (1Dまたは2D)
        """
        self.grid = grid
        self.is_2d = grid.is_2d
        
        # 領域ごとの方程式コレクション（サブクラスで初期化）
        self.equations = {}
        
        # 次元に応じた初期化
        self._initialize_equations()
    
    @abstractmethod
    def _initialize_equations(self):
        """
        次元に応じた方程式コレクションを初期化
        サブクラスで実装する
        """
        pass
    
    def add_equation(self, region: str, equation) -> None:
        """
        指定された領域に方程式を追加
        
        Args:
            region: 領域識別子 ('interior', 'left', 'right', 等)
            equation: 追加する方程式オブジェクト
        """
        # 方程式にグリッドを設定（必要な場合）
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        # 指定された領域に方程式を追加
        if region in self.equations:
            self.equations[region].append(equation)
        else:
            raise ValueError(f"未知の領域: {region}")
    
    def add_equations(self, region: str, equations: List) -> None:
        """
        指定された領域に複数の方程式を一度に追加
        
        Args:
            region: 領域識別子
            equations: 追加する方程式のリスト
        """
        for equation in equations:
            self.add_equation(region, equation)
    
    def add_dominant_equation(self, equation) -> None:
        """
        支配方程式をすべての領域に追加
        
        Args:
            equation: 支配方程式
        """
        for region in self.equations:
            self.add_equation(region, equation)
    
    @abstractmethod
    def build_matrix_system(self):
        """
        行列システムを構築
        
        Returns:
            システム行列 (CSR形式、CPU)
        """
        pass
    
    def _validate_equations(self) -> None:
        """方程式セットの基本的な検証"""
        for region, eqs in self.equations.items():
            if not eqs:
                raise ValueError(f"{region}領域に方程式が設定されていません")
    
    @abstractmethod
    def _get_point_location(self, i: int, j: int = None) -> str:
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス (2Dのみ)
            
        Returns:
            位置を表す文字列 ('interior', 'left', 等)
        """
        pass
    
    def _identify_equation_type(self, equation, i: int, j: int = None) -> str:
        """
        方程式の種類を識別
        
        Args:
            equation: 対象の方程式
            i: x方向のインデックス
            j: y方向のインデックス (2Dのみ)
            
        Returns:
            方程式の種類 ('governing', 'dirichlet', 等) または None (無効な場合)
        """
        # 方程式が有効かチェック
        if self.is_2d:
            if not equation.is_valid_at(i, j):
                return None
        else:
            if not equation.is_valid_at(i):
                return None
        
        # クラス名から方程式種別を判定 (importはローカルに行う)
        from equation.poisson import PoissonEquation, PoissonEquation2D
        from equation.original import OriginalEquation, OriginalEquation2D
        from equation.boundary import (
            DirichletBoundaryEquation, NeumannBoundaryEquation,
            DirichletBoundaryEquation2D
        )
        
        # 1D/2D共通の方程式タイプ
        if isinstance(equation, (PoissonEquation, PoissonEquation2D, OriginalEquation, OriginalEquation2D)):
            return "governing"
        elif isinstance(equation, (DirichletBoundaryEquation, DirichletBoundaryEquation2D)):
            return "dirichlet"
        elif isinstance(equation, NeumannBoundaryEquation):
            return "neumann"
        
        # 2D固有のタイプまたは方向性のある方程式
        if self.is_2d:
            # DirectionalEquation2Dで作られた方向性のある方程式を識別
            from equation.equation_converter import DirectionalEquation2D
            if isinstance(equation, DirectionalEquation2D):
                # 内部の1D方程式がNeumannBoundaryEquationの場合
                if hasattr(equation, 'equation_1d') and isinstance(equation.equation_1d, NeumannBoundaryEquation):
                    # 方向に基づいて適切なノイマンタイプを返す
                    if equation.direction == 'x':
                        return "neumann_x"
                    elif equation.direction == 'y':
                        return "neumann_y"
        
        # それ以外は補助方程式
        return "auxiliary"
    
    def _to_numpy(self, value):
        """
        CuPy配列をNumPy配列に変換する (必要な場合のみ)
        
        Args:
            value: 変換する値
            
        Returns:
            NumPy配列またはスカラー
        """
        if hasattr(value, 'get'):
            return value.get()
        return value
        
    def analyze_sparsity(self) -> Dict[str, Any]:
        """
        行列の疎性を分析
        
        Returns:
            疎性分析結果の辞書
        """
        A = self.build_matrix_system()
        
        total_size = A.shape[0]
        nnz = A.nnz
        max_possible_nnz = total_size * total_size
        sparsity = 1.0 - (nnz / max_possible_nnz)
        
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8 bytes per double
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 8 bytes for value + 4 bytes for indices
        
        return {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }