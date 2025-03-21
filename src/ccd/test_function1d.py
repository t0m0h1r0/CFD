"""
1次元テスト関数モジュール

このモジュールはCCD法で使用する1次元テスト関数のクラスと機能を提供します。
sympyを使用した自動微分機能付きのテスト関数を定義します。
"""

import sympy as sp
import numpy as np
from typing import List, Dict, Tuple, Any, Callable, Optional, Union

from base_test_function import BaseTestFunction

class TestFunction1D(BaseTestFunction):
    """1次元テスト関数クラス（自動微分機能付き）"""
    
    def __init__(self, name: str, expr):
        """
        1Dテスト関数の初期化
        
        Args:
            name: 関数名
            expr: シンボリック表現 (sympy式)
        """
        super().__init__(name)
        
        # シンボリック変数の定義
        self.x = sp.Symbol('x')
        
        # シンボリック表現の保存
        self.expr = expr
        
        # 導関数の計算
        self._calculate_derivatives()
        
        # 呼び出し可能関数の作成
        self._create_callable_functions()
    
    def _calculate_derivatives(self):
        """sympyを使用して1D導関数を計算"""
        self.df_expr = sp.diff(self.expr, self.x)
        self.d2f_expr = sp.diff(self.df_expr, self.x)
        self.d3f_expr = sp.diff(self.d2f_expr, self.x)
    
    def _create_callable_functions(self):
        """シンボリック表現から呼び出し可能関数を作成"""
        # cupyモジュールが問題を起こす可能性があるので、numpy も指定しておく
        modules = ["cupy", "numpy"]
        
        # 関数
        self.f_func = sp.lambdify(self.x, self.expr, modules=modules)
        self.df_func = sp.lambdify(self.x, self.df_expr, modules=modules)
        self.d2f_func = sp.lambdify(self.x, self.d2f_expr, modules=modules)
        self.d3f_func = sp.lambdify(self.x, self.d3f_expr, modules=modules)
    
    def f(self, x) -> float:
        """
        関数値を評価
        
        Args:
            x: x座標
            
        Returns:
            関数値 f(x)
        """
        return float(self.f_func(x))
    
    def df(self, x) -> float:
        """
        1階導関数を評価
        
        Args:
            x: x座標
            
        Returns:
            1階導関数値 f'(x)
        """
        return float(self.df_func(x))
    
    def d2f(self, x) -> float:
        """
        2階導関数を評価
        
        Args:
            x: x座標
            
        Returns:
            2階導関数値 f''(x)
        """
        return float(self.d2f_func(x))
    
    def d3f(self, x) -> float:
        """
        3階導関数を評価
        
        Args:
            x: x座標
            
        Returns:
            3階導関数値 f'''(x)
        """
        return float(self.d3f_func(x))
    
    def get_dimension(self) -> int:
        """
        関数の次元を取得
        
        Returns:
            1 (1次元関数)
        """
        return 1
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"1D TestFunction: {self.name}, f(x) = {self.expr}"
    
    def __repr__(self) -> str:
        """オブジェクト表現"""
        return f"TestFunction1D('{self.name}', {self.expr})"


class TestFunction1DFactory:
    """1Dテスト関数生成ファクトリークラス"""
    
    @staticmethod
    def create_standard_functions() -> List[TestFunction1D]:
        """標準的な1Dテスト関数を作成"""
        x = sp.Symbol('x')
        pi = sp.pi
        
        functions = [
            TestFunction1D("Zero", 0),
            TestFunction1D("QuadPoly", 1 - x**2),
            TestFunction1D("CubicPoly", (1 - x) * (1 + x) * (x + 0.5)),
            TestFunction1D("Sine", sp.sin(pi * x)),
            TestFunction1D("Cosine", sp.cos(2 * pi * x)),
            TestFunction1D("ExpMod", sp.exp(-(x**2)) - sp.exp(-1)),
            TestFunction1D("HigherPoly", x**4 - x**2),
            TestFunction1D("CompoundPoly", x**2 * (1 - x**2)),
            TestFunction1D("Runge", 1 / (1 + 25 * x**2)),
            # ModifiedExp は絶対値関数を含み微分不可能なので除外
            # 代わりに同様の形状を持つ滑らかな関数を追加
            TestFunction1D("SmoothExp", sp.exp(-10 * x**2)),
            TestFunction1D("HighFreqSine", sp.sin(20 * pi * x)),
        ]
        
        return functions
    
    @staticmethod
    def get_function_by_name(name: str) -> Optional[TestFunction1D]:
        """
        名前から関数を取得
        
        Args:
            name: 関数名
            
        Returns:
            対応するTestFunction1D、見つからない場合はNone
        """
        functions = TestFunction1DFactory.create_standard_functions()
        return next((f for f in functions if f.name == name), None)
