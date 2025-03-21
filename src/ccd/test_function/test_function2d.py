"""
2次元テスト関数モジュール

このモジュールはCCD法で使用する2次元テスト関数のクラスと機能を提供します。
sympyを使用した自動微分機能付きのテスト関数を定義します。
"""

import sympy as sp
from typing import List, Optional

from core.base.base_test_function import BaseTestFunction
from test_function.test_function1d import TestFunction1D

class TestFunction2D(BaseTestFunction):
    """2次元テスト関数クラス（自動微分機能付き）"""
    
    def __init__(self, name: str, expr):
        """
        2Dテスト関数の初期化
        
        Args:
            name: 関数名
            expr: シンボリック表現 (x, yの関数)
        """
        super().__init__(name)
        
        # シンボリック変数の定義
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        
        # シンボリック表現の保存
        self.expr = expr
        
        # 導関数の計算
        self._calculate_derivatives()
        
        # 呼び出し可能関数の作成
        self._create_callable_functions()
    
    def _calculate_derivatives(self):
        """sympyを使用して2D偏導関数を計算"""
        # 一階偏導関数
        self.df_dx_expr = sp.diff(self.expr, self.x)
        self.df_dy_expr = sp.diff(self.expr, self.y)
        
        # 二階偏導関数
        self.d2f_dx2_expr = sp.diff(self.df_dx_expr, self.x)
        self.d2f_dy2_expr = sp.diff(self.df_dy_expr, self.y)
        self.d2f_dxdy_expr = sp.diff(self.df_dx_expr, self.y)
        
        # 三階偏導関数
        self.d3f_dx3_expr = sp.diff(self.d2f_dx2_expr, self.x)
        self.d3f_dy3_expr = sp.diff(self.d2f_dy2_expr, self.y)
        self.d3f_dx2dy_expr = sp.diff(self.d2f_dx2_expr, self.y)
        self.d3f_dxdy2_expr = sp.diff(self.d2f_dxdy_expr, self.y)
    
    def _create_callable_functions(self):
        """シンボリック表現から呼び出し可能関数を作成"""
        # cupyモジュールが問題を起こす可能性があるので、numpy も指定しておく
        modules = ["cupy", "numpy"]
        
        # 関数
        self.f_func = sp.lambdify((self.x, self.y), self.expr, modules=modules)
        
        # 一階偏導関数
        self.df_dx_func = sp.lambdify((self.x, self.y), self.df_dx_expr, modules=modules)
        self.df_dy_func = sp.lambdify((self.x, self.y), self.df_dy_expr, modules=modules)
        
        # 二階偏導関数
        self.d2f_dx2_func = sp.lambdify((self.x, self.y), self.d2f_dx2_expr, modules=modules)
        self.d2f_dy2_func = sp.lambdify((self.x, self.y), self.d2f_dy2_expr, modules=modules)
        self.d2f_dxdy_func = sp.lambdify((self.x, self.y), self.d2f_dxdy_expr, modules=modules)
        
        # 三階偏導関数
        self.d3f_dx3_func = sp.lambdify((self.x, self.y), self.d3f_dx3_expr, modules=modules)
        self.d3f_dy3_func = sp.lambdify((self.x, self.y), self.d3f_dy3_expr, modules=modules)
        self.d3f_dx2dy_func = sp.lambdify((self.x, self.y), self.d3f_dx2dy_expr, modules=modules)
        self.d3f_dxdy2_func = sp.lambdify((self.x, self.y), self.d3f_dxdy2_expr, modules=modules)
    
    def f(self, x, y) -> float:
        """
        関数値を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            関数値 f(x,y)
        """
        return float(self.f_func(x, y))
    
    def df_dx(self, x, y) -> float:
        """
        x方向の1階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            x方向偏導関数値 ∂f/∂x
        """
        return float(self.df_dx_func(x, y))
    
    def df_dy(self, x, y) -> float:
        """
        y方向の1階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            y方向偏導関数値 ∂f/∂y
        """
        return float(self.df_dy_func(x, y))
    
    def d2f_dx2(self, x, y) -> float:
        """
        x方向の2階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            x方向2階偏導関数値 ∂²f/∂x²
        """
        return float(self.d2f_dx2_func(x, y))
    
    def d2f_dy2(self, x, y) -> float:
        """
        y方向の2階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            y方向2階偏導関数値 ∂²f/∂y²
        """
        return float(self.d2f_dy2_func(x, y))
    
    def d2f_dxdy(self, x, y) -> float:
        """
        混合2階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            混合2階偏導関数値 ∂²f/∂x∂y
        """
        return float(self.d2f_dxdy_func(x, y))
    
    def d3f_dx3(self, x, y) -> float:
        """
        x方向の3階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            x方向3階偏導関数値 ∂³f/∂x³
        """
        return float(self.d3f_dx3_func(x, y))
    
    def d3f_dy3(self, x, y) -> float:
        """
        y方向の3階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            y方向3階偏導関数値 ∂³f/∂y³
        """
        return float(self.d3f_dy3_func(x, y))
    
    def d3f_dx2dy(self, x, y) -> float:
        """
        混合3階偏導関数(x²y)を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            混合3階偏導関数値 ∂³f/∂x²∂y
        """
        return float(self.d3f_dx2dy_func(x, y))
    
    def d3f_dxdy2(self, x, y) -> float:
        """
        混合3階偏導関数(xy²)を評価
        
        Args:
            x: x座標
            y: y座標
            
        Returns:
            混合3階偏導関数値 ∂³f/∂x∂y²
        """
        return float(self.d3f_dxdy2_func(x, y))
    
    def get_dimension(self) -> int:
        """
        関数の次元を取得
        
        Returns:
            2 (2次元関数)
        """
        return 2
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"2D TestFunction: {self.name}, f(x,y) = {self.expr}"
    
    def __repr__(self) -> str:
        """オブジェクト表現"""
        return f"TestFunction2D('{self.name}', {self.expr})"
    
    @classmethod
    def from_1d(cls, func_1d: TestFunction1D, method='product', suffix='') -> 'TestFunction2D':
        """
        1Dテスト関数から2Dテスト関数を作成
        
        Args:
            func_1d: 1Dテスト関数
            method: 拡張メソッド ('product', 'sum', 'radial')
            suffix: 名前に追加する接尾辞
            
        Returns:
            2Dテスト関数
        """
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        
        if method == 'product':
            # f(x,y) = f(x) * f(y)
            name = f"{func_1d.name}Product{suffix}"
            expr_2d = func_1d.expr.subs(x, x) * func_1d.expr.subs(x, y)
        elif method == 'sum':
            # f(x,y) = f(x) + f(y)
            name = f"{func_1d.name}Sum{suffix}"
            expr_2d = func_1d.expr.subs(x, x) + func_1d.expr.subs(x, y)
        elif method == 'radial':
            # f(x,y) = f(sqrt(x^2 + y^2))
            name = f"{func_1d.name}Radial{suffix}"
            r = sp.sqrt(x**2 + y**2)
            expr_2d = func_1d.expr.subs(x, r)
        else:
            raise ValueError(f"未知の拡張メソッド: {method}")
        
        return cls(name, expr_2d)


class TestFunction2DFactory:
    """2Dテスト関数生成ファクトリークラス"""
    
    @staticmethod
    def create_basic_functions() -> List[TestFunction2D]:
        """基本的な2Dテスト関数を作成"""
        x, y = sp.symbols('x y')
        pi = sp.pi
        
        functions = [
            TestFunction2D("Sine2D", sp.sin(pi * x) * sp.sin(pi * y)),
            TestFunction2D("Exp2D", sp.exp(-(x**2 + y**2))),
            TestFunction2D("Poly2D", (1 - x**2) * (1 - y**2)),
        ]
        
        return functions
    
    @staticmethod
    def create_from_1d(func_1d_names: Optional[List[str]] = None) -> List[TestFunction2D]:
        """
        1D関数から拡張された2Dテスト関数を作成
        
        Args:
            func_1d_names: 拡張する1D関数名のリスト（Noneの場合は代表的な関数を使用）
            
        Returns:
            拡張された2Dテスト関数のリスト
        """
        from test_function.test_function1d import TestFunction1DFactory
        
        # 1Dテスト関数を取得
        functions_1d = TestFunction1DFactory.create_standard_functions()
        
        functions_2d = []
        
        # 代表的な関数だけを選択（指定がない場合）
        if func_1d_names is None:
            func_1d_names = ["Sine", "Cosine", "ExpMod", "Runge"]
            
        selected_1d_funcs = [func for func in functions_1d if func.name in func_1d_names]
        
        for func in selected_1d_funcs:
            # テンソル積拡張
            functions_2d.append(TestFunction2D.from_1d(func, method='product'))
            
            # 和による拡張
            functions_2d.append(TestFunction2D.from_1d(func, method='sum'))
            
            # 半径方向拡張 (特定の関数のみ)
            if func.name in ["Sine", "ExpMod"]:
                functions_2d.append(TestFunction2D.from_1d(func, method='radial'))
        
        return functions_2d
    
    @staticmethod
    def create_standard_functions() -> List[TestFunction2D]:
        """標準的な2Dテスト関数セット（基本 + 拡張）を作成"""
        basic_funcs = TestFunction2DFactory.create_basic_functions()
        extended_funcs = TestFunction2DFactory.create_from_1d()
        
        all_funcs = basic_funcs + extended_funcs
        
        return all_funcs
    
    @staticmethod
    def get_function_by_name(name: str) -> Optional[TestFunction2D]:
        """
        名前から関数を取得
        
        Args:
            name: 関数名
            
        Returns:
            対応するTestFunction2D、見つからない場合はNone
        """
        functions = TestFunction2DFactory.create_standard_functions()
        return next((f for f in functions if f.name == name), None)
