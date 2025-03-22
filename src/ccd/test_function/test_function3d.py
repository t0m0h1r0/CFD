"""
3次元テスト関数モジュール

このモジュールはCCD法で使用する3次元テスト関数のクラスと機能を提供します。
sympyを使用した自動微分機能付きのテスト関数を定義します。
"""

import sympy as sp
from typing import List, Optional

from core.base.base_test_function import BaseTestFunction
from test_function.test_function2d import TestFunction2D
from test_function.test_function1d import TestFunction1D

class TestFunction3D(BaseTestFunction):
    """3次元テスト関数クラス（自動微分機能付き）"""
    
    def __init__(self, name: str, expr):
        """
        3Dテスト関数の初期化
        
        Args:
            name: 関数名
            expr: シンボリック表現 (x, y, zの関数)
        """
        super().__init__(name)
        
        # シンボリック変数の定義
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.z = sp.Symbol('z')
        
        # シンボリック表現の保存
        self.expr = expr
        
        # 導関数の計算
        self._calculate_derivatives()
        
        # 呼び出し可能関数の作成
        self._create_callable_functions()
    
    def _calculate_derivatives(self):
        """sympyを使用して3D偏導関数を計算"""
        # 一階偏導関数
        self.df_dx_expr = sp.diff(self.expr, self.x)
        self.df_dy_expr = sp.diff(self.expr, self.y)
        self.df_dz_expr = sp.diff(self.expr, self.z)
        
        # 二階偏導関数
        self.d2f_dx2_expr = sp.diff(self.df_dx_expr, self.x)
        self.d2f_dy2_expr = sp.diff(self.df_dy_expr, self.y)
        self.d2f_dz2_expr = sp.diff(self.df_dz_expr, self.z)
        self.d2f_dxdy_expr = sp.diff(self.df_dx_expr, self.y)
        self.d2f_dxdz_expr = sp.diff(self.df_dx_expr, self.z)
        self.d2f_dydz_expr = sp.diff(self.df_dy_expr, self.z)
        
        # 三階偏導関数
        self.d3f_dx3_expr = sp.diff(self.d2f_dx2_expr, self.x)
        self.d3f_dy3_expr = sp.diff(self.d2f_dy2_expr, self.y)
        self.d3f_dz3_expr = sp.diff(self.d2f_dz2_expr, self.z)
    
    def _create_callable_functions(self):
        """シンボリック表現から呼び出し可能関数を作成"""
        # cupyモジュールが問題を起こす可能性があるので、numpy も指定しておく
        modules = ["cupy", "numpy"]
        
        # 関数
        self.f_func = sp.lambdify((self.x, self.y, self.z), self.expr, modules=modules)
        
        # 一階偏導関数
        self.df_dx_func = sp.lambdify((self.x, self.y, self.z), self.df_dx_expr, modules=modules)
        self.df_dy_func = sp.lambdify((self.x, self.y, self.z), self.df_dy_expr, modules=modules)
        self.df_dz_func = sp.lambdify((self.x, self.y, self.z), self.df_dz_expr, modules=modules)
        
        # 二階偏導関数
        self.d2f_dx2_func = sp.lambdify((self.x, self.y, self.z), self.d2f_dx2_expr, modules=modules)
        self.d2f_dy2_func = sp.lambdify((self.x, self.y, self.z), self.d2f_dy2_expr, modules=modules)
        self.d2f_dz2_func = sp.lambdify((self.x, self.y, self.z), self.d2f_dz2_expr, modules=modules)
        
        # 三階偏導関数
        self.d3f_dx3_func = sp.lambdify((self.x, self.y, self.z), self.d3f_dx3_expr, modules=modules)
        self.d3f_dy3_func = sp.lambdify((self.x, self.y, self.z), self.d3f_dy3_expr, modules=modules)
        self.d3f_dz3_func = sp.lambdify((self.x, self.y, self.z), self.d3f_dz3_expr, modules=modules)
    
    def f(self, x, y, z) -> float:
        """
        関数値を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            関数値 f(x,y,z)
        """
        return float(self.f_func(x, y, z))
    
    def df_dx(self, x, y, z) -> float:
        """
        x方向の1階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            x方向偏導関数値 ∂f/∂x
        """
        return float(self.df_dx_func(x, y, z))
    
    def df_dy(self, x, y, z) -> float:
        """
        y方向の1階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            y方向偏導関数値 ∂f/∂y
        """
        return float(self.df_dy_func(x, y, z))
    
    def df_dz(self, x, y, z) -> float:
        """
        z方向の1階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            z方向偏導関数値 ∂f/∂z
        """
        return float(self.df_dz_func(x, y, z))
    
    def d2f_dx2(self, x, y, z) -> float:
        """
        x方向の2階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            x方向2階偏導関数値 ∂²f/∂x²
        """
        return float(self.d2f_dx2_func(x, y, z))
    
    def d2f_dy2(self, x, y, z) -> float:
        """
        y方向の2階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            y方向2階偏導関数値 ∂²f/∂y²
        """
        return float(self.d2f_dy2_func(x, y, z))
    
    def d2f_dz2(self, x, y, z) -> float:
        """
        z方向の2階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            z方向2階偏導関数値 ∂²f/∂z²
        """
        return float(self.d2f_dz2_func(x, y, z))
    
    def d3f_dx3(self, x, y, z) -> float:
        """
        x方向の3階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            x方向3階偏導関数値 ∂³f/∂x³
        """
        return float(self.d3f_dx3_func(x, y, z))
    
    def d3f_dy3(self, x, y, z) -> float:
        """
        y方向の3階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            y方向3階偏導関数値 ∂³f/∂y³
        """
        return float(self.d3f_dy3_func(x, y, z))
    
    def d3f_dz3(self, x, y, z) -> float:
        """
        z方向の3階偏導関数を評価
        
        Args:
            x: x座標
            y: y座標
            z: z座標
            
        Returns:
            z方向3階偏導関数値 ∂³f/∂z³
        """
        return float(self.d3f_dz3_func(x, y, z))
    
    def get_dimension(self) -> int:
        """
        関数の次元を取得
        
        Returns:
            3 (3次元関数)
        """
        return 3
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"3D TestFunction: {self.name}, f(x,y,z) = {self.expr}"
    
    def __repr__(self) -> str:
        """オブジェクト表現"""
        return f"TestFunction3D('{self.name}', {self.expr})"
    
    @classmethod
    def from_2d(cls, func_2d: TestFunction2D, method='extrude', suffix='') -> 'TestFunction3D':
        """
        2Dテスト関数から3Dテスト関数を作成
        
        Args:
            func_2d: 2Dテスト関数
            method: 拡張メソッド ('extrude', 'product', 'radial')
            suffix: 名前に追加する接尾辞
            
        Returns:
            3Dテスト関数
        """
        x, y, z = sp.symbols('x y z')
        
        if method == 'extrude':
            # f(x,y,z) = f(x,y)
            name = f"{func_2d.name}Extruded{suffix}"
            expr_3d = func_2d.expr
        elif method == 'product':
            # f(x,y,z) = f(x,y) * g(z) (zに対しては1次元関数をデフォルト使用)
            name = f"{func_2d.name}Product{suffix}"
            # zに対してはsin関数を使用
            z_expr = sp.sin(sp.pi * z)
            expr_3d = func_2d.expr * z_expr
        elif method == 'radial':
            # f(x,y,z) = f(√(x^2 + y^2 + z^2))
            name = f"{func_2d.name}Radial{suffix}"
            r = sp.sqrt(x**2 + y**2 + z**2)
            # 2次元関数を1次元関数として扱う
            r_expr = func_2d.expr.subs([(func_2d.x, r), (func_2d.y, 0)])
            expr_3d = r_expr
        else:
            raise ValueError(f"未知の拡張メソッド: {method}")
        
        return cls(name, expr_3d)
    
    @classmethod
    def from_1d(cls, func_1d: TestFunction1D, method='triple_product', suffix='') -> 'TestFunction3D':
        """
        1Dテスト関数から3Dテスト関数を作成
        
        Args:
            func_1d: 1Dテスト関数
            method: 拡張メソッド ('triple_product', 'additive', 'radial')
            suffix: 名前に追加する接尾辞
            
        Returns:
            3Dテスト関数
        """
        x, y, z = sp.symbols('x y z')
        expr_x = func_1d.expr.subs(func_1d.x, x)
        expr_y = func_1d.expr.subs(func_1d.x, y)
        expr_z = func_1d.expr.subs(func_1d.x, z)
        
        if method == 'triple_product':
            # f(x,y,z) = f(x) * f(y) * f(z)
            name = f"{func_1d.name}TripleProduct{suffix}"
            expr_3d = expr_x * expr_y * expr_z
        elif method == 'additive':
            # f(x,y,z) = f(x) + f(y) + f(z)
            name = f"{func_1d.name}Additive{suffix}"
            expr_3d = expr_x + expr_y + expr_z
        elif method == 'radial':
            # f(x,y,z) = f(√(x^2 + y^2 + z^2))
            name = f"{func_1d.name}Radial{suffix}"
            r = sp.sqrt(x**2 + y**2 + z**2)
            expr_3d = func_1d.expr.subs(func_1d.x, r)
        else:
            raise ValueError(f"未知の拡張メソッド: {method}")
        
        return cls(name, expr_3d)


class TestFunction3DFactory:
    """3Dテスト関数生成ファクトリークラス"""
    
    @staticmethod
    def create_basic_functions() -> List[TestFunction3D]:
        """基本的な3Dテスト関数を作成"""
        x, y, z = sp.symbols('x y z')
        pi = sp.pi
        
        functions = [
            TestFunction3D("Sine3D", sp.sin(pi * x) * sp.sin(pi * y) * sp.sin(pi * z)),
            TestFunction3D("Exp3D", sp.exp(-(x**2 + y**2 + z**2))),
            TestFunction3D("Poly3D", (1 - x**2) * (1 - y**2) * (1 - z**2)),
        ]
        
        return functions
    
    @staticmethod
    def create_from_2d(func_2d_names: Optional[List[str]] = None) -> List[TestFunction3D]:
        """
        2D関数から拡張された3Dテスト関数を作成
        
        Args:
            func_2d_names: 拡張する2D関数名のリスト（Noneの場合は代表的な関数を使用）
            
        Returns:
            拡張された3Dテスト関数のリスト
        """
        from test_function.test_function2d import TestFunction2DFactory
        
        # 2Dテスト関数を取得
        functions_2d = TestFunction2DFactory.create_standard_functions()
        
        functions_3d = []
        
        # 代表的な関数だけを選択（指定がない場合）
        if func_2d_names is None:
            func_2d_names = ["Sine2D", "Exp2D", "Poly2D"]
            
        selected_2d_funcs = [func for func in functions_2d if func.name in func_2d_names]
        
        for func in selected_2d_funcs:
            # 押し出し拡張
            functions_3d.append(TestFunction3D.from_2d(func, method='extrude'))
            
            # 積による拡張
            functions_3d.append(TestFunction3D.from_2d(func, method='product'))
            
            # 関数によっては半径方向拡張も追加
            if func.name in ["Exp2D"]:
                functions_3d.append(TestFunction3D.from_2d(func, method='radial'))
        
        return functions_3d
    
    @staticmethod
    def create_from_1d(func_1d_names: Optional[List[str]] = None) -> List[TestFunction3D]:
        """
        1D関数から拡張された3Dテスト関数を作成
        
        Args:
            func_1d_names: 拡張する1D関数名のリスト（Noneの場合は代表的な関数を使用）
            
        Returns:
            拡張された3Dテスト関数のリスト
        """
        from test_function.test_function1d import TestFunction1DFactory
        
        # 1Dテスト関数を取得
        functions_1d = TestFunction1DFactory.create_standard_functions()
        
        functions_3d = []
        
        # 代表的な関数だけを選択（指定がない場合）
        if func_1d_names is None:
            func_1d_names = ["Sine", "ExpMod", "QuadPoly"]
            
        selected_1d_funcs = [func for func in functions_1d if func.name in func_1d_names]
        
        for func in selected_1d_funcs:
            # 積による拡張
            functions_3d.append(TestFunction3D.from_1d(func, method='triple_product'))
            
            # 和による拡張
            functions_3d.append(TestFunction3D.from_1d(func, method='additive'))
            
            # 半径方向拡張
            if func.name in ["Sine", "ExpMod"]:
                functions_3d.append(TestFunction3D.from_1d(func, method='radial'))
        
        return functions_3d
    
    @staticmethod
    def create_standard_functions() -> List[TestFunction3D]:
        """標準的な3Dテスト関数セット（基本 + 拡張）を作成"""
        basic_funcs = TestFunction3DFactory.create_basic_functions()
        extended_funcs_2d = TestFunction3DFactory.create_from_2d()
        extended_funcs_1d = TestFunction3DFactory.create_from_1d()
        
        all_funcs = basic_funcs + extended_funcs_2d + extended_funcs_1d
        
        return all_funcs
    
    @staticmethod
    def get_function_by_name(name: str) -> Optional[TestFunction3D]:
        """
        名前から関数を取得
        
        Args:
            name: 関数名
            
        Returns:
            対応するTestFunction3D、見つからない場合はNone
        """
        functions = TestFunction3DFactory.create_standard_functions()
        return next((f for f in functions if f.name == name), None)
