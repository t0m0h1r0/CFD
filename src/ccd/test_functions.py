import sympy as sp
from typing import List

class TestFunction:
    """1D、2D、3D対応のテスト関数を統合した自動微分機能付きクラス"""
    
    def __init__(self, name: str, expr_1d=None, expr_2d=None, expr_3d=None, dims=1):
        """
        シンボリック表現によるテスト関数の初期化
        
        Args:
            name: 関数名
            expr_1d: 1Dシンボリック表現（xの関数）
            expr_2d: 2Dシンボリック表現（x, yの関数）
            expr_3d: 3Dシンボリック表現（x, y, zの関数）
            dims: 次元数（1, 2, または3）
        """
        self.name = name
        self.dims = dims
        
        # シンボリック変数の定義
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.z = sp.Symbol('z')
        
        # シンボリック表現の保存
        if dims == 1 and expr_1d is not None:
            self.expr = expr_1d
            # 導関数の計算
            self._calculate_1d_derivatives()
        elif dims == 2 and expr_2d is not None:
            self.expr = expr_2d
            # 導関数の計算
            self._calculate_2d_derivatives()
        elif dims == 3 and expr_3d is not None:
            self.expr = expr_3d
            # 導関数の計算
            self._calculate_3d_derivatives()
        else:
            raise ValueError("無効な次元または表現が欠落しています")
        
        # 呼び出し可能関数の作成
        self._create_callable_functions()
    
    def _calculate_1d_derivatives(self):
        """sympyを使用して1D導関数を計算"""
        self.df_expr = sp.diff(self.expr, self.x)
        self.d2f_expr = sp.diff(self.df_expr, self.x)
        self.d3f_expr = sp.diff(self.d2f_expr, self.x)
    
    def _calculate_2d_derivatives(self):
        """sympyを使用して2D導関数を計算"""
        # 一階導関数
        self.df_dx_expr = sp.diff(self.expr, self.x)
        self.df_dy_expr = sp.diff(self.expr, self.y)
        
        # 二階導関数
        self.d2f_dx2_expr = sp.diff(self.df_dx_expr, self.x)
        self.d2f_dy2_expr = sp.diff(self.df_dy_expr, self.y)
        self.d2f_dxdy_expr = sp.diff(self.df_dx_expr, self.y)
        
        # 三階導関数
        self.d3f_dx3_expr = sp.diff(self.d2f_dx2_expr, self.x)
        self.d3f_dy3_expr = sp.diff(self.d2f_dy2_expr, self.y)
        self.d3f_dx2dy_expr = sp.diff(self.d2f_dx2_expr, self.y)
        self.d3f_dxdy2_expr = sp.diff(self.d2f_dxdy_expr, self.y)
    
    def _calculate_3d_derivatives(self):
        """sympyを使用して3D導関数を計算"""
        # 一階導関数
        self.df_dx_expr = sp.diff(self.expr, self.x)
        self.df_dy_expr = sp.diff(self.expr, self.y)
        self.df_dz_expr = sp.diff(self.expr, self.z)
        
        # 二階導関数
        self.d2f_dx2_expr = sp.diff(self.df_dx_expr, self.x)
        self.d2f_dy2_expr = sp.diff(self.df_dy_expr, self.y)
        self.d2f_dz2_expr = sp.diff(self.df_dz_expr, self.z)
        self.d2f_dxdy_expr = sp.diff(self.df_dx_expr, self.y)
        self.d2f_dxdz_expr = sp.diff(self.df_dx_expr, self.z)
        self.d2f_dydz_expr = sp.diff(self.df_dy_expr, self.z)
        
        # 三階導関数
        self.d3f_dx3_expr = sp.diff(self.d2f_dx2_expr, self.x)
        self.d3f_dy3_expr = sp.diff(self.d2f_dy2_expr, self.y)
        self.d3f_dz3_expr = sp.diff(self.d2f_dz2_expr, self.z)
        self.d3f_dx2dy_expr = sp.diff(self.d2f_dx2_expr, self.y)
        self.d3f_dx2dz_expr = sp.diff(self.d2f_dx2_expr, self.z)
        self.d3f_dxdy2_expr = sp.diff(self.d2f_dxdy_expr, self.y)
        self.d3f_dxdz2_expr = sp.diff(self.d2f_dxdz_expr, self.z)
        self.d3f_dy2dz_expr = sp.diff(self.d2f_dy2_expr, self.z)
        self.d3f_dydz2_expr = sp.diff(self.d2f_dydz_expr, self.z)
    
    def _create_callable_functions(self):
        """シンボリック表現から呼び出し可能関数を作成"""
        # cupyモジュールが問題を起こす可能性があるので、numpy も指定しておく
        modules = ["cupy", "numpy"]
        
        if self.dims == 1:
            # 1D関数
            self.f = sp.lambdify(self.x, self.expr, modules=modules)
            self.df = sp.lambdify(self.x, self.df_expr, modules=modules)
            self.d2f = sp.lambdify(self.x, self.d2f_expr, modules=modules)
            self.d3f = sp.lambdify(self.x, self.d3f_expr, modules=modules)
            
        elif self.dims == 2:
            # 2D関数
            self.f = sp.lambdify((self.x, self.y), self.expr, modules=modules)
            self.df_dx = sp.lambdify((self.x, self.y), self.df_dx_expr, modules=modules)
            self.df_dy = sp.lambdify((self.x, self.y), self.df_dy_expr, modules=modules)
            self.d2f_dx2 = sp.lambdify((self.x, self.y), self.d2f_dx2_expr, modules=modules)
            self.d2f_dy2 = sp.lambdify((self.x, self.y), self.d2f_dy2_expr, modules=modules)
            self.d2f_dxdy = sp.lambdify((self.x, self.y), self.d2f_dxdy_expr, modules=modules)
            self.d3f_dx3 = sp.lambdify((self.x, self.y), self.d3f_dx3_expr, modules=modules)
            self.d3f_dy3 = sp.lambdify((self.x, self.y), self.d3f_dy3_expr, modules=modules)
            self.d3f_dx2dy = sp.lambdify((self.x, self.y), self.d3f_dx2dy_expr, modules=modules)
            self.d3f_dxdy2 = sp.lambdify((self.x, self.y), self.d3f_dxdy2_expr, modules=modules)
            
        else:  # self.dims == 3
            # 3D関数
            self.f = sp.lambdify((self.x, self.y, self.z), self.expr, modules=modules)
            # 一階導関数
            self.df_dx = sp.lambdify((self.x, self.y, self.z), self.df_dx_expr, modules=modules)
            self.df_dy = sp.lambdify((self.x, self.y, self.z), self.df_dy_expr, modules=modules)
            self.df_dz = sp.lambdify((self.x, self.y, self.z), self.df_dz_expr, modules=modules)
            # 二階導関数
            self.d2f_dx2 = sp.lambdify((self.x, self.y, self.z), self.d2f_dx2_expr, modules=modules)
            self.d2f_dy2 = sp.lambdify((self.x, self.y, self.z), self.d2f_dy2_expr, modules=modules)
            self.d2f_dz2 = sp.lambdify((self.x, self.y, self.z), self.d2f_dz2_expr, modules=modules)
            self.d2f_dxdy = sp.lambdify((self.x, self.y, self.z), self.d2f_dxdy_expr, modules=modules)
            self.d2f_dxdz = sp.lambdify((self.x, self.y, self.z), self.d2f_dxdz_expr, modules=modules)
            self.d2f_dydz = sp.lambdify((self.x, self.y, self.z), self.d2f_dydz_expr, modules=modules)
            # 三階導関数
            self.d3f_dx3 = sp.lambdify((self.x, self.y, self.z), self.d3f_dx3_expr, modules=modules)
            self.d3f_dy3 = sp.lambdify((self.x, self.y, self.z), self.d3f_dy3_expr, modules=modules)
            self.d3f_dz3 = sp.lambdify((self.x, self.y, self.z), self.d3f_dz3_expr, modules=modules)
            self.d3f_dx2dy = sp.lambdify((self.x, self.y, self.z), self.d3f_dx2dy_expr, modules=modules)
            self.d3f_dx2dz = sp.lambdify((self.x, self.y, self.z), self.d3f_dx2dz_expr, modules=modules)
            self.d3f_dxdy2 = sp.lambdify((self.x, self.y, self.z), self.d3f_dxdy2_expr, modules=modules)
            self.d3f_dxdz2 = sp.lambdify((self.x, self.y, self.z), self.d3f_dxdz2_expr, modules=modules)
            self.d3f_dy2dz = sp.lambdify((self.x, self.y, self.z), self.d3f_dy2dz_expr, modules=modules)
            self.d3f_dydz2 = sp.lambdify((self.x, self.y, self.z), self.d3f_dydz2_expr, modules=modules)
            
    @classmethod
    def create_1d(cls, name: str, expr):
        """1Dテスト関数を作成"""
        return cls(name, expr_1d=expr, dims=1)
    
    @classmethod
    def create_2d(cls, name: str, expr):
        """2Dテスト関数を作成"""
        return cls(name, expr_2d=expr, dims=2)
    
    @classmethod
    def create_3d(cls, name: str, expr):
        """3Dテスト関数を作成"""
        return cls(name, expr_3d=expr, dims=3)
    
    @classmethod
    def from_1d_to_2d(cls, func_1d, method='product', suffix=''):
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
        
        return cls(name, expr_2d=expr_2d, dims=2)
    
    @classmethod
    def from_1d_to_3d(cls, func_1d, method='product', suffix=''):
        """
        1Dテスト関数から3Dテスト関数を作成
        
        Args:
            func_1d: 1Dテスト関数
            method: 拡張メソッド ('product', 'sum', 'radial')
            suffix: 名前に追加する接尾辞
            
        Returns:
            3Dテスト関数
        """
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        z = sp.Symbol('z')
        
        if method == 'product':
            # f(x,y,z) = f(x) * f(y) * f(z)
            name = f"{func_1d.name}Product3D{suffix}"
            expr_3d = func_1d.expr.subs(x, x) * func_1d.expr.subs(x, y) * func_1d.expr.subs(x, z)
        elif method == 'sum':
            # f(x,y,z) = f(x) + f(y) + f(z)
            name = f"{func_1d.name}Sum3D{suffix}"
            expr_3d = func_1d.expr.subs(x, x) + func_1d.expr.subs(x, y) + func_1d.expr.subs(x, z)
        elif method == 'radial':
            # f(x,y,z) = f(sqrt(x^2 + y^2 + z^2))
            name = f"{func_1d.name}Radial3D{suffix}"
            r = sp.sqrt(x**2 + y**2 + z**2)
            expr_3d = func_1d.expr.subs(x, r)
        else:
            raise ValueError(f"未知の拡張メソッド: {method}")
        
        return cls(name, expr_3d=expr_3d, dims=3)
    
    @classmethod
    def from_2d_to_3d(cls, func_2d, method='extrude', suffix=''):
        """
        2Dテスト関数から3Dテスト関数を作成
        
        Args:
            func_2d: 2Dテスト関数
            method: 拡張メソッド ('extrude', 'rotate', 'product')
            suffix: 名前に追加する接尾辞
            
        Returns:
            3Dテスト関数
        """
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        z = sp.Symbol('z')
        
        if method == 'extrude':
            # 2D関数をz方向に押し出し
            name = f"{func_2d.name}Extrude{suffix}"
            expr_3d = func_2d.expr
        elif method == 'rotate':
            # xy平面の関数をz軸周りに回転
            name = f"{func_2d.name}Rotate{suffix}"
            r = sp.sqrt(x**2 + y**2)
            expr_3d = func_2d.expr.subs([(x, r), (y, z)])
        elif method == 'product':
            # f(x,y,z) = f(x,y) * f(z)
            name = f"{func_2d.name}Product{suffix}"
            # zにはシンプルな関数（例: sin(z)）を使用
            f_z = sp.sin(sp.pi * z)
            expr_3d = func_2d.expr * f_z
        else:
            raise ValueError(f"未知の拡張メソッド: {method}")
        
        return cls(name, expr_3d=expr_3d, dims=3)


class TestFunctionFactory:
    """標準テスト関数生成ファクトリークラス"""
    
    @staticmethod
    def create_standard_1d_functions() -> List[TestFunction]:
        """標準的な1Dテスト関数を作成"""
        x = sp.Symbol('x')
        pi = sp.pi
        
        functions = [
            TestFunction.create_1d("Zero", 0),
            TestFunction.create_1d("QuadPoly", 1 - x**2),
            TestFunction.create_1d("CubicPoly", (1 - x) * (1 + x) * (x + 0.5)),
            TestFunction.create_1d("Sine", sp.sin(pi * x)),
            TestFunction.create_1d("Cosine", sp.cos(2 * pi * x)),
            TestFunction.create_1d("ExpMod", sp.exp(-(x**2)) - sp.exp(-1)),
            TestFunction.create_1d("HigherPoly", x**4 - x**2),
            TestFunction.create_1d("CompoundPoly", x**2 * (1 - x**2)),
            TestFunction.create_1d("Runge", 1 / (1 + 25 * x**2)),
            # ModifiedExp は絶対値関数を含み微分不可能なので除外
            # 代わりに同様の形状を持つ滑らかな関数を追加
            TestFunction.create_1d("SmoothExp", sp.exp(-10 * x**2)),
            TestFunction.create_1d("HighFreqSine", sp.sin(20 * pi * x)),
        ]
        
        return functions
    
    @staticmethod
    def create_basic_2d_functions() -> List[TestFunction]:
        """基本的な2Dテスト関数を作成"""
        x, y = sp.symbols('x y')
        pi = sp.pi
        
        functions = [
            TestFunction.create_2d("Sine2D", sp.sin(pi * x) * sp.sin(pi * y)),
            TestFunction.create_2d("Exp2D", sp.exp(-(x**2 + y**2))),
            TestFunction.create_2d("Poly2D", (1 - x**2) * (1 - y**2)),
        ]
        
        return functions
    
    @staticmethod
    def create_basic_3d_functions() -> List[TestFunction]:
        """基本的な3Dテスト関数を作成"""
        x, y, z = sp.symbols('x y z')
        pi = sp.pi
        
        functions = [
            TestFunction.create_3d("Sine3D", sp.sin(pi * x) * sp.sin(pi * y) * sp.sin(pi * z)),
            TestFunction.create_3d("Exp3D", sp.exp(-(x**2 + y**2 + z**2))),
            TestFunction.create_3d("Poly3D", (1 - x**2) * (1 - y**2) * (1 - z**2)),
            TestFunction.create_3d("Wave3D", sp.sin(pi * x) * sp.exp(-(y**2 + z**2))),
            TestFunction.create_3d("SinCosTan", sp.sin(pi * x) * sp.cos(pi * y) * sp.tan(pi * z / 4)),
        ]
        
        return functions
    
    @staticmethod
    def create_extended_2d_functions() -> List[TestFunction]:
        """1D関数から拡張された2Dテスト関数を作成"""
        # 1Dテスト関数を取得
        functions_1d = TestFunctionFactory.create_standard_1d_functions()
        
        functions_2d = []
        
        # 代表的な関数だけを選択
        selected_1d_funcs = [func for func in functions_1d if func.name in ["Sine", "Cosine", "ExpMod", "Runge"]]
        
        for func in selected_1d_funcs:
            # テンソル積拡張
            functions_2d.append(TestFunction.from_1d_to_2d(func, method='product'))
            
            # 和による拡張
            functions_2d.append(TestFunction.from_1d_to_2d(func, method='sum'))
            
            # 半径方向拡張 (特定の関数のみ)
            if func.name in ["Sine", "ExpMod"]:
                functions_2d.append(TestFunction.from_1d_to_2d(func, method='radial'))
        
        return functions_2d
    
    @staticmethod
    def create_extended_3d_functions() -> List[TestFunction]:
        """1D関数および2D関数から拡張された3Dテスト関数を作成"""
        # 基本テスト関数を取得
        functions_1d = TestFunctionFactory.create_standard_1d_functions()
        functions_2d = TestFunctionFactory.create_basic_2d_functions()
        
        functions_3d = []
        
        # 1D関数から拡張（代表的な関数のみ）
        selected_1d_funcs = [func for func in functions_1d if func.name in ["Sine", "Cosine", "ExpMod", "Runge"]]
        
        for func in selected_1d_funcs:
            # テンソル積拡張
            functions_3d.append(TestFunction.from_1d_to_3d(func, method='product'))
            
            # 和による拡張
            functions_3d.append(TestFunction.from_1d_to_3d(func, method='sum'))
            
            # 半径方向拡張 (特定の関数のみ)
            if func.name in ["Sine", "ExpMod"]:
                functions_3d.append(TestFunction.from_1d_to_3d(func, method='radial'))
        
        # 2D関数から拡張
        for func in functions_2d:
            # z方向への押し出し
            functions_3d.append(TestFunction.from_2d_to_3d(func, method='extrude'))
            
            # z方向とのプロダクト
            functions_3d.append(TestFunction.from_2d_to_3d(func, method='product'))
        
        return functions_3d
    
    @staticmethod
    def create_standard_2d_functions() -> List[TestFunction]:
        """標準的な2Dテスト関数セット（基本 + 拡張）を作成"""
        basic_funcs = TestFunctionFactory.create_basic_2d_functions()
        extended_funcs = TestFunctionFactory.create_extended_2d_functions()
        
        all_funcs = basic_funcs + extended_funcs
        
        return all_funcs
    
    @staticmethod
    def create_standard_3d_functions() -> List[TestFunction]:
        """標準的な3Dテスト関数セット（基本 + 拡張）を作成"""
        basic_funcs = TestFunctionFactory.create_basic_3d_functions()
        extended_funcs = TestFunctionFactory.create_extended_3d_functions()
        
        all_funcs = basic_funcs + extended_funcs
        
        return all_funcs