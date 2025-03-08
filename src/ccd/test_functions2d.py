import cupy as cp
from dataclasses import dataclass
from typing import Callable, List
from test_functions1d import TestFunction, TestFunctionFactory

@dataclass
class TestFunction2D:
    """2Dテスト関数とその偏導関数を保持するデータクラス"""
    
    name: str
    f: Callable[[float, float], float]  # f(x, y)
    
    # 一階偏導関数
    df_dx: Callable[[float, float], float]  # ∂f/∂x
    df_dy: Callable[[float, float], float]  # ∂f/∂y
    
    # 二階偏導関数
    d2f_dx2: Callable[[float, float], float]  # ∂²f/∂x²
    d2f_dy2: Callable[[float, float], float]  # ∂²f/∂y²
    d2f_dxdy: Callable[[float, float], float]  # ∂²f/∂x∂y
    
    # 三階偏導関数
    d3f_dx3: Callable[[float, float], float]  # ∂³f/∂x³
    d3f_dy3: Callable[[float, float], float]  # ∂³f/∂y³
    d3f_dx2dy: Callable[[float, float], float]  # ∂³f/∂x²∂y
    d3f_dxdy2: Callable[[float, float], float]  # ∂³f/∂x∂y²


class TestFunction2DGenerator:
    """1Dテスト関数から2Dテスト関数を生成するクラス"""
    
    @staticmethod
    def product_extension(func1d: TestFunction, suffix: str = "") -> TestFunction2D:
        """
        1次元テスト関数をテンソル積で拡張した2次元関数を生成
        f(x,y) = f1d(x) * f1d(y)
        
        Args:
            func1d: 1次元テスト関数
            suffix: 名前に追加する接尾辞
            
        Returns:
            テンソル積拡張された2次元テスト関数
        """
        name = f"{func1d.name}Product{suffix}"
        
        # 2D関数とその偏導関数を定義
        def f(x, y):
            return func1d.f(x) * func1d.f(y)
        
        # 一階偏導関数
        def df_dx(x, y):
            return func1d.df(x) * func1d.f(y)
        def df_dy(x, y):
            return func1d.f(x) * func1d.df(y)
        
        # 二階偏導関数
        def d2f_dx2(x, y):
            return func1d.d2f(x) * func1d.f(y)
        def d2f_dy2(x, y):
            return func1d.f(x) * func1d.d2f(y)
        def d2f_dxdy(x, y):
            return func1d.df(x) * func1d.df(y)
        
        # 三階偏導関数
        def d3f_dx3(x, y):
            return func1d.d3f(x) * func1d.f(y)
        def d3f_dy3(x, y):
            return func1d.f(x) * func1d.d3f(y)
        def d3f_dx2dy(x, y):
            return func1d.d2f(x) * func1d.df(y)
        def d3f_dxdy2(x, y):
            return func1d.df(x) * func1d.d2f(y)
        
        return TestFunction2D(
            name=name,
            f=f,
            df_dx=df_dx,
            df_dy=df_dy,
            d2f_dx2=d2f_dx2,
            d2f_dy2=d2f_dy2,
            d2f_dxdy=d2f_dxdy,
            d3f_dx3=d3f_dx3,
            d3f_dy3=d3f_dy3,
            d3f_dx2dy=d3f_dx2dy,
            d3f_dxdy2=d3f_dxdy2
        )
    
    @staticmethod
    def sum_extension(func1d: TestFunction, suffix: str = "") -> TestFunction2D:
        """
        1次元テスト関数を和で拡張した2次元関数を生成
        f(x,y) = f1d(x) + f1d(y)
        
        Args:
            func1d: 1次元テスト関数
            suffix: 名前に追加する接尾辞
            
        Returns:
            和で拡張された2次元テスト関数
        """
        name = f"{func1d.name}Sum{suffix}"
        
        # 2D関数とその偏導関数を定義
        def f(x, y):
            return func1d.f(x) + func1d.f(y)
        
        # 一階偏導関数
        def df_dx(x, y):
            return func1d.df(x)
        def df_dy(x, y):
            return func1d.df(y)
        
        # 二階偏導関数
        def d2f_dx2(x, y):
            return func1d.d2f(x)
        def d2f_dy2(x, y):
            return func1d.d2f(y)
        def d2f_dxdy(x, y):
            return 0.0
        
        # 三階偏導関数
        def d3f_dx3(x, y):
            return func1d.d3f(x)
        def d3f_dy3(x, y):
            return func1d.d3f(y)
        def d3f_dx2dy(x, y):
            return 0.0
        def d3f_dxdy2(x, y):
            return 0.0
        
        return TestFunction2D(
            name=name,
            f=f,
            df_dx=df_dx,
            df_dy=df_dy,
            d2f_dx2=d2f_dx2,
            d2f_dy2=d2f_dy2,
            d2f_dxdy=d2f_dxdy,
            d3f_dx3=d3f_dx3,
            d3f_dy3=d3f_dy3,
            d3f_dx2dy=d3f_dx2dy,
            d3f_dxdy2=d3f_dxdy2
        )
    
    @staticmethod
    def radial_extension(func1d: TestFunction, suffix: str = "") -> TestFunction2D:
        """
        1次元テスト関数を半径方向に拡張した2次元関数を生成
        f(x,y) = f1d(sqrt(x^2 + y^2))
        
        Args:
            func1d: 1次元テスト関数
            suffix: 名前に追加する接尾辞
            
        Returns:
            半径方向に拡張された2次元テスト関数
        """
        name = f"{func1d.name}Radial{suffix}"
        
        # 2D関数とその偏導関数を定義
        def f(x, y):
            return func1d.f(cp.sqrt(x**2 + y**2))
        
        # 補助関数
        def r(x, y):
            return cp.sqrt(x**2 + y**2)
        
        def dr_dx(x, y):
            r_val = r(x, y)
            return x / r_val if r_val > 1e-10 else 0.0
        
        def dr_dy(x, y):
            r_val = r(x, y)
            return y / r_val if r_val > 1e-10 else 0.0
        
        def d2r_dx2(x, y):
            r_val = r(x, y)
            if r_val < 1e-10:
                return 0.0
            return y**2 / (r_val**3)
        
        def d2r_dy2(x, y):
            r_val = r(x, y)
            if r_val < 1e-10:
                return 0.0
            return x**2 / (r_val**3)
        
        def d2r_dxdy(x, y):
            r_val = r(x, y)
            if r_val < 1e-10:
                return 0.0
            return -x * y / (r_val**3)
        
        # 一階偏導関数
        def df_dx(x, y):
            return func1d.df(r(x, y)) * dr_dx(x, y)
        def df_dy(x, y):
            return func1d.df(r(x, y)) * dr_dy(x, y)
        
        # 二階偏導関数
        def d2f_dx2(x, y):
            return func1d.d2f(r(x, y)) * dr_dx(x, y)**2 + func1d.df(r(x, y)) * d2r_dx2(x, y)
        def d2f_dy2(x, y):
            return func1d.d2f(r(x, y)) * dr_dy(x, y)**2 + func1d.df(r(x, y)) * d2r_dy2(x, y)
        def d2f_dxdy(x, y):
            return func1d.d2f(r(x, y)) * dr_dx(x, y) * dr_dy(x, y) + func1d.df(r(x, y)) * d2r_dxdy(x, y)
        
        # 三階偏導関数（簡易版 - 完全な式は非常に複雑になるため）
        def d3f_dx3(x, y):
            r_val = r(x, y)
            drx = dr_dx(x, y)
            d2rx = d2r_dx2(x, y)
            return func1d.d3f(r_val) * drx**3 + 3 * func1d.d2f(r_val) * drx * d2rx
        
        def d3f_dy3(x, y):
            r_val = r(x, y)
            dry = dr_dy(x, y)
            d2ry = d2r_dy2(x, y)
            return func1d.d3f(r_val) * dry**3 + 3 * func1d.d2f(r_val) * dry * d2ry
        
        def d3f_dx2dy(x, y):
            r_val = r(x, y)
            drx = dr_dx(x, y)
            dry = dr_dy(x, y)
            d2rxy = d2r_dxdy(x, y)
            return func1d.d3f(r_val) * drx**2 * dry + func1d.d2f(r_val) * 2 * drx * d2rxy
        
        def d3f_dxdy2(x, y):
            r_val = r(x, y)
            drx = dr_dx(x, y)
            dry = dr_dy(x, y)
            d2rxy = d2r_dxdy(x, y)
            return func1d.d3f(r_val) * drx * dry**2 + func1d.d2f(r_val) * 2 * dry * d2rxy
        
        return TestFunction2D(
            name=name,
            f=f,
            df_dx=df_dx,
            df_dy=df_dy,
            d2f_dx2=d2f_dx2,
            d2f_dy2=d2f_dy2,
            d2f_dxdy=d2f_dxdy,
            d3f_dx3=d3f_dx3,
            d3f_dy3=d3f_dy3,
            d3f_dx2dy=d3f_dx2dy,
            d3f_dxdy2=d3f_dxdy2
        )
    
    @staticmethod
    def create_basic_2d_functions() -> List[TestFunction2D]:
        """基本的な2Dテスト関数を生成"""
        # 特別な2D関数を直接定義
        sine_2d = TestFunction2D(
            name="Sine2D",
            f=lambda x, y: cp.sin(cp.pi * x) * cp.sin(cp.pi * y),
            df_dx=lambda x, y: cp.pi * cp.cos(cp.pi * x) * cp.sin(cp.pi * y),
            df_dy=lambda x, y: cp.pi * cp.sin(cp.pi * x) * cp.cos(cp.pi * y),
            d2f_dx2=lambda x, y: -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y),
            d2f_dy2=lambda x, y: -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y),
            d2f_dxdy=lambda x, y: (cp.pi**2) * cp.cos(cp.pi * x) * cp.cos(cp.pi * y),
            d3f_dx3=lambda x, y: -(cp.pi**3) * cp.cos(cp.pi * x) * cp.sin(cp.pi * y),
            d3f_dy3=lambda x, y: -(cp.pi**3) * cp.sin(cp.pi * x) * cp.cos(cp.pi * y),
            d3f_dx2dy=lambda x, y: -(cp.pi**3) * cp.sin(cp.pi * x) * cp.cos(cp.pi * y),
            d3f_dxdy2=lambda x, y: -(cp.pi**3) * cp.cos(cp.pi * x) * cp.sin(cp.pi * y)
        )
        
        exp_2d = TestFunction2D(
            name="Exp2D",
            f=lambda x, y: cp.exp(-(x**2 + y**2)),
            df_dx=lambda x, y: -2 * x * cp.exp(-(x**2 + y**2)),
            df_dy=lambda x, y: -2 * y * cp.exp(-(x**2 + y**2)),
            d2f_dx2=lambda x, y: (-2 + 4 * x**2) * cp.exp(-(x**2 + y**2)),
            d2f_dy2=lambda x, y: (-2 + 4 * y**2) * cp.exp(-(x**2 + y**2)),
            d2f_dxdy=lambda x, y: 4 * x * y * cp.exp(-(x**2 + y**2)),
            d3f_dx3=lambda x, y: (12 * x - 8 * x**3) * cp.exp(-(x**2 + y**2)),
            d3f_dy3=lambda x, y: (12 * y - 8 * y**3) * cp.exp(-(x**2 + y**2)),
            d3f_dx2dy=lambda x, y: (-4 * y + 8 * x**2 * y) * cp.exp(-(x**2 + y**2)),
            d3f_dxdy2=lambda x, y: (-4 * x + 8 * x * y**2) * cp.exp(-(x**2 + y**2))
        )
        
        poly_2d = TestFunction2D(
            name="Poly2D",
            f=lambda x, y: (1 - x**2) * (1 - y**2),
            df_dx=lambda x, y: -2 * x * (1 - y**2),
            df_dy=lambda x, y: -2 * y * (1 - x**2),
            d2f_dx2=lambda x, y: -2 * (1 - y**2),
            d2f_dy2=lambda x, y: -2 * (1 - x**2),
            d2f_dxdy=lambda x, y: 4 * x * y,
            d3f_dx3=lambda x, y: 0.0,
            d3f_dy3=lambda x, y: 0.0,
            d3f_dx2dy=lambda x, y: 4 * y,
            d3f_dxdy2=lambda x, y: 4 * x
        )
        
        return [sine_2d, exp_2d, poly_2d]
    
    @staticmethod
    def create_extended_functions() -> List[TestFunction2D]:
        """1D関数から拡張された2Dテスト関数を生成"""
        # 1Dテスト関数を取得
        functions_1d = TestFunctionFactory.create_standard_functions()
        
        functions_2d = []
        
        # 代表的な関数だけを拡張
        selected_1d_funcs = [func for func in functions_1d if func.name in ["Sine", "Cosine", "ExpMod", "Runge"]]
        
        for func in selected_1d_funcs:
            # テンソル積拡張
            functions_2d.append(TestFunction2DGenerator.product_extension(func))
            
            # 和による拡張
            functions_2d.append(TestFunction2DGenerator.sum_extension(func))
            
            # 半径方向拡張 (Sineなどの特定の関数のみ)
            if func.name in ["Sine", "ExpMod"]:
                functions_2d.append(TestFunction2DGenerator.radial_extension(func))
        
        return functions_2d
    
    @staticmethod
    def create_standard_functions() -> List[TestFunction2D]:
        """標準的な2Dテスト関数セットを作成（基本関数と拡張関数の組み合わせ）"""
        # 基本的な2D関数
        basic_funcs = TestFunction2DGenerator.create_basic_2d_functions()
        
        # 拡張された2D関数
        extended_funcs = TestFunction2DGenerator.create_extended_functions()
        
        # すべての関数を結合
        all_funcs = basic_funcs + extended_funcs
        
        return all_funcs