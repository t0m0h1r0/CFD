"""
テスト関数アダプターモジュール

既存のtest_functions.pyモジュールを2次元CCD用に活用するアダプターを提供します。
SOLID原則に従い、既存コードを変更せずに拡張します。
"""

import cupy as cp
from typing import List, Callable, Dict, Any, Tuple

# 既存の1次元テスト関数をインポート
from test_functions import TestFunction, TestFunctionFactory


class TestFunction2DAdapter:
    """
    1次元テスト関数を2次元用に変換するアダプタークラス
    
    このクラスは、既存の1次元テスト関数を変更せずに2次元テスト関数として使用できるようにします。
    (SOLID - OCP: 既存コードを変更せずに拡張)
    """
    
    def __init__(self, test_function_1d: TestFunction, extension_type: str = "product"):
        """
        1次元テスト関数を元に2次元テスト関数アダプタを初期化
        
        Args:
            test_function_1d: 元となる1次元テスト関数
            extension_type: 拡張方法
                - "product": f(x,y) = f(x) * f(y)
                - "sum": f(x,y) = f(x) + f(y)
                - "x_only": f(x,y) = f(x)
                - "y_only": f(x,y) = f(y)
        """
        self.test_function_1d = test_function_1d
        self.extension_type = extension_type
        self.name = f"{test_function_1d.name}2D"
    
    def f(self, x: float, y: float) -> float:
        """2次元関数値"""
        f1d = self.test_function_1d.f
        
        if self.extension_type == "product":
            return f1d(x) * f1d(y)
        elif self.extension_type == "sum":
            return f1d(x) + f1d(y)
        elif self.extension_type == "x_only":
            return f1d(x)
        elif self.extension_type == "y_only":
            return f1d(y)
        else:
            raise ValueError(f"Unknown extension type: {self.extension_type}")
    
    def f_x(self, x: float, y: float) -> float:
        """x方向偏導関数"""
        f1d = self.test_function_1d.f
        df1d = self.test_function_1d.df
        
        if self.extension_type == "product":
            return df1d(x) * f1d(y)
        elif self.extension_type == "sum":
            return df1d(x)
        elif self.extension_type == "x_only":
            return df1d(x)
        elif self.extension_type == "y_only":
            return 0.0
        else:
            raise ValueError(f"Unknown extension type: {self.extension_type}")
    
    def f_y(self, x: float, y: float) -> float:
        """y方向偏導関数"""
        f1d = self.test_function_1d.f
        df1d = self.test_function_1d.df
        
        if self.extension_type == "product":
            return f1d(x) * df1d(y)
        elif self.extension_type == "sum":
            return df1d(y)
        elif self.extension_type == "x_only":
            return 0.0
        elif self.extension_type == "y_only":
            return df1d(y)
        else:
            raise ValueError(f"Unknown extension type: {self.extension_type}")
    
    def f_xx(self, x: float, y: float) -> float:
        """x方向2階偏導関数"""
        f1d = self.test_function_1d.f
        d2f1d = self.test_function_1d.d2f
        
        if self.extension_type == "product":
            return d2f1d(x) * f1d(y)
        elif self.extension_type == "sum":
            return d2f1d(x)
        elif self.extension_type == "x_only":
            return d2f1d(x)
        elif self.extension_type == "y_only":
            return 0.0
        else:
            raise ValueError(f"Unknown extension type: {self.extension_type}")
    
    def f_yy(self, x: float, y: float) -> float:
        """y方向2階偏導関数"""
        f1d = self.test_function_1d.f
        d2f1d = self.test_function_1d.d2f
        
        if self.extension_type == "product":
            return f1d(x) * d2f1d(y)
        elif self.extension_type == "sum":
            return d2f1d(y)
        elif self.extension_type == "x_only":
            return 0.0
        elif self.extension_type == "y_only":
            return d2f1d(y)
        else:
            raise ValueError(f"Unknown extension type: {self.extension_type}")
    
    def f_xy(self, x: float, y: float) -> float:
        """混合偏導関数"""
        df1d = self.test_function_1d.df
        
        if self.extension_type == "product":
            return df1d(x) * df1d(y)
        elif self.extension_type == "sum":
            return 0.0
        elif self.extension_type == "x_only":
            return 0.0
        elif self.extension_type == "y_only":
            return 0.0
        else:
            raise ValueError(f"Unknown extension type: {self.extension_type}")


class TestFunction2D:
    """2次元テスト関数クラス"""
    
    def __init__(
        self,
        name: str,
        f: Callable[[float, float], float],
        f_x: Callable[[float, float], float],
        f_y: Callable[[float, float], float],
        f_xx: Callable[[float, float], float],
        f_yy: Callable[[float, float], float],
        f_xy: Callable[[float, float], float],
    ):
        """
        2次元テスト関数の初期化
        
        Args:
            name: 関数の名前
            f: 関数値
            f_x: x方向の1階偏導関数
            f_y: y方向の1階偏導関数
            f_xx: x方向の2階偏導関数
            f_yy: y方向の2階偏導関数
            f_xy: 混合偏導関数
        """
        self.name = name
        self.f = f
        self.f_x = f_x
        self.f_y = f_y
        self.f_xx = f_xx
        self.f_yy = f_yy
        self.f_xy = f_xy


def create_test_functions() -> List[TestFunction2D]:
    """2次元テスト関数のリストを作成"""
    test_functions = []
    
    # 既存の1次元テスト関数を活用
    one_d_functions = TestFunctionFactory.create_standard_functions()
    
    # 追加する固有の2次元テスト関数
    
    # 1. 2次元ガウス関数
    def gaussian(x, y):
        return cp.exp(-(x**2 + y**2))

    def gaussian_x(x, y):
        return -2 * x * cp.exp(-(x**2 + y**2))

    def gaussian_y(x, y):
        return -2 * y * cp.exp(-(x**2 + y**2))

    def gaussian_xx(x, y):
        return (-2 + 4 * x**2) * cp.exp(-(x**2 + y**2))

    def gaussian_yy(x, y):
        return (-2 + 4 * y**2) * cp.exp(-(x**2 + y**2))

    def gaussian_xy(x, y):
        return 4 * x * y * cp.exp(-(x**2 + y**2))

    test_functions.append(
        TestFunction2D(
            "Gaussian",
            gaussian,
            gaussian_x,
            gaussian_y,
            gaussian_xx,
            gaussian_yy,
            gaussian_xy,
        )
    )

    # 2. 2次元サイン関数
    def sin_function(x, y):
        return cp.sin(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_x(x, y):
        return cp.pi * cp.cos(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_y(x, y):
        return cp.pi * cp.sin(cp.pi * x) * cp.cos(cp.pi * y)

    def sin_function_xx(x, y):
        return -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_yy(x, y):
        return -(cp.pi**2) * cp.sin(cp.pi * x) * cp.sin(cp.pi * y)

    def sin_function_xy(x, y):
        return (cp.pi**2) * cp.cos(cp.pi * x) * cp.cos(cp.pi * y)

    test_functions.append(
        TestFunction2D(
            "Sine",
            sin_function,
            sin_function_x,
            sin_function_y,
            sin_function_xx,
            sin_function_yy,
            sin_function_xy,
        )
    )
    
    # 1次元関数を2次元に変換
    for tf1d in one_d_functions:
        # 積形式 (x,y両方に依存する関数)
        if tf1d.name in ["Sine", "Cosine", "QuadPoly"]:  # 代表的な関数のみ変換
            adapter = TestFunction2DAdapter(tf1d, "product")
            test_functions.append(
                TestFunction2D(
                    f"{tf1d.name}Product",
                    adapter.f,
                    adapter.f_x,
                    adapter.f_y,
                    adapter.f_xx,
                    adapter.f_yy,
                    adapter.f_xy
                )
            )
    
    return test_functions
