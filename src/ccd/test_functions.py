"""
テスト関数モジュール (互換性用)

後方互換性のために、リファクタリングされたテスト関数モジュールからのインポートを提供します。
既存のコードの動作に影響を与えないように、元のクラス名とインターフェースを維持しています。
"""

from base_test_function import BaseTestFunction
from test_function1d import TestFunction1D, TestFunction1DFactory
from test_function2d import TestFunction2D, TestFunction2DFactory
from test_function_factory import TestFunctionFactory

# 後方互換性のためのクラス名
class TestFunction(TestFunction1D):
    """元のTestFunctionクラスを互換性のために維持"""
    
    @classmethod
    def create_1d(cls, name: str, expr):
        """1Dテスト関数を作成"""
        return TestFunction1D(name, expr)
    
    @classmethod
    def create_2d(cls, name: str, expr):
        """2Dテスト関数を作成"""
        return TestFunction2D(name, expr)
    
    @classmethod
    def from_1d_to_2d(cls, func_1d, method='product', suffix=''):
        """1Dテスト関数から2Dテスト関数を作成"""
        return TestFunction2D.from_1d(func_1d, method, suffix)

# 後方互換性のためにすべてをエクスポート
__all__ = [
    'BaseTestFunction', 
    'TestFunction', 
    'TestFunction1D', 
    'TestFunction2D',
    'TestFunctionFactory',
    'TestFunction1DFactory',
    'TestFunction2DFactory'
]