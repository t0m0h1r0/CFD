"""
テスト関数ファクトリーモジュール

このモジュールはCCD法で使用するテスト関数を生成する統一されたインターフェースを提供します。
1D, 2D両方のテスト関数を作成するファクトリークラスを定義します。
"""

import sympy as sp
from typing import List, Dict, Tuple, Any, Callable, Optional, Union

from base_test_function import BaseTestFunction
from test_function1d import TestFunction1D, TestFunction1DFactory
from test_function2d import TestFunction2D, TestFunction2DFactory

class TestFunctionFactory:
    """統合されたテスト関数生成ファクトリークラス"""
    
    @staticmethod
    def create_standard_1d_functions() -> List[TestFunction1D]:
        """
        標準的な1Dテスト関数を作成
        
        Returns:
            1Dテスト関数のリスト
        """
        return TestFunction1DFactory.create_standard_functions()
    
    @staticmethod
    def create_standard_2d_functions() -> List[TestFunction2D]:
        """
        標準的な2Dテスト関数を作成
        
        Returns:
            2Dテスト関数のリスト
        """
        return TestFunction2DFactory.create_standard_functions()
    
    @staticmethod
    def get_function_by_name(name: str, dimension: Optional[int] = None) -> Optional[BaseTestFunction]:
        """
        名前と次元からテスト関数を取得
        
        Args:
            name: 関数名
            dimension: 次元（1または2）、Noneの場合は両方を検索
            
        Returns:
            対応するテスト関数、見つからない場合はNone
        """
        if dimension == 1 or dimension is None:
            # 1D関数を検索
            func_1d = TestFunction1DFactory.get_function_by_name(name)
            if func_1d:
                return func_1d
                
        if dimension == 2 or dimension is None:
            # 2D関数を検索
            func_2d = TestFunction2DFactory.get_function_by_name(name)
            if func_2d:
                return func_2d
        
        return None
    
    @staticmethod
    def create_1d_to_2d(func_name: str, method: str = 'product', suffix: str = '') -> Optional[TestFunction2D]:
        """
        1Dテスト関数から2Dテスト関数を作成
        
        Args:
            func_name: 1D関数名
            method: 拡張メソッド ('product', 'sum', 'radial')
            suffix: 名前に追加する接尾辞
            
        Returns:
            2Dテスト関数、1D関数が見つからない場合はNone
        """
        # 1D関数を取得
        func_1d = TestFunction1DFactory.get_function_by_name(func_name)
        if not func_1d:
            return None
            
        # 2D関数に変換
        return TestFunction2D.from_1d(func_1d, method, suffix)
    
    @staticmethod
    def list_available_functions(dimension: Optional[int] = None) -> Dict[str, List[str]]:
        """
        利用可能なテスト関数名のリストを取得
        
        Args:
            dimension: 次元（1または2）、Noneの場合は両方
            
        Returns:
            次元ごとの関数名リストを含む辞書
        """
        result = {}
        
        if dimension == 1 or dimension is None:
            funcs_1d = TestFunction1DFactory.create_standard_functions()
            result['1D'] = [func.name for func in funcs_1d]
            
        if dimension == 2 or dimension is None:
            funcs_2d = TestFunction2DFactory.create_standard_functions()
            result['2D'] = [func.name for func in funcs_2d]
            
        return result
