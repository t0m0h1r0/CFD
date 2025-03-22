"""
基本テスト関数モジュール

このモジュールはCCD法で使用するテスト関数の基底クラスを提供します。
1D, 2D両方のテスト関数をサポートする統一インターフェースを定義します。
"""

from abc import ABC, abstractmethod

class BaseTestFunction(ABC):
    """テスト関数の抽象基底クラス"""
    
    def __init__(self, name: str):
        """
        テスト関数の初期化
        
        Args:
            name: 関数名
        """
        self.name = name
    
    @abstractmethod
    def f(self, *args) -> float:
        """
        関数値を評価
        
        Args:
            *args: 座標値 (xまたはx,y)
            
        Returns:
            関数値
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        関数の次元を取得
        
        Returns:
            関数の次元数 (1または2)
        """
        pass
