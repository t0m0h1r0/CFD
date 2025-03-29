# preconditioner/base.py
"""
前処理器の基底クラス

このモジュールは、反復法で使用する前処理行列Mを計算するための
基底クラスを定義します。
"""

from abc import ABC, abstractmethod

class BasePreconditioner(ABC):
    """前処理器の抽象基底クラス"""
    
    def __init__(self):
        """初期化"""
        self.M = None  # 前処理行列/演算子
    
    @abstractmethod
    def setup(self, A):
        """
        行列Aに対して前処理行列Mを計算
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        pass
    
    def __call__(self, b):
        """
        前処理を適用 (M⁻¹b を計算)
        
        Args:
            b: ベクトル
            
        Returns:
            前処理したベクトル
        """
        if self.M is None:
            return b
        return self.M @ b
    
    @property
    def name(self):
        """前処理器の名前"""
        return self.__class__.__name__
    
    @property
    def description(self):
        """前処理器の説明"""
        return "抽象基底クラス"