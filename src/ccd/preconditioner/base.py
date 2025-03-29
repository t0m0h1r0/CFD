"""
前処理手法の基底クラス

このモジュールは、CCD法の反復ソルバーの収束を加速するための
前処理手法の基底クラスを定義します。
"""

from abc import ABC, abstractmethod

class BasePreconditioner(ABC):
    """前処理手法の抽象基底クラス"""
    
    def __init__(self):
        """初期化"""
        self.M = None  # 前処理行列/演算子
    
    @abstractmethod
    def setup(self, A):
        """
        行列Aに対して前処理を設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        pass
    
    def __call__(self, b):
        """
        前処理を適用 (M*x = bを解く)
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            前処理適用後のベクトル
        """
        # デフォルト実装: Mは近似逆行列と仮定
        if self.M is None:
            return b
        return self.M @ b
    
    @property
    def matrix(self):
        """前処理行列または演算子を返す"""
        return self.M
    
    @property
    @abstractmethod
    def name(self):
        """前処理手法の名前を返す"""
        pass
    
    @property
    @abstractmethod
    def description(self):
        """前処理手法の説明を返す"""
        pass
    
    def is_sparse(self, matrix):
        """行列が疎行列かどうか判定"""
        return hasattr(matrix, 'format') or 'sparse' in str(type(matrix))