from abc import ABC, abstractmethod
import cupy as cp

class BaseScaling(ABC):
    """行列スケーリング手法の基底クラス
    
    SOLID原則に基づいたスケーリング手法のインターフェース定義：
    - 単一責任の原則: 各スケーリング手法は行列のスケーリングという単一の責任を持つ
    - オープン・クローズドの原則: 新しいスケーリング手法を追加可能だが、既存コードの修正は不要
    - リスコフの置換原則: どのスケーリング手法も互いに置き換え可能
    - インターフェース分離の原則: 必要最小限のインターフェースのみを定義
    - 依存性逆転の原則: 高レベルモジュール(ソルバー)は抽象(このインターフェース)に依存
    """
    
    @abstractmethod
    def scale(self, A, b):
        """行列Aと右辺ベクトルbをスケーリングする
        
        Args:
            A: システム行列 (スパース)
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
                scaled_A: スケーリングされたシステム行列
                scaled_b: スケーリングされた右辺ベクトル
                scale_info: 解をアンスケールするための情報を含む辞書
        """
        pass
    
    @abstractmethod
    def unscale(self, x, scale_info):
        """スケーリング情報を使用して解ベクトルをアンスケールする
        
        Args:
            x: 解ベクトル
            scale_info: scaleメソッドから返されたスケーリング情報
            
        Returns:
            unscaled_x: アンスケールされた解ベクトル
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """スケーリング手法の名前を返す"""
        pass
    
    @property
    @abstractmethod
    def description(self):
        """スケーリング手法の説明を返す"""
        pass