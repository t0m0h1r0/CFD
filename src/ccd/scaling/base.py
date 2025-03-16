from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp

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
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """行列Aと右辺ベクトルbをスケーリングする
        
        Args:
            A: システム行列 (GPU上のスパース行列)
            b: 右辺ベクトル (GPU上のベクトル)
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
                scaled_A: スケーリングされたシステム行列
                scaled_b: スケーリングされた右辺ベクトル
                scale_info: 解をアンスケールするための情報を含む辞書
        """
        pass
    
    @abstractmethod
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """スケーリング情報を使用して解ベクトルをアンスケールする
        
        Args:
            x: 解ベクトル (GPU上)
            scale_info: scaleメソッドから返されたスケーリング情報
            
        Returns:
            unscaled_x: アンスケールされた解ベクトル (GPU上)
        """
        pass
    
    def scale_b_only(self, b: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        右辺ベクトルbのみをスケーリング（最適化用）
        
        Args:
            b: 右辺ベクトル (GPU上)
            scale_info: 以前のスケーリングから得られた情報
            
        Returns:
            スケーリングされた右辺ベクトル
        """
        # デフォルト実装：完全なスケーリングを実行して右辺のみ返す
        import cupyx.scipy.sparse as sp
        dummy_A = sp.eye(b.shape[0], format='csr')
        _, b_scaled, _ = self.scale(dummy_A, b)
        return b_scaled
    
    @property
    @abstractmethod
    def name(self) -> str:
        """スケーリング手法の名前を返す"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """スケーリング手法の説明を返す"""
        pass