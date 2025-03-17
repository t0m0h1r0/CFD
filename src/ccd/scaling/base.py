"""
行列スケーリングモジュールの基底クラス
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp

class BaseScaling(ABC):
    """行列スケーリング手法の基底クラス"""
    
    @abstractmethod
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        行列Aと右辺ベクトルbをスケーリングする
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        pass
    
    @abstractmethod
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        スケーリング情報を使用して解ベクトルをアンスケールする
        
        Args:
            x: 解ベクトル
            scale_info: スケーリング情報
            
        Returns:
            unscaled_x: アンスケールされた解ベクトル
        """
        pass
    
    def scale_b_only(self, b: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        右辺ベクトルbのみをスケーリング（最適化用）
        
        Args:
            b: 右辺ベクトル
            scale_info: スケーリング情報
            
        Returns:
            スケーリングされた右辺ベクトル
        """
        # デフォルト実装：行スケール係数があればそれを使用
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
            
        # それ以外は各サブクラスで効率的に実装
        return b
    
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