# equation/base.py
from abc import ABC, abstractmethod
import cupy as cp  # NumPyではなくCuPyを使用
from typing import Dict, Optional

class Equation(ABC):
    """差分方程式の基底クラス"""
    
    @abstractmethod
    def get_stencil_coefficients(self, i: int, n: int, h: float) -> Dict[int, cp.ndarray]:
        """
        グリッド点iにおけるステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            n: グリッド点の総数
            h: グリッド間隔
            
        Returns:
            {offset: coeffs, ...} の辞書
            offsetはグリッド点iからの相対位置（i+offset）
            coeffsは [psi, psi', psi'', psi'''] に対応する4成分ベクトル
        """
        pass
    
    @abstractmethod
    def get_rhs(self, i: int, n: int, h: float) -> float:
        """方程式の右辺を返す"""
        pass
    
    @abstractmethod
    def is_valid_at(self, i: int, n: int) -> bool:
        """方程式がグリッド点iに適用可能かを判定"""
        pass