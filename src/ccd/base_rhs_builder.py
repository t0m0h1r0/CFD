"""
高精度コンパクト差分法 (CCD) 用の右辺ベクトル構築モジュール

このモジュールは、ポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築する基底クラスを提供します。
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union, List


class RHSBuilder(ABC):
    """右辺ベクトルを構築する抽象基底クラス"""
    
    def __init__(self, system, grid, enable_dirichlet=True, enable_neumann=True):
        """
        初期化
        
        Args:
            system: 方程式システム
            grid: グリッドオブジェクト
            enable_dirichlet: ディリクレ境界条件を有効にするフラグ
            enable_neumann: ノイマン境界条件を有効にするフラグ
        """
        self.system = system
        self.grid = grid
        self.enable_dirichlet = enable_dirichlet
        self.enable_neumann = enable_neumann
    
    @abstractmethod
    def build_rhs_vector(self, f_values=None, **boundary_values):
        """
        右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値
            **boundary_values: 境界値の辞書
            
        Returns:
            右辺ベクトル（NumPy配列）
        """
        pass
    
    def _to_numpy(self, arr):
        """
        CuPy配列をNumPy配列に変換する (必要な場合のみ)
        
        Args:
            arr: 変換する配列
            
        Returns:
            NumPy配列またはスカラー
        """
        if arr is None:
            return None
            
        if hasattr(arr, 'get'):
            return arr.get()
        return arr
