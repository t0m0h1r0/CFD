"""
スケーリングを行わないデフォルト実装
"""

from typing import Dict, Any, Tuple
from .base import BaseScaling


class NoScaling(BaseScaling):
    """スケーリングを行わないデフォルトプラグイン"""
    
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """スケーリングを行わず、元のAとbを返す"""
        return A, b, {}
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """アンスケールを行わず、元のxを返す"""
        return x
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """右辺ベクトルのスケーリングを行わず、元のbを返す"""
        return b
    
    @property
    def name(self) -> str:
        return "NoScaling"
    
    @property
    def description(self) -> str:
        return "スケーリングを行いません（恒等スケーリング）"