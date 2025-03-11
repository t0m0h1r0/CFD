from typing import Dict, Any, Tuple, Union
import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class NoScaling(BaseScaling):
    """スケーリングを行わないデフォルトプラグイン"""
    
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """スケーリングを行わず、元のAとbを返す"""
        return A, b, {}
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """アンスケールを行わず、元のxを返す"""
        return x
    
    @property
    def name(self) -> str:
        return "NoScaling"
    
    @property
    def description(self) -> str:
        return "スケーリングを行いません（恒等スケーリング）"