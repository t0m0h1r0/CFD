from .base import BaseScaling

class NoScaling(BaseScaling):
    """スケーリングを行わないデフォルトプラグイン"""
    
    def scale(self, A, b):
        """スケーリングを行わず、元のAとbを返す"""
        return A, b, {}
    
    def unscale(self, x, scale_info):
        """アンスケールを行わず、元のxを返す"""
        return x
    
    @property
    def name(self):
        return "NoScaling"
    
    @property
    def description(self):
        return "スケーリングを行いません（恒等スケーリング）"