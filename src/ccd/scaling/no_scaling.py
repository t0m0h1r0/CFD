"""
No scaling implementation (identity scaling).
"""

from .base import BaseScaling

class NoScaling(BaseScaling):
    """No scaling (identity scaling)"""
    
    def __init__(self):
        super().__init__()
        
    @property
    def name(self):
        return "NoScaling"
        
    @property
    def description(self):
        return "No scaling (identity scaling)"
    
    def scale(self, A, b):
        """
        No scaling - return original matrix and vector
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Tuple of (A, b, {})
        """
        return A, b, {}
    
    def unscale(self, x, scale_info):
        """
        No unscaling - return original solution
        
        Args:
            x: Solution vector
            scale_info: Empty dictionary
            
        Returns:
            Original solution vector
        """
        return x
    
    def scale_b_only(self, b, scale_info):
        """
        No scaling of right-hand side
        
        Args:
            b: Right-hand side vector
            scale_info: Empty dictionary
            
        Returns:
            Original right-hand side vector
        """
        return b