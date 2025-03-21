"""
No scaling implementation - baseline for comparison.
"""

import numpy as np
from .base import BaseScaling


class NoScaling(BaseScaling):
    """No scaling is applied, identity scaling."""
    
    def scale(self, A, b):
        """
        Return A and b unchanged.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (A, b, scale_info)
        """
        # Return original matrix and vector with empty scaling info
        return A, b, {}
    
    def unscale(self, x, scale_info):
        """
        Return x unchanged.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Original solution (unchanged)
        """
        return x
    
    def scale_b_only(self, b, scale_info):
        """
        Return b unchanged.
        
        Args:
            b: Right-hand side vector
            scale_info: Scaling information
            
        Returns:
            scaled_b: Original vector (unchanged)
        """
        return b
    
    @property
    def name(self):
        """Return the name of the scaling method."""
        return "NoScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return "No scaling is applied (identity scaling)."