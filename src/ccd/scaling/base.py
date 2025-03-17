"""
Base class for matrix scaling implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional
import numpy as np

# Import our array utilities
from .array_utils import ArrayBackend


class BaseScaling(ABC):
    """Base class for matrix scaling techniques."""
    
    def __init__(self, backend='numpy'):
        """
        Initialize scaling with specified backend.
        
        Args:
            backend: 'numpy', 'cupy', or 'jax'
        """
        self.array_utils = ArrayBackend(backend)
    
    @abstractmethod
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Scale matrix A and right-hand side vector b.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        pass
    
    @abstractmethod
    def unscale(self, x, scale_info: Dict[str, Any]):
        """
        Unscale the solution vector using scaling information.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        pass
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """
        Scale only the right-hand side vector (optimization for multiple solves).
        
        Args:
            b: Right-hand side vector
            scale_info: Scaling information
            
        Returns:
            scaled_b: Scaled right-hand side
        """
        # Default implementation: use row scale if available
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
            
        # Otherwise, each subclass should implement efficiently
        return b
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the scaling method."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of the scaling method."""
        pass
    
    def set_backend(self, backend: str):
        """
        Change the computational backend.
        
        Args:
            backend: 'numpy', 'cupy', or 'jax'
        """
        self.array_utils = ArrayBackend(backend)
