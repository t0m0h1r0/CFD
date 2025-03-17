"""
Adaptive matrix scaling implementation

This scaling method automatically selects and combines multiple scaling strategies
based on matrix properties. It analyzes the input matrix and chooses the most
appropriate scaling approach to optimize numerical stability and performance.
"""

from typing import Dict, Any, Tuple, Union, List
import cupy as cp
import cupyx.scipy.sparse as sp
import logging
from .base import BaseScaling
from .row_scaling import RowScaling
from .column_scaling import ColumnScaling
from .symmetric_scaling import SymmetricScaling
from .equilibration_scaling import EquilibrationScaling


class AdaptiveScaling(BaseScaling):
    """
    Adaptive scaling technique that automatically selects optimal scaling strategy.
    
    This method analyzes matrix properties to determine which scaling approach
    (row, column, symmetric, or equilibration) will be most effective, then
    applies the selected strategy or a combination of strategies.
    """
    
    def __init__(self):
        """Initialize the adaptive scaling algorithm."""
        self._logger = logging.getLogger(__name__)
        self.selected_strategy = None
        self.strategy_info = {}
        
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        Analyze matrix A and apply the optimal scaling strategy.
        
        Args:
            A: System matrix to scale
            b: Right-hand side vector
            
        Returns:
            Tuple of (scaled_A, scaled_b, scaling_info)
        """
        m, n = A.shape
        
        # Analyze matrix properties
        properties = self._analyze_matrix(A)
        
        # Select scaling strategy based on matrix properties
        strategy, strategy_name = self._select_strategy(properties)
        
        # Apply selected scaling strategy
        scaled_A, scaled_b, scale_info = strategy.scale(A, b)
        
        # Store information about selected strategy
        self.selected_strategy = strategy_name
        self.strategy_info = properties
        
        # Extend scale_info with metadata
        scale_info['selected_strategy'] = strategy_name
        scale_info['matrix_properties'] = properties
        
        return scaled_A, scaled_b, scale_info
    
    def _analyze_matrix(self, A) -> Dict[str, Any]:
        """
        Analyze matrix properties to guide scaling strategy selection.
        
        Args:
            A: Matrix to analyze
            
        Returns:
            Dictionary of matrix properties
        """
        properties = {}
        m, n = A.shape
        
        # Basic properties
        properties['is_square'] = m == n
        properties['size'] = max(m, n)
        properties['shape'] = (m, n)
        
        # Sparsity analysis
        if hasattr(A, 'nnz'):
            properties['nnz'] = A.nnz
            properties['density'] = A.nnz / (m * n)
        else:
            if hasattr(A, 'toarray'):
                nnz = cp.count_nonzero(A.toarray())
            else:
                nnz = cp.count_nonzero(A)
            properties['nnz'] = nnz
            properties['density'] = nnz / (m * n)
        
        # Symmetry check (for square matrices)
        if properties['is_square']:
            if hasattr(A, 'toarray'):
                # For sparse matrices, check a sample of elements
                sample_size = min(1000, m)
                indices = cp.random.choice(m, sample_size, replace=False)
                
                symmetry_violations = 0
                for i in indices:
                    for j in indices:
                        if i < j:  # Only check upper triangle
                            if A[i, j] != A[j, i]:
                                symmetry_violations += 1
                
                properties['is_symmetric'] = symmetry_violations < sample_size * 0.05
            else:
                # For dense matrices
                properties['is_symmetric'] = cp.allclose(A, A.T)
        else:
            properties['is_symmetric'] = False
            
        # Analyze row and column variation
        row_norms = self._compute_row_norms(A)
        col_norms = self._compute_column_norms(A)
        
        properties['row_norm_max'] = float(cp.max(row_norms))
        properties['row_norm_min'] = float(cp.min(row_norms))
        properties['row_norm_ratio'] = float(properties['row_norm_max'] / max(properties['row_norm_min'], 1e-15))
        
        properties['col_norm_max'] = float(cp.max(col_norms))
        properties['col_norm_min'] = float(cp.min(col_norms))
        properties['col_norm_ratio'] = float(properties['col_norm_max'] / max(properties['col_norm_min'], 1e-15))
        
        # Diagonal dominance (for square matrices)
        if properties['is_square']:
            try:
                diag = A.diagonal()
                diag_abs = cp.abs(diag)
                
                if hasattr(A, 'format') and A.format == 'csr':
                    # For CSR matrices
                    row_sums = cp.zeros_like(diag)
                    for i in range(m):
                        start, end = A.indptr[i], A.indptr[i+1]
                        row_data = cp.abs(A.data[start:end])
                        row_sums[i] = cp.sum(row_data) - diag_abs[i]
                else:
                    # General case
                    if hasattr(A, 'toarray'):
                        A_abs = cp.abs(A.toarray())
                    else:
                        A_abs = cp.abs(A)
                    row_sums = cp.sum(A_abs, axis=1) - diag_abs
                
                # Ratio of diagonal to off-diagonal elements
                with cp.errstate(divide='ignore', invalid='ignore'):
                    diag_ratios = diag_abs / cp.maximum(row_sums, 1e-15)
                    diag_ratios = cp.where(cp.isfinite(diag_ratios), diag_ratios, 0)
                
                properties['diag_dominance'] = float(cp.mean(diag_ratios))
            except Exception as e:
                self._logger.warning(f"Error analyzing diagonal dominance: {e}")
                properties['diag_dominance'] = 0.0
        
        return properties
    
    def _select_strategy(self, properties) -> Tuple[BaseScaling, str]:
        """
        Select optimal scaling strategy based on matrix properties.
        
        Args:
            properties: Dictionary of matrix properties
            
        Returns:
            Tuple of (scaling_strategy, strategy_name)
        """
        # Default to equilibration for severely ill-conditioned matrices
        if (properties.get('row_norm_ratio', 0) > 1e8 or 
            properties.get('col_norm_ratio', 0) > 1e8):
            return EquilibrationScaling(max_iterations=5), "EquilibrationScaling"
        
        # For symmetric matrices, symmetric scaling is often best
        if properties.get('is_symmetric', False):
            return SymmetricScaling(), "SymmetricScaling"
        
        # For diagonally dominant matrices, row scaling works well
        if properties.get('diag_dominance', 0) > 0.8:
            return RowScaling(), "RowScaling"
        
        # Compare row and column variations
        row_ratio = properties.get('row_norm_ratio', 1.0)
        col_ratio = properties.get('col_norm_ratio', 1.0)
        
        if row_ratio > 10 * col_ratio:
            return RowScaling(), "RowScaling"
        elif col_ratio > 10 * row_ratio:
            return ColumnScaling(), "ColumnScaling"
        
        # Default to equilibration for balanced improvement
        return EquilibrationScaling(), "EquilibrationScaling"
    
    def _compute_row_norms(self, A):
        """Compute row norms of matrix A"""
        m = A.shape[0]
        row_norms = cp.zeros(m)
        
        if hasattr(A, 'format') and A.format == 'csr':
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_norms[i] = cp.linalg.norm(A.data[start:end])
        else:
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = cp.linalg.norm(row)
        
        return row_norms
    
    def _compute_column_norms(self, A):
        """Compute column norms of matrix A"""
        n = A.shape[1]
        col_norms = cp.zeros(n)
        
        if hasattr(A, 'format') and A.format == 'csc':
            for j in range(n):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    col_norms[j] = cp.linalg.norm(A.data[start:end])
        else:
            for j in range(n):
                col = A[:, j]
                if hasattr(col, 'toarray'):
                    col = col.toarray().flatten()
                col_norms[j] = cp.linalg.norm(col)
        
        return col_norms
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        Unscale the solution vector.
        
        Args:
            x: Solution vector to unscale
            scale_info: Scaling information from the scale method
            
        Returns:
            Unscaled solution vector
        """
        # Extract the selected strategy
        selected_strategy = scale_info.get('selected_strategy')
        
        # Forward to the appropriate unscaling method
        if selected_strategy == "RowScaling":
            return RowScaling().unscale(x, scale_info)
        elif selected_strategy == "ColumnScaling":
            return ColumnScaling().unscale(x, scale_info)
        elif selected_strategy == "SymmetricScaling":
            return SymmetricScaling().unscale(x, scale_info)
        elif selected_strategy == "EquilibrationScaling":
            return EquilibrationScaling().unscale(x, scale_info)
        
        # Default case: column scaling may have been applied
        col_scale = scale_info.get('col_scale')
        if col_scale is not None:
            return x / col_scale
        return x
    
    def scale_b_only(self, b: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
        """
        Scale only the right-hand side vector.
        
        Args:
            b: Right-hand side vector to scale
            scale_info: Scaling information from the scale method
            
        Returns:
            Scaled right-hand side vector
        """
        # Extract the selected strategy
        selected_strategy = scale_info.get('selected_strategy')
        
        # Forward to the appropriate scaling method
        if selected_strategy == "RowScaling":
            return RowScaling().scale_b_only(b, scale_info)
        elif selected_strategy == "ColumnScaling":
            return ColumnScaling().scale_b_only(b, scale_info)
        elif selected_strategy == "SymmetricScaling":
            return SymmetricScaling().scale_b_only(b, scale_info)
        elif selected_strategy == "EquilibrationScaling":
            return EquilibrationScaling().scale_b_only(b, scale_info)
        
        # Default case: row scaling
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
        return b
    
    @property
    def name(self) -> str:
        return "AdaptiveScaling"
    
    @property
    def description(self) -> str:
        return "Adaptive scaling that automatically selects the optimal strategy"
