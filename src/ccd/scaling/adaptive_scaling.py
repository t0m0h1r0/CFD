"""
Adaptive scaling implementation that selects the best scaling method.
"""

import numpy as np
from .base import BaseScaling
from .row_scaling import RowScaling
from .column_scaling import ColumnScaling
from .diagonal_scaling import DiagonalScaling
from .ruiz_scaling import RuizScaling
from .no_scaling import NoScaling


class AdaptiveScaling(BaseScaling):
    """Adaptively choose the best scaling method based on matrix properties."""
    
    def __init__(self, strategy="auto"):
        """
        Initialize adaptive scaling.
        
        Args:
            strategy: Strategy for choosing scaling method
                     - "auto": Automatically choose based on matrix properties
                     - "sparsity": Choose based on matrix sparsity
                     - "condition": Choose based on approximate condition number
        """
        super().__init__()
        self.strategy = strategy
        self.selected_scaler = None
        self.scalers = {
            "none": NoScaling(),
            "row": RowScaling(),
            "column": ColumnScaling(),
            "diagonal": DiagonalScaling(),
            "ruiz": RuizScaling(max_iter=3)  # Limit iterations for efficiency
        }
    
    def scale(self, A, b):
        """
        Scale matrix A and right-hand side vector b using the best scaling method.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # Analyze matrix and choose best scaling method
        self.selected_scaler = self._choose_scaling_method(A)
        print(f"Adaptive scaling selected: {self.selected_scaler.name}")
        
        # Apply selected scaling
        scaled_A, scaled_b, scale_info = self.selected_scaler.scale(A, b)
        
        # Add selected scaler name to scale_info
        scale_info["selected_scaler"] = self.selected_scaler.name
        
        return scaled_A, scaled_b, scale_info
    
    def unscale(self, x, scale_info):
        """
        Unscale the solution vector using the selected scaling method.
        
        Args:
            x: Solution vector
            scale_info: Scaling information
            
        Returns:
            unscaled_x: Unscaled solution
        """
        # Get selected scaler name
        scaler_name = scale_info.get("selected_scaler")
        
        # Use the correct scaler
        if scaler_name and scaler_name in self.scalers:
            scaler = self.scalers[scaler_name.split('_')[0].lower()]
        elif self.selected_scaler:
            scaler = self.selected_scaler
        else:
            # Default to no scaling
            return x
        
        # Unscale using the selected scaler
        return scaler.unscale(x, scale_info)
    
    def scale_b_only(self, b, scale_info):
        """
        Scale the right-hand side vector using the selected scaling method.
        
        Args:
            b: Right-hand side vector
            scale_info: Scaling information
            
        Returns:
            scaled_b: Scaled right-hand side
        """
        # Get selected scaler name
        scaler_name = scale_info.get("selected_scaler")
        
        # Use the correct scaler
        if scaler_name and scaler_name in self.scalers:
            scaler = self.scalers[scaler_name.split('_')[0].lower()]
        elif self.selected_scaler:
            scaler = self.selected_scaler
        else:
            # Default to no scaling
            return b
        
        # Scale using the selected scaler
        return scaler.scale_b_only(b, scale_info)
    
    def _choose_scaling_method(self, A):
        """
        Choose the best scaling method based on matrix properties.
        
        Args:
            A: System matrix
            
        Returns:
            scaler: Selected scaling method
        """
        if self.strategy == "sparsity":
            return self._choose_by_sparsity(A)
        elif self.strategy == "condition":
            return self._choose_by_condition(A)
        else:  # "auto" or any other value
            return self._choose_automatic(A)
    
    def _choose_by_sparsity(self, A):
        """
        Choose scaling method based on matrix sparsity.
        
        Args:
            A: System matrix
            
        Returns:
            scaler: Selected scaling method
        """
        # Check if A is a sparse matrix
        is_sparse = self.is_sparse(A)
        
        # Get matrix size
        n_rows, n_cols = A.shape
        
        if is_sparse:
            # For large sparse matrices, prefer row scaling (memory efficient)
            if n_rows > 10000 or n_cols > 10000:
                return self.scalers["row"]
            else:
                # For moderate-sized sparse matrices, use Ruiz scaling
                return self.scalers["ruiz"]
        else:
            # For small dense matrices, use diagonal scaling
            if n_rows < 1000 and n_cols < 1000:
                return self.scalers["diagonal"]
            else:
                # For larger dense matrices, use row scaling
                return self.scalers["row"]
    
    def _choose_by_condition(self, A):
        """
        Choose scaling method based on approximate condition number.
        
        Args:
            A: System matrix
            
        Returns:
            scaler: Selected scaling method
        """
        # Check if A is square
        n_rows, n_cols = A.shape
        if n_rows != n_cols:
            # For non-square matrices, use row scaling
            return self.scalers["row"]
        
        # Estimate condition by comparing row norms
        row_norms = self.compute_row_norms(A, "inf")
        if row_norms.max() == 0:
            return self.scalers["none"]
            
        row_ratio = row_norms.max() / row_norms[row_norms > 0].min()
        
        # Check column norms (using our helper)
        col_norms = self._compute_column_norms(A, "inf")
        if col_norms.max() == 0:
            return self.scalers["none"]
            
        col_ratio = col_norms.max() / col_norms[col_norms > 0].min()
        
        # Choose based on the estimated conditioning
        if max(row_ratio, col_ratio) > 1e6:
            # Very ill-conditioned - use Ruiz scaling
            return self.scalers["ruiz"]
        elif max(row_ratio, col_ratio) > 1e3:
            # Moderately ill-conditioned - use diagonal scaling
            return self.scalers["diagonal"]
        elif row_ratio > col_ratio * 10:
            # Row scaling needed more
            return self.scalers["row"]
        elif col_ratio > row_ratio * 10:
            # Column scaling needed more
            return self.scalers["column"]
        else:
            # Balanced conditioning - use simple row scaling
            return self.scalers["row"]
    
    def _choose_automatic(self, A):
        """
        Automatically choose scaling method based on matrix properties.
        
        Args:
            A: System matrix
            
        Returns:
            scaler: Selected scaling method
        """
        # Check if A is a sparse matrix
        is_sparse = self.is_sparse(A)
        
        # Get matrix size
        n_rows, n_cols = A.shape
        
        # For very small matrices, don't bother scaling
        if n_rows < 10 and n_cols < 10:
            return self.scalers["none"]
        
        # For non-square matrices, use row scaling
        if n_rows != n_cols:
            return self.scalers["row"]
        
        # Analyze matrix structure
        if is_sparse:
            # Check sparsity pattern
            if self._is_diagonally_dominant(A):
                # Diagonally dominant matrices work well with diagonal scaling
                return self.scalers["diagonal"]
            else:
                # Otherwise use Ruiz for sparse matrices with moderate size
                if n_rows < 5000:
                    return self.scalers["ruiz"]
                else:
                    # For very large sparse matrices, use row scaling (memory efficient)
                    return self.scalers["row"]
        else:
            # For dense matrices, check conditioning
            return self._choose_by_condition(A)
    
    def _is_diagonally_dominant(self, A):
        """
        Check if matrix A is diagonally dominant.
        
        Args:
            A: System matrix
            
        Returns:
            is_dominant: True if matrix is diagonally dominant
        """
        # Need square matrix
        if A.shape[0] != A.shape[1]:
            return False
        
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Sparse matrix implementation
            if hasattr(A, "tocsr"):
                A_csr = A.tocsr()
            else:
                A_csr = A
            
            n = A.shape[0]
            # Test a sample of rows for efficiency
            sample_size = min(n, 100)
            sample_rows = np.random.choice(n, sample_size, replace=False)
            
            dominant_count = 0
            for i in sample_rows:
                start, end = A_csr.indptr[i], A_csr.indptr[i+1]
                row_data = A_csr.data[start:end]
                row_indices = A_csr.indices[start:end]
                
                # Find diagonal element
                diag_idx = np.where(row_indices == i)[0]
                if len(diag_idx) == 0:
                    continue
                
                diag_val = abs(row_data[diag_idx[0]])
                
                # Sum of off-diagonal elements
                off_diag_sum = np.sum(np.abs(row_data)) - diag_val
                
                if diag_val >= off_diag_sum:
                    dominant_count += 1
            
            # Consider diagonally dominant if majority of sampled rows are dominant
            return dominant_count > sample_size // 2
        else:
            # Dense matrix implementation
            diag = np.abs(np.diag(A))
            row_sums = np.sum(np.abs(A), axis=1) - diag
            
            # Check if diagonal elements are >= sum of off-diagonal elements
            return np.mean(diag >= row_sums) > 0.5
    
    def _compute_column_norms(self, A, norm_type="inf"):
        """
        Compute column norms of matrix A.
        
        Args:
            A: Matrix
            norm_type: Type of norm to use ("inf", "1", "2")
            
        Returns:
            col_norms: Array of column norms
        """
        # Check if A is a sparse matrix
        if self.is_sparse(A):
            # Convert to CSC for column operations
            if hasattr(A, "tocsc"):
                A_csc = A.tocsc()
            else:
                A_csc = A
                
            n_cols = A.shape[1]
            col_norms = np.zeros(n_cols)
            
            # For each column
            for j in range(n_cols):
                # Get column slice
                start, end = A_csc.indptr[j], A_csc.indptr[j+1]
                if start < end:
                    col_data = A_csc.data[start:end]
                    if norm_type == "inf":
                        col_norms[j] = np.max(np.abs(col_data))
                    elif norm_type == "1":
                        col_norms[j] = np.sum(np.abs(col_data))
                    else:  # default to "2"
                        col_norms[j] = np.sqrt(np.sum(col_data * col_data))
            
            return col_norms
        else:
            # Handle dense matrix
            if norm_type == "inf":
                return np.max(np.abs(A), axis=0)
            elif norm_type == "1":
                return np.sum(np.abs(A), axis=0)
            else:  # default to "2"
                return np.sqrt(np.sum(A * A, axis=0))
    
    @property
    def name(self):
        return "AdaptiveScaling"
    
    @property
    def description(self):
        """Return the description of the scaling method."""
        return f"Adaptive scaling using {self.strategy} strategy."