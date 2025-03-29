"""
Matrix System and Preconditioner Visualizer Module

This module provides functionality to visualize linear system matrices (Ax = b)
and preconditioner matrices (M) to help understand their structure and characteristics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

class MatrixVisualizer:
    """Class for visualizing matrix systems (Ax = b) and preconditioners"""
    
    def __init__(self, output_dir="results"):
        """Initialize the visualizer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Component names by dimension
        self.components = {
            1: ["ψ", "ψ'", "ψ''", "ψ'''"],
            2: ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"],
            3: ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
        }
    
    def visualize(self, A, b, x, exact_x, title, dimension, scaling=None, preconditioner=None):
        """Visualize the matrix system and optionally the preconditioner matrix"""
        # Convert data to NumPy
        data = self._prepare_data(A, b, x, exact_x)
        M_np = self._extract_preconditioner_matrix(preconditioner, A.shape[0] if A is not None else 0)
        
        # Create plot
        has_preconditioner = M_np is not None
        fig, axes = self._create_plot_layout(has_preconditioner)
        
        # Plot data
        self._plot_all_panels(fig, axes, data, M_np, dimension, scaling, preconditioner)
        
        # Add title
        precond_name = self._get_attribute(preconditioner, 'name', 'Unknown')
        precond_info = f", Preconditioner: {precond_name}" if preconditioner else ""
        plt.suptitle(f"{dimension}D {title}" + 
                    (f" (Scaling: {scaling})" if scaling else "") + 
                    precond_info, fontsize=16)
        
        # Save output
        output_path = f"{self.output_dir}/{title}{'_'+scaling if scaling else ''}_matrix.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _prepare_data(self, A, b, x, exact_x):
        """Prepare and convert all data to appropriate NumPy arrays"""
        A_np = self._to_dense_numpy(A)
        b_np = self._reshape_vector(self._to_dense_numpy(b))
        x_np = self._reshape_vector(self._to_dense_numpy(x))
        exact_np = self._reshape_vector(self._to_dense_numpy(exact_x))
        error_np = np.abs(x_np - exact_np) if x_np is not None and exact_np is not None else None
        
        return {'A': A_np, 'b': b_np, 'x': x_np, 'exact': exact_np, 'error': error_np}
    
    def _reshape_vector(self, v):
        """Reshape vector to column format if not None"""
        return v.reshape(-1, 1) if v is not None else None
    
    def _create_plot_layout(self, has_preconditioner):
        """Create appropriate plot layout based on presence of preconditioner"""
        if has_preconditioner:
            fig = plt.figure(figsize=(18, 12))
            gs = plt.GridSpec(2, 3, height_ratios=[3, 2])
            axes = {
                'matrix': fig.add_subplot(gs[0, 0]),
                'precond': fig.add_subplot(gs[0, 1]),
                'solution': fig.add_subplot(gs[0, 2]),
                'error': fig.add_subplot(gs[1, 0]),
                'comparison': fig.add_subplot(gs[1, 1]),
                'stats': fig.add_subplot(gs[1, 2])
            }
        else:
            fig = plt.figure(figsize=(14, 10))
            gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
            axes = {
                'matrix': fig.add_subplot(gs[0, 0]),
                'solution': fig.add_subplot(gs[0, 1]),
                'error': fig.add_subplot(gs[1, 0]),
                'stats': fig.add_subplot(gs[1, 1])
            }
        return fig, axes
    
    def _plot_all_panels(self, fig, axes, data, M_np, dimension, scaling, preconditioner):
        """Plot all panels of the visualization"""
        # Always plot these panels
        self._plot_matrix(axes['matrix'], data['A'], "System Matrix A")
        if data['x'] is not None:
            self._plot_solution_comparison(axes['solution'], data['x'], data['exact'])
        if data['error'] is not None:
            self._plot_error_distribution(axes['error'], data['error'])
        
        # Plot preconditioner-related panels if available
        if M_np is not None:
            self._plot_matrix(axes['precond'], M_np, "Preconditioner Matrix M")
            if data['A'] is not None:
                self._plot_matrix_comparison(axes['comparison'], data['A'], M_np)
        
        # Plot statistics panel
        self._plot_statistics(axes['stats'], data['A'], M_np, data['error'], 
                             dimension, scaling, preconditioner)
    
    def _to_dense_numpy(self, matrix):
        """Convert any matrix-like object to a dense NumPy array"""
        if matrix is None:
            return None
            
        # CuPy -> NumPy
        if hasattr(matrix, 'get'):
            matrix = matrix.get()
            
        # Sparse -> Dense
        if hasattr(matrix, 'toarray'):
            return matrix.toarray()
        elif hasattr(matrix, 'todense'):
            return matrix.todense()
            
        # Already dense
        return np.array(matrix)
    
    def _extract_preconditioner_matrix(self, preconditioner, matrix_size):
        """Extract matrix representation from preconditioner object"""
        if preconditioner is None:
            return None
            
        try:
            # Handle different preconditioner representations using chain of responsibility
            extractors = [
                lambda p: self._to_dense_numpy(p) if isinstance(p, (np.ndarray, sp.spmatrix)) else None,
                lambda p: self._to_dense_numpy(p.M) if hasattr(p, 'M') and p.M is not None else None,
                lambda p: self._to_dense_numpy(p.matrix) if hasattr(p, 'matrix') and p.matrix is not None else None,
                lambda p: self._linearoperator_to_matrix(p, matrix_size) if isinstance(p, LinearOperator) or 'LinearOperator' in str(type(p)) else None,
                lambda p: self._linearoperator_to_matrix(p.M, matrix_size) if hasattr(p, 'M') and (isinstance(p.M, LinearOperator) or 'LinearOperator' in str(type(p.M))) else None,
                lambda p: self._callable_to_matrix(p, matrix_size) if hasattr(p, '__call__') else None
            ]
            
            # Try each extractor until one works
            for extract in extractors:
                result = extract(preconditioner)
                if result is not None:
                    return result
                    
            print(f"Warning: Could not extract matrix from preconditioner of type {type(preconditioner)}")
            return None
            
        except Exception as e:
            print(f"Error extracting preconditioner matrix: {e}")
            return None
    
    def _linearoperator_to_matrix(self, op, size):
        """Convert a LinearOperator to a dense matrix by applying it to unit vectors"""
        try:
            # Use sampling for large matrices
            if size > 1000:
                sample_size = min(500, size)
                indices = np.linspace(0, size-1, sample_size, dtype=int)
                matrix = np.zeros((sample_size, sample_size))
                
                for i, idx in enumerate(indices):
                    unit = np.zeros(size)
                    unit[idx] = 1.0
                    result = op.matvec(unit)
                    matrix[i, :] = result[indices]
                return matrix
            
            # Full matrix for smaller systems
            matrix = np.zeros((size, size))
            for i in range(size):
                unit = np.zeros(size)
                unit[i] = 1.0
                matrix[:, i] = op.matvec(unit)
            return matrix
                
        except Exception as e:
            print(f"Error converting operator to matrix: {e}")
            return np.eye(size)  # Return identity as fallback
    
    def _callable_to_matrix(self, func, size):
        """Convert a callable preconditioner to a matrix"""
        # This is essentially the same as _linearoperator_to_matrix
        return self._linearoperator_to_matrix(func, size)
    
    def _plot_matrix(self, ax, matrix, title):
        """Plot a matrix with appropriate scaling and colorbar"""
        if matrix is None:
            ax.text(0.5, 0.5, "Matrix not available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
            
        # Handle colormap and scaling
        threshold = 1e-15
        non_zero = matrix[np.abs(matrix) > threshold]
        if len(non_zero) > 0:
            vmin = np.min(np.abs(non_zero))
            vmax = np.max(np.abs(matrix))
            # Ensure minimum values for stability
            im = ax.imshow(np.abs(matrix), norm=LogNorm(vmin=max(vmin, threshold), 
                                                      vmax=max(vmax, 1e-14)), 
                          cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label='Absolute Value (Log Scale)')
        else:
            ax.imshow(np.abs(matrix), cmap='viridis', aspect='auto')
            
        # Set labels
        ax.set_title(title)
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        
        # Thin out ticks for large matrices
        self._adjust_ticks_for_size(ax, matrix.shape)
            
        # Add sparsity info
        nnz = np.count_nonzero(np.abs(matrix) > threshold)
        sparsity = 1.0 - (nnz / matrix.size)
        ax.text(0.05, 0.05, f"Non-zeros: {nnz}\nSparsity: {sparsity:.4f}", 
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def _plot_solution_comparison(self, ax, x, exact_x):
        """Plot solution vector and exact solution comparison"""
        # Plot numerical solution
        ax.plot(x, np.arange(len(x)), 'r-', label='Numerical', linewidth=1)
        
        # Add exact solution if available
        if exact_x is not None:
            ax.plot(exact_x, np.arange(len(exact_x)), 'b--', label='Exact', linewidth=1)
            
            # Set view limits
            all_vals = np.concatenate([x.ravel(), exact_x.ravel()])
            min_val, max_val = all_vals.min(), all_vals.max()
            buffer = 0.1 * (max_val - min_val)
            ax.set_xlim([min_val - buffer, max_val + buffer])
        
        # Set labels and legend
        ax.set_title("Solution Vectors")
        ax.set_xlabel("Value")
        ax.set_ylabel("Index")
        ax.legend(loc='upper right')
        
        # Thin out ticks for large vectors
        if len(x) > 100:
            ticks = np.linspace(0, len(x)-1, 10, dtype=int)
            ax.set_yticks(ticks)
    
    def _plot_error_distribution(self, ax, error):
        """Plot error distribution"""
        # Plot error vs index
        ax.semilogy(np.arange(len(error)), error, 'r-', linewidth=1)
        ax.set_title("Error Distribution")
        ax.set_xlabel("Index")
        ax.set_ylabel("Absolute Error (Log Scale)")
        
        # Add grid and reference lines
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        mean_err = np.mean(error)
        max_err = np.max(error)
        ax.axhline(y=mean_err, color='g', linestyle='--', label=f'Mean: {mean_err:.2e}')
        ax.axhline(y=max_err, color='orange', linestyle='--', label=f'Max: {max_err:.2e}')
        ax.legend(loc='upper right')
        
        # Thin out ticks for large vectors
        if len(error) > 100:
            ticks = np.linspace(0, len(error)-1, 10, dtype=int)
            ax.set_xticks(ticks)
    
    def _plot_matrix_comparison(self, ax, A, M):
        """Plot comparison between system matrix and preconditioner"""
        try:
            # Use sampling for large matrices
            if A.shape[0] > 1000:
                sample_size = min(500, A.shape[0])
                step = max(1, A.shape[0] // sample_size)
                indices = np.arange(0, A.shape[0], step)
                A_sample = A[indices][:, indices]
                M_sample = M[indices][:, indices]
                product = A_sample @ M_sample
                ax.set_title("Sampled A×M Product (ideal: Identity)")
            else:
                product = A @ M
                ax.set_title("A×M Product (ideal: Identity)")
            
            # Plot product matrix
            im = ax.imshow(np.abs(product), norm=LogNorm(vmin=1e-15), cmap='inferno', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("Column Index")
            ax.set_ylabel("Row Index")
            
            # Calculate deviation from identity
            n = min(product.shape)
            identity = np.eye(n)
            diff = np.abs(product - identity)
            deviation = np.mean(diff)
            ax.text(0.05, 0.05, f"Mean Deviation from Identity: {deviation:.2e}", 
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Matrix comparison calculation error:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Matrix Comparison (Failed)")
    
    def _plot_statistics(self, ax, A, M, error, dimension, scaling, preconditioner):
        """Plot statistics about the matrix system and preconditioner"""
        ax.axis('off')  # Hide frame
        
        # Build statistics information
        info_text = []
        
        # System matrix stats
        if A is not None:
            info_text.extend(self._get_matrix_stats(A))
        
        # Preconditioner stats
        if M is not None:
            info_text.extend(self._get_preconditioner_stats(M, preconditioner))
        
        # Error stats
        if error is not None:
            info_text.extend(self._get_error_stats(error))
        
        # Scaling info
        if scaling:
            info_text.append(f"\nScaling Method: {scaling}")
        
        # Component-wise errors
        if error is not None and dimension in self.components:
            info_text.extend(self._get_component_errors(error, dimension))
        
        # Display all stats
        ax.text(0, 1, "\n".join(info_text), ha='left', va='top', fontsize=9,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    def _get_matrix_stats(self, A):
        """Get statistics about the system matrix"""
        stats = []
        threshold = 1e-15
        sparsity = 1.0 - (np.count_nonzero(np.abs(A) > threshold) / A.size)
        stats.append(f"Matrix Size: {A.shape[0]}×{A.shape[1]}")
        stats.append(f"Sparsity: {sparsity:.4f}")
        stats.append(f"Non-zeros: {np.count_nonzero(np.abs(A) > threshold)}")
        
        # Condition number (only for smaller matrices)
        if min(A.shape) < 1000:
            try:
                cond = np.linalg.cond(A)
                stats.append(f"Condition Number: {cond:.2e}")
            except:
                pass
                
        return stats
    
    def _get_preconditioner_stats(self, M, preconditioner):
        """Get statistics about the preconditioner"""
        stats = ["\nPreconditioner Statistics:"]
        threshold = 1e-15
        
        # Basic stats
        M_sparsity = 1.0 - (np.count_nonzero(np.abs(M) > threshold) / M.size)
        stats.append(f"Sparsity: {M_sparsity:.4f}")
        stats.append(f"Non-zeros: {np.count_nonzero(np.abs(M) > threshold)}")
        
        # Structure type
        diag_ratio = self._check_diagonal_dominance(M)
        if diag_ratio > 0.95:
            stats.append("Type: Mostly Diagonal")
        elif diag_ratio > 0.7:
            stats.append("Type: Block Diagonal")
        else:
            stats.append("Type: General Sparse")
            
        # Symmetry check
        try:
            is_symmetric = np.allclose(M, M.T, rtol=1e-5, atol=1e-8)
            stats.append(f"Symmetric: {'Yes' if is_symmetric else 'No'}")
        except:
            pass
        
        # Additional preconditioner info
        if preconditioner is not None:
            description = self._get_attribute(preconditioner, 'description')
            if description:
                stats.append(f"Description: {description}")
                
            name = self._get_attribute(preconditioner, 'name')
            if name:
                stats.append(f"Name: {name}")
                
        return stats
    
    def _get_error_stats(self, error):
        """Get error statistics"""
        return [
            "\nError Statistics:",
            f"Max Error: {np.max(error):.4e}",
            f"Mean Error: {np.mean(error):.4e}",
            f"Median Error: {np.median(error):.4e}"
        ]
    
    def _get_component_errors(self, error, dimension):
        """Get component-wise errors based on dimension"""
        stats = ["\nComponent Errors:"]
        
        components = self.components[dimension]
        var_count = len(components)
        
        # Only process if error length is compatible with the component count
        if len(error) % var_count == 0:
            for i, name in enumerate(components):
                indices = range(i, len(error), var_count)
                comp_error = np.max(error[indices])
                stats.append(f"{name}: {comp_error:.4e}")
                
        return stats
    
    def _check_diagonal_dominance(self, M):
        """Check if a matrix is diagonally dominant or block-diagonal"""
        try:
            # Calculate diagonal dominance ratio
            n = min(M.shape)
            diag_elements = np.sum(np.abs(np.diag(M[:n, :n])))
            total_elements = np.sum(np.abs(M))
            
            if total_elements == 0:
                return 0
                
            # Basic diagonal ratio
            diag_ratio = diag_elements / total_elements
            
            # For smaller matrices, also check near-diagonal elements
            if n < 1000:
                bandwidth = max(3, n // 50)
                near_diag_mask = np.zeros_like(M, dtype=bool)
                
                for i in range(-bandwidth, bandwidth+1):
                    diag_indices = np.diag_indices(n)
                    offset_indices = (diag_indices[0], np.clip(diag_indices[1] + i, 0, n-1))
                    near_diag_mask[offset_indices] = True
                    
                near_diag_elements = np.sum(np.abs(M * near_diag_mask))
                near_diag_ratio = near_diag_elements / total_elements
                
                return max(diag_ratio, near_diag_ratio)
                
            return diag_ratio
        except Exception as e:
            print(f"Error in diagonal dominance check: {e}")
            return 0.0
    
    def _adjust_ticks_for_size(self, ax, shape):
        """Adjust axis ticks based on matrix/vector size"""
        rows, cols = shape
        if rows > 100:
            row_ticks = np.linspace(0, rows-1, 10, dtype=int)
            ax.set_yticks(row_ticks)
        if cols > 100:
            col_ticks = np.linspace(0, cols-1, 10, dtype=int)
            ax.set_xticks(col_ticks)
    
    def _get_attribute(self, obj, attr_name, default=None):
        """Safely get attribute from an object"""
        if obj is None:
            return default
        return getattr(obj, attr_name, default)