"""
Matrix System and Preconditioner Visualizer Module

This module provides functionality to visualize linear system matrices (Ax = b)
and preconditioner matrices (M) to help understand their structure and characteristics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


class MatrixVisualizer:
    """Class for visualizing matrix systems (Ax = b) and preconditioners"""
    
    def __init__(self, output_dir="results"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize(self, A, b, x, exact_x, title, dimension, scaling=None, preconditioner=None):
        """
        Visualize the matrix system and optionally the preconditioner matrix
        
        Args:
            A: System matrix (CPU/SciPy format)
            b: Right-hand side vector (CPU/NumPy format)
            x: Solution vector (CPU/NumPy format)
            exact_x: Exact solution vector (CPU/NumPy format)
            title: Title
            dimension: Dimension (1 or 2 or 3)
            scaling: Scaling method name (optional)
            preconditioner: Preconditioner object (optional)
            
        Returns:
            Output file path
        """
        # Generate output path
        scale_suffix = f"_{scaling}" if scaling else ""
        output_path = f"{self.output_dir}/{title}{scale_suffix}_matrix.png"
        
        # Data conversion function
        def to_numpy(arr):
            if arr is None:
                return None
            if hasattr(arr, 'toarray'):
                return arr.toarray() if not hasattr(arr, 'get') else arr.get().toarray()
            return arr.get() if hasattr(arr, 'get') else arr
        
        # Convert all data to NumPy format
        A_np = to_numpy(A)
        b_np = to_numpy(b).reshape(-1, 1) if b is not None else None
        x_np = to_numpy(x).reshape(-1, 1) if x is not None else None
        exact_np = to_numpy(exact_x).reshape(-1, 1) if exact_x is not None else None
        error_np = np.abs(x_np - exact_np) if x_np is not None and exact_np is not None else None
        
        # Extract preconditioner matrix if available
        M_np = None
        if preconditioner is not None:
            if hasattr(preconditioner, 'M') and preconditioner.M is not None:
                M_np = to_numpy(preconditioner.M)
            elif hasattr(preconditioner, 'matrix') and preconditioner.matrix is not None:
                M_np = to_numpy(preconditioner.matrix)
        
        # Determine layout based on whether preconditioner is available
        if M_np is not None:
            # Layout for both A and M visualization (2x3 grid)
            fig = plt.figure(figsize=(18, 12))
            gs = plt.GridSpec(2, 3, height_ratios=[3, 2])
            
            # 1. System matrix A visualization (top left)
            ax_matrix = fig.add_subplot(gs[0, 0])
            self._plot_matrix(ax_matrix, A_np, "System Matrix A")
            
            # 2. Preconditioner matrix M visualization (top middle)
            ax_precond = fig.add_subplot(gs[0, 1])
            self._plot_matrix(ax_precond, M_np, "Preconditioner Matrix M")
            
            # 3. Solution vectors comparison (top right)
            ax_solution = fig.add_subplot(gs[0, 2])
            if x_np is not None:
                self._plot_solution_comparison(ax_solution, x_np, exact_np)
                
            # 4. Error distribution (bottom left)
            ax_error = fig.add_subplot(gs[1, 0])
            if error_np is not None:
                self._plot_error_distribution(ax_error, error_np)
                
            # 5. Matrix comparison (bottom middle)
            ax_compare = fig.add_subplot(gs[1, 1])
            if A_np is not None and M_np is not None:
                self._plot_matrix_comparison(ax_compare, A_np, M_np)
                
            # 6. Statistics and info (bottom right)
            ax_stats = fig.add_subplot(gs[1, 2])
            self._plot_statistics(ax_stats, A_np, M_np, error_np, dimension, scaling, preconditioner)
            
        else:
            # Original layout for A only (2x2 grid)
            fig = plt.figure(figsize=(14, 10))
            gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
            
            # 1. System matrix visualization (top left)
            ax_matrix = fig.add_subplot(gs[0, 0])
            self._plot_matrix(ax_matrix, A_np, "System Matrix A")
            
            # 2. Solution vectors comparison (top right)
            ax_solution = fig.add_subplot(gs[0, 1])
            if x_np is not None:
                self._plot_solution_comparison(ax_solution, x_np, exact_np)
                
            # 3. Error distribution (bottom left)
            ax_error = fig.add_subplot(gs[1, 0])
            if error_np is not None:
                self._plot_error_distribution(ax_error, error_np)
                
            # 4. Statistics (bottom right)
            ax_stats = fig.add_subplot(gs[1, 1])
            self._plot_statistics(ax_stats, A_np, None, error_np, dimension, scaling, preconditioner)
        
        # Set overall title
        precond_info = ""
        if preconditioner is not None:
            precond_name = preconditioner.name if hasattr(preconditioner, 'name') else "Unknown"
            precond_info = f", Preconditioner: {precond_name}"
            
        plt.suptitle(f"{dimension}D {title}" + 
                    (f" (Scaling: {scaling})" if scaling else "") + 
                    precond_info, 
                    fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_matrix(self, ax, matrix, title):
        """
        Plot a matrix with appropriate scaling and colorbar
        
        Args:
            ax: Matplotlib axis
            matrix: Matrix to plot
            title: Plot title
        """
        if matrix is None:
            ax.text(0.5, 0.5, "Matrix not available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
            
        # Check non-zero elements range
        non_zero = matrix[matrix > 0]
        if len(non_zero) > 0:
            vmin = non_zero.min()
            vmax = matrix.max()
            # Visualize matrix (log scale)
            im = ax.imshow(np.abs(matrix), norm=LogNorm(vmin=vmin, vmax=vmax), 
                          cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label='Absolute Value (Log Scale)')
        else:
            ax.imshow(np.abs(matrix), cmap='viridis', aspect='auto')
            
        ax.set_title(title)
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        
        # Improve visualization for large matrices
        rows, cols = matrix.shape
        if rows > 100:
            row_ticks = np.linspace(0, rows-1, 10, dtype=int)
            ax.set_yticks(row_ticks)
        if cols > 100:
            col_ticks = np.linspace(0, cols-1, 10, dtype=int)
            ax.set_xticks(col_ticks)
            
        # Add sparsity info to the plot
        nnz = np.count_nonzero(matrix)
        sparsity = 1.0 - (nnz / matrix.size)
        ax.text(0.05, 0.05, f"Non-zeros: {nnz}\nSparsity: {sparsity:.4f}", 
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def _plot_solution_comparison(self, ax, x, exact_x):
        """
        Plot solution vector and exact solution comparison
        
        Args:
            ax: Matplotlib axis
            x: Solution vector
            exact_x: Exact solution vector
        """
        # Plot solution vector
        ax.plot(x, np.arange(len(x)), 'r-', label='Numerical', linewidth=1)
        
        if exact_x is not None:
            # Compare with exact solution
            ax.plot(exact_x, np.arange(len(exact_x)), 'b--', label='Exact', linewidth=1)
            
            # Adjust viewing range for better visibility
            all_vals = np.concatenate([x.ravel(), exact_x.ravel()])
            min_val, max_val = all_vals.min(), all_vals.max()
            buffer = 0.1 * (max_val - min_val)
            ax.set_xlim([min_val - buffer, max_val + buffer])
        
        ax.set_title("Solution Vectors")
        ax.set_xlabel("Value")
        ax.set_ylabel("Index")
        ax.legend(loc='upper right')
        
        # Thin out ticks for large vectors
        if len(x) > 100:
            solution_ticks = np.linspace(0, len(x)-1, 10, dtype=int)
            ax.set_yticks(solution_ticks)
    
    def _plot_error_distribution(self, ax, error):
        """
        Plot error distribution
        
        Args:
            ax: Matplotlib axis
            error: Error vector
        """
        # Plot error vs index
        ax.semilogy(np.arange(len(error)), error, 'r-', linewidth=1)
        ax.set_title("Error Distribution")
        ax.set_xlabel("Index")
        ax.set_ylabel("Absolute Error (Log Scale)")
        
        # Add grid lines
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Display mean and max error as horizontal lines
        mean_err = np.mean(error)
        max_err = np.max(error)
        ax.axhline(y=mean_err, color='g', linestyle='--', label=f'Mean: {mean_err:.2e}')
        ax.axhline(y=max_err, color='orange', linestyle='--', label=f'Max: {max_err:.2e}')
        ax.legend(loc='upper right')
        
        # Thin out ticks for large vectors
        if len(error) > 100:
            error_ticks = np.linspace(0, len(error)-1, 10, dtype=int)
            ax.set_xticks(error_ticks)
    
    def _plot_matrix_comparison(self, ax, A, M):
        """
        Plot comparison between system matrix and preconditioner
        
        Args:
            ax: Matplotlib axis
            A: System matrix
            M: Preconditioner matrix
        """
        # Calculate matrix-preconditioner product (ideally close to identity)
        try:
            # For large matrices, use sampling to visualize product regions
            if A.shape[0] > 1000:
                # Sample a subset of the matrix for visualization
                sample_size = min(500, A.shape[0])
                step = max(1, A.shape[0] // sample_size)
                indices = np.arange(0, A.shape[0], step)
                A_sample = A[indices][:, indices]
                M_sample = M[indices][:, indices]
                product = A_sample @ M_sample
                ax.set_title(f"Sampled A×M Product (ideal: Identity)")
            else:
                product = A @ M
                ax.set_title(f"A×M Product (ideal: Identity)")
            
            # Visualize the product
            im = ax.imshow(np.abs(product), norm=LogNorm(), cmap='inferno', aspect='auto')
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
        """
        Plot statistics about the matrix system and preconditioner
        
        Args:
            ax: Matplotlib axis
            A: System matrix
            M: Preconditioner matrix
            error: Error vector
            dimension: Problem dimension
            scaling: Scaling method
            preconditioner: Preconditioner object
        """
        ax.axis('off')  # Hide frame
        
        # Generate statistics text
        info_text = []
        
        # Matrix system information
        if A is not None:
            sparsity = 1.0 - (np.count_nonzero(A) / A.size)
            info_text.append(f"Matrix Size: {A.shape[0]}×{A.shape[1]}")
            info_text.append(f"Sparsity: {sparsity:.4f}")
            info_text.append(f"Non-zeros: {np.count_nonzero(A)}")
            
            # Condition number (only for smaller matrices)
            if min(A.shape) < 1000:  # Expensive for large matrices
                try:
                    cond = np.linalg.cond(A)
                    info_text.append(f"Condition Number: {cond:.2e}")
                except:
                    pass
        
        # Preconditioner information
        if M is not None:
            info_text.append("\nPreconditioner Statistics:")
            M_sparsity = 1.0 - (np.count_nonzero(M) / M.size)
            info_text.append(f"Sparsity: {M_sparsity:.4f}")
            info_text.append(f"Non-zeros: {np.count_nonzero(M)}")
            
            # Check if M is diagonal or block-diagonal
            diag_ratio = self._check_diagonal_dominance(M)
            if diag_ratio > 0.95:
                info_text.append("Type: Mostly Diagonal")
            elif diag_ratio > 0.7:
                info_text.append("Type: Block Diagonal")
            else:
                info_text.append("Type: General Sparse")
                
            # Check preconditioner symmetry
            is_symmetric = np.allclose(M, M.T, rtol=1e-5, atol=1e-8)
            info_text.append(f"Symmetric: {'Yes' if is_symmetric else 'No'}")
            
            # Add detailed info from preconditioner if available
            if preconditioner is not None:
                if hasattr(preconditioner, 'description'):
                    info_text.append(f"Description: {preconditioner.description}")
                if hasattr(preconditioner, 'name'):
                    info_text.append(f"Name: {preconditioner.name}")
        
        # Error information
        if error is not None:
            info_text.append("\nError Statistics:")
            info_text.append(f"Max Error: {np.max(error):.4e}")
            info_text.append(f"Mean Error: {np.mean(error):.4e}")
            info_text.append(f"Median Error: {np.median(error):.4e}")
        
        # Scaling information
        if scaling:
            info_text.append(f"\nScaling Method: {scaling}")
        
        # Component errors by dimension
        if error is not None and A is not None:
            info_text.append("\nComponent Errors:")
            
            if dimension == 1 and len(error) % 4 == 0:  # 1D
                components = ["ψ", "ψ'", "ψ''", "ψ'''"]
                for i, name in enumerate(components):
                    indices = range(i, len(error), 4)
                    comp_error = np.max(error[indices])
                    info_text.append(f"{name}: {comp_error:.4e}")
                    
            elif dimension == 2 and len(error) % 7 == 0:  # 2D
                components = ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
                for i, name in enumerate(components):
                    indices = range(i, len(error), 7)
                    comp_error = np.max(error[indices])
                    info_text.append(f"{name}: {comp_error:.4e}")
            
            elif dimension == 3 and len(error) % 10 == 0:  # 3D
                components = ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
                for i, name in enumerate(components):
                    indices = range(i, len(error), 10)
                    comp_error = np.max(error[indices])
                    info_text.append(f"{name}: {comp_error:.4e}")
        
        # Display statistics
        ax.text(0, 1, "\n".join(info_text), ha='left', va='top', fontsize=9,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    def _check_diagonal_dominance(self, M):
        """
        Check if a matrix is diagonally dominant or block-diagonal
        
        Args:
            M: Matrix to check
            
        Returns:
            Ratio of diagonal/block-diagonal elements to total non-zeros
        """
        # Simple check for diagonal dominance
        n = min(M.shape)
        diag_elements = np.sum(np.abs(np.diag(M[:n, :n])))
        total_elements = np.sum(np.abs(M))
        
        if total_elements == 0:
            return 0
            
        # Calculate ratio of diagonal elements to total
        diag_ratio = diag_elements / total_elements
        
        # For small matrices, also check near-diagonal elements
        if n < 1000:
            # Include elements within bandwidth of diagonal
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