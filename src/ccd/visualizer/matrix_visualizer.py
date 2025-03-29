"""
Matrix System Visualization Utility

Provides robust visualization of linear system matrices, solution vectors, 
and associated statistical information with improved LinearOperator support.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


class MatrixVisualizer:
    """Robust visualization of matrix systems and solver results"""
    
    def __init__(self, output_dir="results"):
        """
        Initialize output directory for visualizations
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    def _to_numpy(self, arr):
        """
        Safely convert various array types to NumPy array
        
        Args:
            arr: Input array or matrix
        
        Returns:
            NumPy array or None if conversion fails
        """
        try:
            # Handle CuPy arrays
            if hasattr(arr, 'get'):
                arr = arr.get()
            
            # Handle sparse matrices
            if hasattr(arr, 'toarray'):
                arr = arr.toarray()
            
            # Handle LinearOperator by extracting approximation
            if isinstance(arr, ScipyLinearOperator):
                # Try to create an approximation matrix
                try:
                    n = arr.shape[0]
                    # Create an identity-like matrix and apply operator
                    test_matrix = np.eye(min(n, 500))  # Limit size to avoid memory issues
                    return arr(test_matrix)
                except Exception as e:
                    print(f"LinearOperator approximation failed: {e}")
                    return None
            
            # Ensure NumPy array
            return np.asarray(arr)
        except Exception as e:
            print(f"Array conversion error: {e}")
            return None
    
    def _safe_matrix_abs(self, matrix):
        """
        Safely compute absolute value of matrix
        
        Args:
            matrix: Input matrix
        
        Returns:
            Absolute value matrix or None if failed
        """
        if matrix is None:
            return None
        
        try:
            # Handle potential complex or non-numeric data
            abs_matrix = np.abs(matrix)
            
            # Replace inf and nan with 0
            abs_matrix[np.isinf(abs_matrix)] = 0
            abs_matrix[np.isnan(abs_matrix)] = 0
            
            return abs_matrix
        except Exception as e:
            print(f"Matrix absolute value computation error: {e}")
            return None
    
    def visualize(self, A, b, x, exact_x, title, dimension, scaling=None, preconditioner=None):
        """
        Generate comprehensive matrix system visualization with robust error handling
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x: Numerical solution
            exact_x: Exact solution
            title: Base visualization title
            dimension: Problem dimension
            scaling: Optional scaling method
            preconditioner: Optional preconditioner
        
        Returns:
            Output file path
        """
        # Numpy conversion with error handling
        A_np = self._to_numpy(A)
        x_np = self._to_numpy(x)
        exact_np = self._to_numpy(exact_x)
        
        # Error computation
        error_np = None
        if x_np is not None and exact_np is not None:
            try:
                error_np = np.abs(x_np.flatten() - exact_np.flatten())
            except Exception as e:
                print(f"Error computation failed: {e}")
        
        # Preconditioner handling
        M_np = None
        try:
            # Multiple ways to extract preconditioner matrix
            if preconditioner is not None:
                if hasattr(preconditioner, 'M'):
                    M_np = self._to_numpy(preconditioner.M)
                elif hasattr(preconditioner, 'matrix'):
                    M_np = self._to_numpy(preconditioner.matrix)
                elif hasattr(preconditioner, 'toarray'):
                    M_np = self._to_numpy(preconditioner.toarray())
        except Exception as e:
            print(f"Preconditioner matrix extraction failed: {e}")
        
        # Create visualization with fallback mechanism
        try:
            plt.figure(figsize=(16, 10))
            plt.suptitle(f"{dimension}D {title} Visualization", fontsize=16)
            
            # Grid layout
            gs = plt.GridSpec(2, 3)
            
            # 1. System Matrix Visualization
            plt.subplot(gs[0, 0])
            self._plot_matrix(A_np, "System Matrix A")
            
            # 2. Solution Vectors
            plt.subplot(gs[0, 1])
            self._plot_solution_vectors(x_np, exact_np)
            
            # 3. Error Distribution
            plt.subplot(gs[0, 2])
            self._plot_error_distribution(error_np)
            
            # 4. Preconditioner Matrix (if available)
            plt.subplot(gs[1, 0])
            self._plot_matrix(M_np, "Preconditioner Matrix M")
            
            # 5. Matrix Product Visualization
            plt.subplot(gs[1, 1])
            self._plot_matrix_product(A_np, M_np)
            
            # 6. Statistics
            plt.subplot(gs[1, 2])
            self._plot_system_statistics(A_np, M_np, error_np, dimension, scaling, preconditioner)
            
            # Finalize and save
            plt.tight_layout()
            output_path = f"{self.output_dir}/{title}_matrix_analysis.png"
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        
        except Exception as e:
            print(f"Visualization generation failed: {e}")
            return None
    
    def _plot_matrix(self, matrix, title):
        """
        Plot matrix structure with log-scale visualization and robust error handling
        
        Args:
            matrix: Input matrix
            title: Plot title
        """
        plt.title(title)
        
        # Validate matrix
        abs_matrix = self._safe_matrix_abs(matrix)
        
        if abs_matrix is None or abs_matrix.size == 0:
            plt.text(0.5, 0.5, "No Matrix Data", ha='center', va='center')
            return
        
        # Ensure non-zero elements exist for log normalization
        non_zero = abs_matrix[abs_matrix > 0]
        
        if len(non_zero) == 0:
            plt.imshow(abs_matrix, cmap='viridis', aspect='auto')
        else:
            plt.imshow(abs_matrix, 
                       norm=LogNorm(vmin=non_zero.min(), vmax=abs_matrix.max()), 
                       cmap='viridis', aspect='auto')
        
        plt.colorbar(label='Absolute Value')
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        
        # Sparsity information
        nnz = np.count_nonzero(abs_matrix)
        plt.text(0.05, 0.05, f"Non-zeros: {nnz}\nSparsity: {1-nnz/abs_matrix.size:.4f}", 
                 transform=plt.gca().transAxes, fontsize=8)
    
    def _plot_solution_vectors(self, x, exact_x):
        """
        Compare numerical and exact solution vectors with robust error handling
        
        Args:
            x: Numerical solution
            exact_x: Exact solution
        """
        plt.title("Solution Vectors")
        
        # Validate inputs
        if x is None:
            plt.text(0.5, 0.5, "No Solution Data", ha='center', va='center')
            return
        
        # Flatten to ensure 1D
        x_flat = x.flatten()
        plt.plot(x_flat, label='Numerical', color='red')
        
        if exact_x is not None:
            exact_flat = exact_x.flatten()
            plt.plot(exact_flat, label='Exact', color='blue', linestyle='--')
        
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
    
    def _plot_error_distribution(self, error):
        """
        Visualize error distribution on log scale with robust error handling
        
        Args:
            error: Error vector
        """
        plt.title("Error Distribution")
        
        if error is None or len(error) == 0:
            plt.text(0.5, 0.5, "No Error Data", ha='center', va='center')
            return
        
        # Ensure positive, non-zero errors for log scale
        valid_error = error[error > 0]
        
        if len(valid_error) == 0:
            plt.text(0.5, 0.5, "Zero/Invalid Errors", ha='center', va='center')
            return
        
        plt.semilogy(valid_error, color='red')
        plt.xlabel("Index")
        plt.ylabel("Absolute Error (Log Scale)")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.text(0.05, 0.95, f"Max Error: {np.max(error):.2e}", 
                 transform=plt.gca().transAxes, va='top')
    
    def _plot_matrix_product(self, A, M):
        """
        Visualize matrix-preconditioner product with robust error handling
        
        Args:
            A: System matrix
            M: Preconditioner matrix
        """
        plt.title("Matrix Product")
        
        if A is None or M is None:
            plt.text(0.5, 0.5, "Cannot Compute Product", ha='center', va='center')
            return
        
        try:
            # Compute and visualize matrix product
            product = self._safe_matrix_abs(A @ M)
            
            if product is None or product.size == 0:
                plt.text(0.5, 0.5, "Product Computation Failed", ha='center', va='center')
                return
            
            non_zero = product[product > 0]
            if len(non_zero) == 0:
                plt.imshow(product, cmap='inferno')
            else:
                plt.imshow(product, norm=LogNorm(), cmap='inferno')
            
            plt.colorbar(label='Product Magnitude')
            plt.xlabel("Column Index")
            plt.ylabel("Row Index")
        except Exception as e:
            print(f"Matrix product visualization error: {e}")
            plt.text(0.5, 0.5, f"Product Error: {str(e)}", ha='center', va='center')
    
    def _plot_system_statistics(self, A, M, error, dimension, scaling, preconditioner):
        """
        Generate textual statistics about the matrix system with robust error handling
        
        Args:
            A: System matrix
            M: Preconditioner matrix
            error: Error vector
            dimension: Problem dimension
            scaling: Scaling method
            preconditioner: Preconditioner object
        """
        plt.axis('off')
        stats = []
        
        # Matrix system stats
        if A is not None and A.size > 0:
            stats.extend([
                f"Matrix Size: {A.shape[0]}×{A.shape[1]}",
                f"Sparsity: {1 - np.count_nonzero(A)/A.size:.4f}",
                f"Non-zeros: {np.count_nonzero(A)}"
            ])
        
        # Preconditioner stats
        if M is not None and M.size > 0:
            stats.extend([
                "\nPreconditioner Stats:",
                f"Sparsity: {1 - np.count_nonzero(M)/M.size:.4f}",
                f"Non-zeros: {np.count_nonzero(M)}"
            ])
        
        # Error stats
        if error is not None and len(error) > 0:
            stats.extend([
                "\nError Stats:",
                f"Max Error: {np.max(error):.2e}",
                f"Mean Error: {np.mean(error):.2e}"
            ])
        
        # Additional info
        if scaling:
            stats.append(f"\nScaling: {scaling}")
        
        plt.text(0.05, 0.95, '\n'.join(stats), 
                 transform=plt.gca().transAxes, va='top', fontsize=9)
        plt.title("System Statistics")

# Configure plot style
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10