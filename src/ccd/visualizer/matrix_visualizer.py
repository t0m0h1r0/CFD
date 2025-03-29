"""
Matrix System Visualization Utility

Provides robust visualization of linear system matrices, solution vectors, 
and associated statistical information.
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
            preconditioner: Ignored (legacy parameter)
        
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
            
            # 4. Eigenvalue estimation (empty space where preconditioner was)
            plt.subplot(gs[1, 0])
            self._plot_eigenvalue_estimate(A_np)
            
            # 5. Sparsity pattern (empty space where product was)
            plt.subplot(gs[1, 1])
            self._plot_sparsity_pattern(A_np)
            
            # 6. Statistics
            plt.subplot(gs[1, 2])
            self._plot_system_statistics(A_np, error_np, dimension, scaling)
            
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
    
    def _plot_eigenvalue_estimate(self, A):
        """
        Estimate and plot eigenvalues for small matrices
        
        Args:
            A: System matrix
        """
        plt.title("Matrix Properties")
        
        if A is None or A.size == 0:
            plt.text(0.5, 0.5, "No Matrix Data", ha='center', va='center')
            return
        
        try:
            # Only compute eigenvalues for small matrices to avoid memory issues
            if A.shape[0] <= 1000:
                # For sparse matrices
                if hasattr(A, 'todense'):
                    A_dense = A.todense()
                else:
                    A_dense = A
                
                # For very large matrices, use a subset
                if A_dense.shape[0] > 500:
                    # Get a subset of eigenvalues
                    from scipy.sparse.linalg import eigs
                    try:
                        # Try to get a few eigenvalues
                        eigvals = eigs(A, k=min(20, A.shape[0]-2), return_eigenvectors=False)
                        plt.scatter(np.real(eigvals), np.imag(eigvals), color='blue', alpha=0.6)
                        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                        plt.xlabel("Real Part")
                        plt.ylabel("Imaginary Part")
                        plt.text(0.05, 0.95, "Eigenvalue Estimates (subset)", 
                                transform=plt.gca().transAxes, va='top')
                    except Exception as e:
                        plt.text(0.5, 0.5, f"Eigenvalue computation failed: {str(e)}", 
                                ha='center', va='center')
                else:
                    # For small matrices, compute full spectrum
                    try:
                        eigvals = np.linalg.eigvals(A_dense)
                        plt.scatter(np.real(eigvals), np.imag(eigvals), color='blue', alpha=0.6)
                        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                        plt.xlabel("Real Part")
                        plt.ylabel("Imaginary Part")
                        
                        # Condition number
                        cond = np.linalg.cond(A_dense)
                        plt.text(0.05, 0.95, f"Condition Number: {cond:.2e}", 
                                transform=plt.gca().transAxes, va='top')
                    except Exception as e:
                        plt.text(0.5, 0.5, f"Eigenvalue computation failed: {str(e)}", 
                                ha='center', va='center')
            else:
                plt.text(0.5, 0.5, "Matrix too large for eigenvalue computation", 
                        ha='center', va='center')
        except Exception as e:
            plt.text(0.5, 0.5, f"Analysis Error: {str(e)}", ha='center', va='center')
    
    def _plot_sparsity_pattern(self, A):
        """
        Visualize the sparsity pattern of the matrix
        
        Args:
            A: System matrix
        """
        plt.title("Sparsity Pattern")
        
        if A is None or A.size == 0:
            plt.text(0.5, 0.5, "No Matrix Data", ha='center', va='center')
            return
        
        try:
            # Create a binary pattern (0 for zero, 1 for non-zero)
            if A.shape[0] > 1000:
                # For large matrices, sample a subset
                n = A.shape[0]
                stride = max(1, n // 1000)
                pattern = np.zeros((n//stride + 1, n//stride + 1))
                for i in range(0, n, stride):
                    for j in range(0, n, stride):
                        i_idx, j_idx = i//stride, j//stride
                        if i_idx < pattern.shape[0] and j_idx < pattern.shape[1]:
                            submatrix = A[i:min(i+stride, n), j:min(j+stride, n)]
                            pattern[i_idx, j_idx] = np.any(submatrix != 0)
            else:
                pattern = (A != 0).astype(float)
            
            plt.imshow(pattern, cmap='binary', aspect='auto')
            plt.xlabel("Column Index")
            plt.ylabel("Row Index")
            
            # Pattern statistics
            nnz = np.count_nonzero(pattern)
            sparsity = 1 - nnz/pattern.size
            plt.text(0.05, 0.05, f"Pattern Density: {1-sparsity:.6f}", 
                    transform=plt.gca().transAxes, fontsize=8)
        except Exception as e:
            plt.text(0.5, 0.5, f"Pattern Analysis Error: {str(e)}", ha='center', va='center')
    
    def _plot_system_statistics(self, A, error, dimension, scaling):
        """
        Generate textual statistics about the matrix system with robust error handling
        
        Args:
            A: System matrix
            error: Error vector
            dimension: Problem dimension
            scaling: Scaling method
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
            
            # Try to compute condition number for small matrices
            if A.shape[0] <= 500:
                try:
                    cond = np.linalg.cond(A)
                    stats.append(f"Condition Number: {cond:.2e}")
                except:
                    pass
        
        # Error stats
        if error is not None and len(error) > 0:
            stats.extend([
                "\nError Stats:",
                f"Max Error: {np.max(error):.2e}",
                f"Mean Error: {np.mean(error):.2e}"
            ])
        
        # System info
        stats.append(f"\nDimension: {dimension}D")
        if scaling:
            stats.append(f"Scaling: {scaling}")
        
        plt.text(0.05, 0.95, '\n'.join(stats), 
                 transform=plt.gca().transAxes, va='top', fontsize=9)
        plt.title("System Statistics")

# Configure plot style
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10