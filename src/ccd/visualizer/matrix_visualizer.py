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
        if arr is None:
            return None
            
        try:
            # Handle scalar values
            import numpy as np
            if np.isscalar(arr):
                return np.array([arr])
                
            # Handle CuPy arrays
            if hasattr(arr, 'get') and callable(getattr(arr, 'get')):
                arr = arr.get()
            
            # Import scipy.sparse inside the function to avoid global dependency
            import scipy.sparse as sparse
            
            # Handle scipy.sparse matrices
            if isinstance(arr, sparse.spmatrix):
                # For large sparse matrices, convert more efficiently
                if max(arr.shape) > 1000:
                    # Sample a smaller representation
                    n = arr.shape[0]
                    sample_size = min(n, 64)
                    indices = np.linspace(0, n-1, sample_size, dtype=int)
                    M_sample = np.zeros((n, sample_size))
                    
                    for idx, i in enumerate(indices):
                        e_i = np.zeros(n)
                        e_i[i] = 1.0
                        if arr.shape[1] == n:  # Square matrix
                            M_sample[:, idx] = arr @ e_i
                        else:
                            # Non-square: Just extract the column if possible
                            if i < arr.shape[1]:
                                if hasattr(arr, 'getcol'):
                                    M_sample[:, idx] = arr.getcol(i).toarray().flatten()
                                else:
                                    M_sample[:, idx] = arr[:, i].toarray().flatten()
                    
                    return M_sample
                else:
                    # For smaller matrices, direct conversion is fine
                    return arr.toarray()
            
            # Handle sparse matrices with toarray method
            if hasattr(arr, 'toarray') and callable(getattr(arr, 'toarray')):
                return arr.toarray()
            
            # Handle LinearOperator
            if isinstance(arr, ScipyLinearOperator):
                try:
                    n = arr.shape[0]
                    sample_size = min(n, 64)
                    indices = np.linspace(0, n-1, sample_size, dtype=int)
                    M = np.zeros((n, sample_size))
                    
                    for idx, i in enumerate(indices):
                        e_i = np.zeros(n)
                        e_i[i] = 1.0
                        M[:, idx] = arr.matvec(e_i)
                    
                    return M
                except Exception as e:
                    print(f"LinearOperator approximation failed: {e}")
                    return None
            
            # Handle JAX arrays
            if 'jax' in str(type(arr)):
                return np.array(arr)
            
            # Handle Python lists, tuples, or other array-like objects
            return np.asarray(arr)
        except Exception as e:
            print(f"Array conversion error: {e}")
            return None
    
    def _get_preconditioner_matrix(self, preconditioner, A_size=None):
        """
        Extract matrix representation from preconditioner
        
        Args:
            preconditioner: Preconditioner object
            A_size: Size of system matrix for identity creation (optional)
            
        Returns:
            Matrix representation of preconditioner or None
        """
        if preconditioner is None:
            return None
        
        try:
            precond_type = type(preconditioner).__name__
            print(f"Extracting matrix from {precond_type}")
            
            # Special case for IdentityPreconditioner
            if precond_type == 'IdentityPreconditioner':
                # Create identity matrix of appropriate size
                if A_size is not None:
                    n = A_size[0]
                    return np.eye(n)
                else:
                    # Default size if we can't determine actual size
                    return np.eye(128)
            
            # For AMGPreconditioner, try to extract from ml
            if precond_type == 'AMGPreconditioner' and hasattr(preconditioner, 'ml'):
                try:
                    print("Extracting AMG matrix representation...")
                    # Get size from pyAMG levels
                    n = 128  # Default
                    if hasattr(preconditioner.ml, 'levels'):
                        levels = preconditioner.ml.levels
                        if levels and hasattr(levels[0], 'A'):
                            n = levels[0].A.shape[0]
                    elif A_size is not None:
                        n = A_size[0]
                    
                    # Sample some columns for visualization
                    sample_size = min(n, 64)
                    indices = np.linspace(0, n-1, sample_size, dtype=int)
                    
                    # Create approximation matrix
                    M = np.zeros((n, sample_size))
                    for idx, i in enumerate(indices):
                        e_i = np.zeros(n)
                        e_i[i] = 1.0
                        # Apply one cycle of AMG
                        try:
                            M[:, idx] = preconditioner.ml.solve(e_i, tol=1e-12, maxiter=1, cycle=preconditioner.cycle_type)
                        except Exception as solve_err:
                            print(f"AMG solve error: {solve_err}")
                            # Try fallback to __call__
                            try:
                                M[:, idx] = self._to_numpy(preconditioner(e_i))
                            except:
                                M[:, idx] = e_i  # Default to identity
                    
                    return M
                except Exception as e:
                    print(f"AMG matrix extraction failed: {e}")
            
            # For SSORPreconditioner, build from components
            if precond_type == 'SSORPreconditioner' and all(hasattr(preconditioner, attr) for attr in ['D', 'L', 'U']):
                try:
                    print("Extracting SSOR matrix representation...")
                    import scipy.sparse as sparse
                    
                    # Extract SSOR components
                    D = self._to_numpy(preconditioner.D)
                    L = self._to_numpy(preconditioner.L)
                    U = self._to_numpy(preconditioner.U)
                    omega = preconditioner.omega if hasattr(preconditioner, 'omega') else 1.0
                    
                    if D is not None and L is not None and U is not None:
                        n = D.shape[0]
                        
                        # Sample some columns for visualization
                        sample_size = min(n, 64)
                        indices = np.linspace(0, n-1, sample_size, dtype=int)
                        
                        # Create approximation matrix
                        M = np.zeros((n, sample_size))
                        
                        # Extract D diagonal
                        if hasattr(D, 'diagonal'):
                            D_diag = D.diagonal()
                        else:
                            D_diag = np.diag(D)
                        
                        D_inv_diag = 1.0 / (D_diag + 1e-15)  # Add small value to avoid division by zero
                        
                        # Create identity matrix
                        I = np.eye(n)
                        
                        # Apply SSOR steps directly through __call__ if possible
                        if hasattr(preconditioner, '__call__') and callable(getattr(preconditioner, '__call__')):
                            for idx, i in enumerate(indices):
                                e_i = np.zeros(n)
                                e_i[i] = 1.0
                                try:
                                    M[:, idx] = self._to_numpy(preconditioner(e_i))
                                except Exception as call_err:
                                    print(f"SSOR call error: {call_err}")
                                    M[:, idx] = e_i  # Default to identity
                        else:
                            # Manual SSOR implementation
                            import scipy.sparse.linalg as spla
                            for idx, i in enumerate(indices):
                                e_i = np.zeros(n)
                                e_i[i] = 1.0
                                
                                try:
                                    # Forward sweep: (I + ω D⁻¹ L) z = e_i
                                    L_term = I + omega * np.diag(D_inv_diag) @ L
                                    z = np.linalg.solve(L_term, e_i)
                                    
                                    # Diagonal scaling: y = ω D⁻¹ z
                                    y = omega * D_inv_diag * z
                                    
                                    # Backward sweep: (I + ω D⁻¹ U) x = y
                                    U_term = I + omega * np.diag(D_inv_diag) @ U
                                    M[:, idx] = np.linalg.solve(U_term, y)
                                except Exception as solve_err:
                                    print(f"SSOR solve error: {solve_err}")
                                    M[:, idx] = e_i  # Default to identity
                        
                        return M
                except Exception as e:
                    print(f"SSOR matrix extraction failed: {e}")
            
            # Check for explicit matrix M
            if hasattr(preconditioner, 'M') and preconditioner.M is not None:
                M = preconditioner.M
                print(f"Found explicit M attribute of type: {type(M)}")
                
                # Special handling for SciPy sparse matrix
                import scipy.sparse as sparse
                if isinstance(M, sparse.spmatrix):
                    print(f"Converting sparse matrix of format {M.format} with shape {M.shape}")
                    
                    # Sample the sparse matrix for visualization
                    n = M.shape[0]
                    sample_size = min(n, 64)
                    indices = np.linspace(0, n-1, sample_size, dtype=int)
                    
                    # Extract columns by multiplying with unit vectors
                    M_sample = np.zeros((n, sample_size))
                    for idx, i in enumerate(indices):
                        e_i = np.zeros(n)
                        e_i[i] = 1.0
                        M_sample[:, idx] = M @ e_i
                    
                    return M_sample
                
                return self._to_numpy(M)
            
            # Check for matrix attribute
            if hasattr(preconditioner, 'matrix') and preconditioner.matrix is not None:
                return self._to_numpy(preconditioner.matrix)
            
            # Check for toarray method
            if hasattr(preconditioner, 'toarray') and callable(getattr(preconditioner, 'toarray')):
                return self._to_numpy(preconditioner.toarray())
            
            # Handle diagonal preconditioners (like JacobiPreconditioner)
            if hasattr(preconditioner, 'diag_vals') and preconditioner.diag_vals is not None:
                diag_vals = self._to_numpy(preconditioner.diag_vals)
                if diag_vals is not None:
                    n = len(diag_vals)
                    # Create diagonal matrix efficiently
                    M = np.zeros((n, n))
                    np.fill_diagonal(M, diag_vals)
                    return M
            
            # For other callable preconditioners, sample the action
            if hasattr(preconditioner, '__call__') and callable(getattr(preconditioner, '__call__')):
                try:
                    print("Sampling callable preconditioner...")
                    # Get size from A_size
                    if A_size is not None:
                        n = A_size[0]
                    else:
                        # Try to guess from other attributes
                        n = 128
                        for attr_name in ['D', 'M', 'ml']:
                            if hasattr(preconditioner, attr_name):
                                attr = getattr(preconditioner, attr_name)
                                if hasattr(attr, 'shape'):
                                    n = attr.shape[0]
                                    break
                    
                    # Sample some columns for visualization
                    sample_size = min(n, 64)
                    indices = np.linspace(0, n-1, sample_size, dtype=int)
                    
                    # Create approximation matrix
                    M = np.zeros((n, sample_size))
                    for idx, i in enumerate(indices):
                        e_i = np.zeros(n)
                        e_i[i] = 1.0
                        try:
                            # Apply the preconditioner
                            result = preconditioner(e_i)
                            M[:, idx] = self._to_numpy(result)
                        except Exception as call_err:
                            print(f"Preconditioner call error: {call_err}")
                            M[:, idx] = e_i  # Default to identity
                    
                    return M
                except Exception as e:
                    print(f"General preconditioner sampling failed: {e}")
            
            # Last resort: debug info
            print(f"Could not extract matrix from {precond_type}")
            if hasattr(preconditioner, '__dict__'):
                print(f"Available attributes: {list(preconditioner.__dict__.keys())}")
                
            return None
        except Exception as e:
            print(f"Preconditioner matrix extraction failed: {e}")
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
            # For large matrices, compute a sampled product
            if A.shape[0] > 500 or (hasattr(M, 'shape') and M.shape[0] > 500):
                n = A.shape[0]
                sample_size = min(n, 32)
                indices = np.linspace(0, n-1, sample_size, dtype=int)
                
                # Sample rows and columns for visualization
                A_sample = A[indices, :]
                if hasattr(M, 'shape') and M.shape[1] == n:
                    M_sample = M[:, indices]
                    product = A_sample @ M_sample
                else:
                    # If M doesn't have right shape, sample the action
                    product = np.zeros((sample_size, sample_size))
                    for i, idx_i in enumerate(indices):
                        for j, idx_j in enumerate(indices):
                            e_j = np.zeros(n)
                            e_j[idx_j] = 1.0
                            
                            # Apply M to e_j, then extract idx_i component
                            if hasattr(M, 'shape') and len(M.shape) == 2 and M.shape[1] > idx_j:
                                Me_j = M[:, idx_j]
                            else:
                                # Try calling M as function
                                try:
                                    Me_j = self._to_numpy(M(e_j))
                                except:
                                    Me_j = e_j  # Fallback to identity
                            
                            # Apply A to Me_j, then extract idx_i component
                            product[i, j] = A[idx_i, :] @ Me_j
            else:
                # For smaller matrices, direct multiplication
                product = self._safe_matrix_abs(A @ M)
            
            if product is None or product.size == 0:
                plt.text(0.5, 0.5, "Product Computation Failed", ha='center', va='center')
                return
            
            non_zero = product[product > 0]
            if len(non_zero) == 0:
                plt.imshow(product, cmap='inferno')
            else:
                plt.imshow(product, norm=LogNorm(vmin=non_zero.min(), vmax=non_zero.max()), cmap='inferno')
            
            plt.colorbar(label='Product Magnitude')
            plt.xlabel("Column Index")
            plt.ylabel("Row Index")
        except Exception as e:
            print(f"Matrix product visualization error: {e}")
            plt.text(0.5, 0.5, f"Product Error: {str(e)}", ha='center', va='center')
    
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
        
        # Preconditioner matrix extraction using improved method
        M_np = self._get_preconditioner_matrix(preconditioner, A.shape if hasattr(A, 'shape') else None)
        
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
        
        # Preconditioner info
        if preconditioner:
            precond_name = type(preconditioner).__name__
            stats.append(f"\nPreconditioner: {precond_name}")
            if hasattr(preconditioner, 'description'):
                desc = preconditioner.description
                # Check if description is in Japanese and convert to English if needed
                if "単位行列" in desc:
                    desc = "Identity matrix preconditioner (effectively no preconditioning)"
                stats.append(f"{desc}")
        
        plt.text(0.05, 0.95, '\n'.join(stats), 
                 transform=plt.gca().transAxes, va='top', fontsize=9)
        plt.title("System Statistics")

# Configure plot style
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10