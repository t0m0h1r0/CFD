"""
GPU-based linear system solver using CuPy

This module provides an efficient implementation of the LinearSolver
using CuPy's GPU-accelerated sparse linear algebra solvers.
"""

from typing import Dict, Any, Tuple, Optional, List, Union, Callable
import numpy as np

from .base import LinearSolver
from .cpu_solver import CPULinearSolver


class GPULinearSolver(LinearSolver):
    """GPU-based linear solver implementation using CuPy"""
    
    def _initialize(self) -> None:
        """Initialize GPU-specific properties"""
        # Import CuPy and initialize capabilities
        try:
            import cupy as cp
            import cupyx.scipy.sparse.linalg as splinalg
            self.cp = cp
            self.splinalg = splinalg
            self.has_cupy = True
            
            # Convert matrix to CuPy format
            self.A = self._to_cupy_matrix(self.original_A)
            
            # Initialize scaling if specified
            if self.scaling_method:
                self._initialize_scaling()
            
            # Register available solvers
            self.solvers = {
                "direct": self._solve_direct,
                "gmres": self._solve_gmres,
                "cg": self._solve_cg,
                "cgs": self._solve_cgs,
                "minres": self._solve_minres,
                "bicgstab": self._solve_bicgstab,
                "lsqr": self._solve_lsqr,
                "lsmr": self._solve_lsmr
            }
            
            # Set up preconditioner if provided
            self._setup_preconditioner()
            
        except ImportError as e:
            print(f"Warning: CuPy not available: {e}")
            self.has_cupy = False
            self._init_cpu_fallback()
        except Exception as e:
            print(f"GPU initialization error: {e}")
            self.has_cupy = False
            self._init_cpu_fallback()
    
    def _init_cpu_fallback(self) -> None:
        """Initialize CPU fallback when CuPy is not available"""
        self.cpu_solver = CPULinearSolver(
            self.original_A, 
            self.enable_dirichlet, 
            self.enable_neumann, 
            self.scaling_method,
            self.preconditioner_name if self.preconditioner_name else self.preconditioner
        )
        # Copy solvers from CPU implementation
        self.solvers = self.cpu_solver.solvers
        
        # Copy preconditioner from CPU implementation
        if hasattr(self.cpu_solver, 'preconditioner'):
            self.preconditioner = self.cpu_solver.preconditioner
    
    def _initialize_scaling(self) -> None:
        """Initialize scaling method"""
        if not self.has_cupy:
            return
            
        try:
            from scaling import plugin_manager
            self.scaler = plugin_manager.get_plugin(self.scaling_method)
            
            # Use NumPy for scaling info computation
            dummy_b = np.ones(self.A.shape[0])
            A_np = self._to_numpy_matrix(self.A)
            _, _, scale_info = self.scaler.scale(A_np, dummy_b)
            
            # Convert scaling info to CuPy
            self.scaling_info = {}
            for key, value in scale_info.items():
                if isinstance(value, np.ndarray):
                    self.scaling_info[key] = self.cp.array(value)
                else:
                    self.scaling_info[key] = value
                    
        except Exception as e:
            print(f"Scaling initialization error: {e}")
            self.scaler = None
            self.scaling_info = None
    
    def solve(self, 
              b: Any, 
              method: Optional[str] = None, 
              options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Solve the linear system Ax=b (with CPU fallback)
        
        Args:
            b: Right-hand side vector
            method: Solver method (overrides self.solver_method if provided)
            options: Solver options (updates self.solver_options if provided)
            
        Returns:
            Solution vector x
        """
        if not self.has_cupy:
            return self.cpu_solver.solve(b, method, options)
        
        # Prepare options for GPU
        if options and "x0" in options:
            options = options.copy()
            try:
                options["x0"] = self._preprocess_vector(options["x0"])
                print(f"Set CuPy x0 (shape: {options['x0'].shape})")
            except Exception as e:
                print(f"x0 conversion error: {e}")
                del options["x0"]
        
        # Ensure preconditioner is set up
        self._setup_preconditioner()
        
        # Delegate to parent class implementation
        return super().solve(b, method, options)
    
    def _to_cupy_matrix(self, A: Any) -> Any:
        """Convert matrix to CuPy format"""
        if not self.has_cupy:
            return A
            
        try:
            # Check if already on GPU
            if hasattr(A, 'device') and str(type(A)).find('cupy') >= 0:
                return A
            
            # Handle None
            if A is None:
                return None
                
            # Handle sparse matrices
            if hasattr(A, 'format'):
                # Ensure CSR format for efficiency
                if A.format != 'csr':
                    A = A.tocsr()
                
                # Convert CSR matrix to CuPy
                return self.cp.sparse.csr_matrix(
                    (self.cp.array(A.data), 
                     self.cp.array(A.indices), 
                     self.cp.array(A.indptr)),
                    shape=A.shape
                )
                
            # Handle linear operators
            elif hasattr(A, 'matvec'):
                from cupyx.scipy.sparse.linalg import LinearOperator
                
                # Create a CuPy-compatible matvec function
                def gpu_matvec(x):
                    # Convert to CPU if needed
                    if hasattr(x, 'get'):
                        x_cpu = x.get()
                    else:
                        x_cpu = np.array(x)
                    
                    # Execute on CPU
                    y_cpu = A.matvec(x_cpu)
                    
                    # Convert result back to GPU
                    return self.cp.array(y_cpu)
                
                return LinearOperator(A.shape, matvec=gpu_matvec)
                
            # Handle dense matrices
            else:
                # Convert to dense format if needed
                if hasattr(A, 'toarray'):
                    A = A.toarray()
                
                # Convert to CuPy CSR matrix
                return self.cp.sparse.csr_matrix(self.cp.array(A))
            
        except Exception as e:
            print(f"GPU matrix conversion error: {e}")
            print("Falling back to CPU solver")
            self.has_cupy = False
            self._init_cpu_fallback()
            return A
    
    def _preprocess_vector(self, b: Any) -> Any:
        """Convert vector to CuPy format"""
        if not self.has_cupy:
            return b
            
        # Check if already a CuPy array
        if hasattr(b, 'device') and str(type(b)).find('cupy') >= 0:
            return b
        
        try:
            # Convert to CuPy array
            return self.cp.array(b)
        except Exception as e:
            print(f"Vector conversion error: {e}")
            self.has_cupy = False
            self._init_cpu_fallback()
            return b
    
    def _apply_scaling_to_b(self, b: Any) -> Any:
        """Apply scaling to the right-hand side vector"""
        if not self.has_cupy:
            return self.cpu_solver._apply_scaling_to_b(b)
            
        if self.scaler and self.scaling_info:
            row_scale = self.scaling_info.get('row_scale')
            if row_scale is not None:
                return b * row_scale
            
            D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
            if D_sqrt_inv is not None:
                return b * D_sqrt_inv
                
        return b
    
    def _apply_unscaling_to_x(self, x: Any) -> Any:
        """Apply unscaling to the solution vector"""
        if not self.has_cupy:
            return self.cpu_solver._apply_unscaling_to_x(x)
            
        if self.scaler and self.scaling_info:
            col_scale = self.scaling_info.get('col_scale')
            if col_scale is not None:
                return x / col_scale
            
            D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
            if D_sqrt_inv is not None:
                return x * D_sqrt_inv
                
        return x
    
    def _to_numpy_matrix(self, A: Any) -> np.ndarray:
        """Convert CuPy matrix to NumPy format"""
        if hasattr(A, 'get'):
            # For CuPy sparse matrices
            if hasattr(A, 'format') and A.format == 'csr':
                from scipy import sparse
                return sparse.csr_matrix(
                    (A.data.get(), A.indices.get(), A.indptr.get()),
                    shape=A.shape
                )
            # For dense CuPy arrays
            return A.get()
        
        # Return as is if already NumPy
        return A
    
    def _to_numpy_vector(self, b: Any) -> np.ndarray:
        """Convert CuPy vector to NumPy format"""
        if hasattr(b, 'get'):
            return b.get()
        return b
    
    def _setup_preconditioner(self) -> None:
        """Set up preconditioner and convert to GPU format if needed"""
        if not self.has_cupy or not self.preconditioner or not hasattr(self.preconditioner, 'setup'):
            return
            
        # Skip if already set up
        if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
            # Check if we need to convert to GPU
            if not hasattr(self.preconditioner.M, 'device'):
                self._convert_preconditioner_to_gpu()
            return
            
        try:
            # Convert matrix to CPU for preconditioner setup
            A_cpu = self._to_numpy_matrix(self.A)
            self.preconditioner.setup(A_cpu)
            print(f"Set up preconditioner: {self.preconditioner.name}")
            
            # Convert to GPU format
            self._convert_preconditioner_to_gpu()
        except Exception as e:
            print(f"Preconditioner setup error: {e}")
    
    def _convert_preconditioner_to_gpu(self) -> None:
        """Convert preconditioner to GPU format"""
        if not hasattr(self.preconditioner, 'M') or self.preconditioner.M is None:
            return
            
        try:
            M = self.preconditioner.M
            
            # Convert matrix-based preconditioners
            if hasattr(M, 'toarray') or hasattr(M, 'todense'):
                # Convert to dense first
                M_cpu = M.toarray() if hasattr(M, 'toarray') else M.todense()
                M_gpu = self.cp.array(M_cpu)
                
                # Convert back to sparse if needed
                if hasattr(M, 'format') and M.format == 'csr':
                    from cupyx.scipy.sparse import csr_matrix
                    M_gpu = csr_matrix(M_gpu)
                
                # Update preconditioner
                self.preconditioner.M = M_gpu
                print(f"Converted preconditioner matrix to GPU (shape: {M_gpu.shape})")
            
            # Convert LinearOperator-based preconditioners
            elif hasattr(M, 'matvec'):
                from cupyx.scipy.sparse.linalg import LinearOperator
                
                # Create GPU-compatible matvec function
                def gpu_matvec(x):
                    # Convert to CPU if needed
                    if hasattr(x, 'get'):
                        x_cpu = x.get()
                    else:
                        x_cpu = np.array(x)
                    
                    # Apply preconditioner on CPU
                    y_cpu = M.matvec(x_cpu)
                    
                    # Convert result back to GPU
                    return self.cp.array(y_cpu)
                
                # Create new LinearOperator
                self.preconditioner.M = LinearOperator(M.shape, matvec=gpu_matvec)
                print("Converted LinearOperator preconditioner to GPU")
        except Exception as e:
            print(f"Preconditioner GPU conversion error: {e}")
    
    def _create_preconditioner_operator(self) -> Optional[Any]:
        """Create preconditioner operator for iterative solvers"""
        if not self.has_cupy or not self.preconditioner:
            return None
            
        try:
            # Use matrix directly if available
            if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
                M = self.preconditioner.M
                
                # If already on GPU, return as is
                if hasattr(M, 'device') or 'cupy' in str(type(M)):
                    return M
                
                # Convert to GPU if needed
                print("Converting CPU preconditioner to GPU")
                self._convert_preconditioner_to_gpu()
                M = self.preconditioner.M
                
                if hasattr(M, 'device') or 'cupy' in str(type(M)):
                    return M
                
                # Create wrapper for non-GPU operators
                if hasattr(M, 'matvec'):
                    from cupyx.scipy.sparse.linalg import LinearOperator
                    
                    def gpu_matvec(x):
                        try:
                            if hasattr(x, 'get'):
                                x_cpu = x.get()
                            else:
                                x_cpu = np.array(x)
                            
                            # Apply preconditioner on CPU
                            y_cpu = M.matvec(x_cpu)
                            
                            # Convert result back to GPU
                            return self.cp.array(y_cpu)
                        except Exception as e:
                            print(f"Preconditioner matvec error: {e}")
                            return x
                    
                    return LinearOperator(self.A.shape, matvec=gpu_matvec)
            
            # Use __call__ method if available
            elif hasattr(self.preconditioner, '__call__'):
                from cupyx.scipy.sparse.linalg import LinearOperator
                
                # Create GPU-compatible function
                def preconditioner_func(x):
                    try:
                        # Convert to CPU
                        if hasattr(x, 'get'):
                            x_cpu = x.get()
                        else:
                            x_cpu = np.array(x)
                        
                        # Apply preconditioner on CPU
                        y_cpu = self.preconditioner(x_cpu)
                        
                        # Convert result back to GPU
                        return self.cp.array(y_cpu)
                    except Exception as e:
                        print(f"Preconditioner application error: {e}")
                        return x
                
                return LinearOperator(self.A.shape, matvec=preconditioner_func)
        
        except Exception as e:
            print(f"GPU preconditioner operator creation error: {e}")
        
        return None
    
    # ===== Solver implementations =====
    
    def _solve_direct(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, None]:
        """Direct solver implementation using CuPy's spsolve"""
        try:
            x = self.splinalg.spsolve(A, b)
            return x, None
        except Exception as e:
            print(f"Direct solver error: {e}")
            return self._direct_fallback(A, b)
    
    def _direct_fallback(self, A: Any, b: Any) -> Tuple[Any, None]:
        """CPU fallback for direct solver"""
        import scipy.sparse.linalg as cpu_splinalg
        A_cpu = self._to_numpy_matrix(A)
        b_cpu = self._to_numpy_vector(b)
        x_cpu = cpu_splinalg.spsolve(A_cpu, b_cpu)
        return self.cp.array(x_cpu), None
    
    def _solve_gmres(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """GMRES solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(200, max(20, b.size // 10)))
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"GMRES error: {e}")
            print("Retrying GMRES without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
            return result[0], result[1]
    
    def _solve_cg(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """Conjugate Gradient solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"CG error: {e}")
            print("Retrying CG without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]
    
    def _solve_cgs(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """CGS solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"CGS error: {e}")
            print("Retrying CGS without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]
    
    def _solve_minres(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """MINRES solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"MINRES error: {e}")
            print("Retrying MINRES without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]
    
    def _solve_bicgstab(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """BiCGSTAB solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"BiCGSTAB error: {e}")
            print("Retrying BiCGSTAB without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]
    
    def _solve_lsqr(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """LSQR solver implementation (no preconditioner support)"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # Note: LSQR doesn't support preconditioners
        # Note: LSQR interface might differ between SciPy and CuPy
        try:
            # CuPy's LSQR might have a different interface
            result = self.splinalg.lsqr(A, b, atol=tol, btol=tol, iter_lim=maxiter)
            return result[0], result[2]  # results[2] contains iteration count
        except (AttributeError, TypeError) as e:
            print(f"LSQR interface error, adjusting parameters: {e}")
            # Try different parameter combinations
            try:
                result = self.splinalg.lsqr(A, b)
                return result[0], result[2]
            except Exception as e2:
                print(f"LSQR fallback error: {e2}")
                # Fall back to CPU
                return self._lsqr_cpu_fallback(A, b, options)
    
    def _lsqr_cpu_fallback(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """CPU fallback for LSQR"""
        import scipy.sparse.linalg as cpu_splinalg
        # Convert to CPU
        A_cpu = self._to_numpy_matrix(A)
        b_cpu = self._to_numpy_vector(b)
        
        # Extract options
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # Solve using CPU
        result = cpu_splinalg.lsqr(A_cpu, b_cpu, atol=tol, btol=tol, iter_lim=maxiter)
        # Convert result back to GPU
        return self.cp.array(result[0]), result[2]
    
    def _solve_lsmr(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """LSMR solver implementation (no preconditioner support)"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # Note: LSMR doesn't support preconditioners
        try:
            # LSMR might have a different interface
            result = self.splinalg.lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
            return result[0], result[2]
        except Exception as e:
            print(f"LSMR error: {e}")
            # Fall back to CPU
            import scipy.sparse.linalg as cpu_splinalg
            A_cpu = self._to_numpy_matrix(A)
            b_cpu = self._to_numpy_vector(b)
            result = cpu_splinalg.lsmr(A_cpu, b_cpu, atol=tol, btol=tol, maxiter=maxiter)
            return self.cp.array(result[0]), result[2]