"""
JAX-based linear system solver 

This module provides an implementation of the LinearSolver
using JAX's hardware-accelerated linear algebra.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np

from .base import LinearSolver
from .cpu_solver import CPULinearSolver


class JAXLinearSolver(LinearSolver):
    """JAX-based linear solver implementation"""
    
    def _initialize(self) -> None:
        """Initialize JAX-specific properties"""
        try:
            import jax
            import jax.numpy as jnp
            import jax.scipy.sparse.linalg as splinalg
            
            # Enable high-precision and prioritize GPU
            jax.config.update('jax_enable_x64', True)
            jax.config.update('jax_platform_name', 'gpu')
            
            self.jax = jax
            self.jnp = jnp
            self.splinalg = splinalg
            self.has_jax = True
            
            # Convert matrix to JAX format
            self.A = self._to_jax_matrix(self.original_A)
            
            # Initialize scaling if specified
            if self.scaling_method:
                self._initialize_scaling()
            
            # Register available solvers
            self.solvers = {
                "direct": self._solve_direct,
                "gmres": self._solve_gmres,
                "cg": self._solve_cg,
                "bicgstab": self._solve_bicgstab
            }
            
            # Set up preconditioner if provided
            self._setup_preconditioner()
            
        except ImportError as e:
            print(f"Warning: JAX not available: {e}")
            self.has_jax = False
            self._init_cpu_fallback()
        except Exception as e:
            print(f"JAX initialization error: {e}")
            self.has_jax = False
            self._init_cpu_fallback()
    
    def _init_cpu_fallback(self) -> None:
        """Initialize CPU fallback when JAX is not available"""
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
        if not self.has_jax:
            return
            
        try:
            from scaling import plugin_manager
            self.scaler = plugin_manager.get_plugin(self.scaling_method)
            
            # Use NumPy for scaling info computation
            dummy_b = np.ones(self.A.shape[0])
            A_np = self._to_numpy_matrix(self.A)
            _, _, scale_info = self.scaler.scale(A_np, dummy_b)
            
            # Convert scaling info to JAX arrays
            self.scaling_info = {}
            for key, value in scale_info.items():
                if isinstance(value, np.ndarray):
                    self.scaling_info[key] = self.jnp.array(value)
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
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
        # Prepare options for JAX
        if options and "x0" in options:
            options = options.copy()
            try:
                options["x0"] = self._preprocess_vector(options["x0"])
                print(f"Set JAX x0 (shape: {options['x0'].shape})")
            except Exception as e:
                print(f"x0 conversion error: {e}")
                del options["x0"]
        
        # Ensure preconditioner is set up
        self._setup_preconditioner()
        
        # Delegate to parent class implementation
        return super().solve(b, method, options)
    
    def _to_jax_matrix(self, A: Any) -> Any:
        """Convert matrix to JAX format"""
        if not self.has_jax:
            return A
            
        try:
            # Check if already in JAX
            if 'jax' in str(type(A)):
                return A
                
            # Handle None
            if A is None:
                return None
            
            # Handle linear operators
            if hasattr(A, 'matvec'):
                from jax.scipy.sparse.linalg import LinearOperator
                
                # Create a JAX-compatible matvec function
                def jax_matvec(x):
                    # Convert to CPU if needed
                    if 'jax' in str(type(x)):
                        x_cpu = np.array(x)
                    else:
                        x_cpu = x
                    
                    # Execute on CPU
                    y_cpu = A.matvec(x_cpu)
                    
                    # Convert result to JAX
                    return self.jnp.array(y_cpu)
                
                return LinearOperator(A.shape, matvec=jax_matvec)
            
            # Convert to dense array
            # Note: JAX doesn't fully support sparse matrices like SciPy/CuPy
            if hasattr(A, 'toarray'):
                A_dense = A.toarray()
            else:
                A_dense = np.array(A)
            
            # Clean up small values for numerical stability
            A_dense[np.abs(A_dense) < 1e-15] = 0.0
            
            # Handle large matrices in chunks to avoid memory issues
            if A_dense.size > 1e7:  # ~100MB
                print("Processing large matrix in chunks...")
                if len(A_dense.shape) == 2:
                    rows = []
                    chunk_size = min(1000, A_dense.shape[0])
                    for i in range(0, A_dense.shape[0], chunk_size):
                        end = min(i + chunk_size, A_dense.shape[0])
                        rows.append(self.jnp.array(A_dense[i:end]))
                    return self.jnp.vstack(rows)
                else:
                    return self.jnp.array(A_dense)
            else:
                return self.jnp.array(A_dense)
                
        except Exception as e:
            print(f"JAX matrix conversion error: {e}")
            print("Falling back to CPU solver")
            self.has_jax = False
            self._init_cpu_fallback()
            return A
    
    def _preprocess_vector(self, b: Any) -> Any:
        """Convert vector to JAX format"""
        if not self.has_jax:
            return b
            
        # Check if already in JAX
        if 'jax' in str(type(b)):
            return b
        
        try:
            # Convert from CuPy if needed
            if hasattr(b, 'get'):
                return self.jnp.array(b.get())
            # Convert from NumPy or other formats
            return self.jnp.array(b)
        except Exception as e:
            print(f"Vector conversion error: {e}")
            self.has_jax = False
            self._init_cpu_fallback()
            return b
    
    def _apply_scaling_to_b(self, b: Any) -> Any:
        """Apply scaling to the right-hand side vector"""
        if not self.has_jax:
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
        if not self.has_jax:
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
        """Convert JAX matrix to NumPy format"""
        if 'jax' in str(type(A)):
            return np.array(A)
        return A
    
    def _to_numpy_vector(self, b: Any) -> np.ndarray:
        """Convert JAX vector to NumPy format"""
        if 'jax' in str(type(b)):
            return np.array(b)
        return b
    
    def _setup_preconditioner(self) -> None:
        """Set up preconditioner and convert to JAX format if needed"""
        if not self.has_jax or not self.preconditioner or not hasattr(self.preconditioner, 'setup'):
            return
            
        # Skip if already set up
        if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
            # Check if we need to convert to JAX
            if 'jax' not in str(type(self.preconditioner.M)):
                self._convert_preconditioner_to_jax()
            return
            
        try:
            # Convert matrix to CPU for preconditioner setup
            A_cpu = self._to_numpy_matrix(self.A)
            self.preconditioner.setup(A_cpu)
            print(f"Set up preconditioner: {self.preconditioner.name}")
            
            # Convert to JAX format
            self._convert_preconditioner_to_jax()
        except Exception as e:
            print(f"Preconditioner setup error: {e}")
    
    def _convert_preconditioner_to_jax(self) -> None:
        """Convert preconditioner to JAX format"""
        if not hasattr(self.preconditioner, 'M') or self.preconditioner.M is None:
            return
            
        try:
            M = self.preconditioner.M
            
            # Convert matrix-based preconditioners
            if hasattr(M, 'toarray') or hasattr(M, 'todense'):
                # Convert to dense first
                M_cpu = M.toarray() if hasattr(M, 'toarray') else M.todense()
                
                # Convert to JAX array
                M_jax = self.jnp.array(M_cpu)
                
                # Update preconditioner
                self.preconditioner.M = M_jax
                print(f"Converted preconditioner matrix to JAX (shape: {M_jax.shape})")
            
            # Convert LinearOperator-based preconditioners
            elif hasattr(M, 'matvec'):
                from jax.scipy.sparse.linalg import LinearOperator
                
                # Create JAX-compatible matvec function
                def jax_matvec(x):
                    # Convert to CPU if needed
                    if 'jax' in str(type(x)):
                        x_cpu = np.array(x)
                    else:
                        x_cpu = x
                    
                    # Apply preconditioner on CPU
                    y_cpu = M.matvec(x_cpu)
                    
                    # Convert result to JAX
                    return self.jnp.array(y_cpu)
                
                # Create new LinearOperator
                self.preconditioner.M = LinearOperator(M.shape, matvec=jax_matvec)
                print("Converted LinearOperator preconditioner to JAX")
        except Exception as e:
            print(f"Preconditioner JAX conversion error: {e}")
    
    def _create_preconditioner_operator(self) -> Optional[Any]:
        """Create preconditioner operator for iterative solvers"""
        if not self.has_jax or not self.preconditioner:
            return None
            
        try:
            # Use matrix directly if available
            if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
                M = self.preconditioner.M
                
                # If already in JAX, return as is
                if 'jax' in str(type(M)):
                    return M
                
                # Convert to JAX if needed
                print("Converting CPU preconditioner to JAX")
                self._convert_preconditioner_to_jax()
                M = self.preconditioner.M
                
                if 'jax' in str(type(M)):
                    return M
                
                # Create wrapper for non-JAX operators
                if hasattr(M, 'matvec'):
                    from jax.scipy.sparse.linalg import LinearOperator
                    
                    def jax_matvec(x):
                        try:
                            if 'jax' in str(type(x)):
                                x_cpu = np.array(x)
                            else:
                                x_cpu = x
                            
                            # Apply preconditioner on CPU
                            y_cpu = M.matvec(x_cpu)
                            
                            # Convert result to JAX
                            return self.jnp.array(y_cpu)
                        except Exception as e:
                            print(f"Preconditioner matvec error: {e}")
                            return x
                    
                    return LinearOperator(self.A.shape, matvec=jax_matvec)
            
            # Use __call__ method if available
            elif hasattr(self.preconditioner, '__call__'):
                from jax.scipy.sparse.linalg import LinearOperator
                
                # Create JAX-compatible function
                def preconditioner_func(x):
                    try:
                        # Convert to CPU
                        if 'jax' in str(type(x)):
                            x_cpu = np.array(x)
                        else:
                            x_cpu = x
                        
                        # Apply preconditioner on CPU
                        y_cpu = self.preconditioner(x_cpu)
                        
                        # Convert result to JAX
                        return self.jnp.array(y_cpu)
                    except Exception as e:
                        print(f"Preconditioner application error: {e}")
                        return x
                
                return LinearOperator(self.A.shape, matvec=preconditioner_func)
        
        except Exception as e:
            print(f"JAX preconditioner operator creation error: {e}")
        
        return None
    
    # ===== Solver implementations =====
    
    def _solve_direct(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, None]:
        """Direct solver implementation for JAX"""
        try:
            # Check if JAX has a direct sparse solver
            if hasattr(self.jax.scipy.linalg, 'solve'):
                x = self.jax.scipy.linalg.solve(A, b, assume_a='gen')
                return x, None
            else:
                # Use CG as fallback (JAX lacks direct sparse solvers)
                print("JAX direct solver not available, using CG with high precision")
                options_copy = options.copy() if options else {}
                options_copy["tol"] = options_copy.get("tol", 1e-12)
                return self._solve_cg(A, b, options_copy)
        except Exception as e:
            print(f"JAX direct solver error: {e}")
            return self._direct_fallback(A, b)
    
    def _direct_fallback(self, A: Any, b: Any) -> Tuple[Any, None]:
        """CPU fallback for direct solver"""
        import scipy.sparse.linalg as cpu_splinalg
        A_cpu = self._to_numpy_matrix(A)
        b_cpu = self._to_numpy_vector(b)
        x_cpu = cpu_splinalg.spsolve(A_cpu, b_cpu)
        return self.jnp.array(x_cpu), None
    
    def _solve_gmres(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """GMRES solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(20, max(5, b.size // 20)))
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, restart=restart, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"GMRES error: {e}")
            print("Retrying GMRES without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, restart=restart)
            return result[0], result[1]
    
    def _solve_cg(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """Conjugate Gradient solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"CG error: {e}")
            print("Retrying CG without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]
    
    def _solve_bicgstab(self, A: Any, b: Any, options: Dict[str, Any]) -> Tuple[Any, int]:
        """BiCGSTAB solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        try:
            # Solve
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"BiCGSTAB error: {e}")
            print("Retrying BiCGSTAB without preconditioner")
            # Retry without preconditioner
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]