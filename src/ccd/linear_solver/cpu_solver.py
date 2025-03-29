"""
CPU-based linear system solver using SciPy

This module provides an efficient implementation of the LinearSolver
using SciPy's sparse linear algebra solvers.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg import LinearOperator

from .base import LinearSolver


class CPULinearSolver(LinearSolver):
    """CPU-based linear solver implementation using SciPy"""
    
    def _initialize(self) -> None:
        """Initialize CPU-specific properties"""
        # Convert matrix to NumPy/SciPy format
        self.A = self._ensure_scipy_matrix(self.original_A)
        
        # Initialize scaling if specified
        if self.scaling_method:
            self._initialize_scaling()
        
        # Register available solvers
        self.solvers = {
            "direct": self._solve_direct,
            "gmres": self._solve_gmres,
            "lgmres": self._solve_lgmres,
            "cg": self._solve_cg,
            "cgs": self._solve_cgs,
            "bicg": self._solve_bicg,
            "bicgstab": self._solve_bicgstab,
            "qmr": self._solve_qmr,
            "tfqmr": self._solve_tfqmr,
            "minres": self._solve_minres,
            "gcrotmk": self._solve_gcrotmk,
            "lsqr": self._solve_lsqr,
            "lsmr": self._solve_lsmr
        }
    
    def _initialize_scaling(self) -> None:
        """Initialize scaling method"""
        try:
            from scaling import plugin_manager
            self.scaler = plugin_manager.get_plugin(self.scaling_method)
            
            # Compute scaling information
            dummy_b = np.ones(self.A.shape[0])
            _, _, self.scaling_info = self.scaler.scale(self.A, dummy_b)
        except Exception as e:
            print(f"Scaling initialization error: {e}")
            self.scaler = None
            self.scaling_info = None
    
    def _ensure_scipy_matrix(self, A: Any) -> Any:
        """Convert matrix to SciPy/NumPy format from any backend"""
        # Handle CuPy arrays
        if hasattr(A, 'get'):
            return A.get()
        
        # Handle JAX arrays
        if 'jax' in str(type(A)):
            return np.array(A)
            
        # Return as is if already in NumPy/SciPy format
        return A
    
    def _preprocess_vector(self, b: Any) -> np.ndarray:
        """Convert vector to NumPy format from any backend"""
        # Handle CuPy arrays
        if hasattr(b, 'get'):
            return b.get()
        
        # Handle JAX arrays
        if 'jax' in str(type(b)):
            return np.array(b)
        
        # Return as is if already in NumPy format
        return b
    
    def _to_numpy_matrix(self, A: Any) -> np.ndarray:
        """Convert matrix to NumPy format (identity for CPU solver)"""
        return self._ensure_scipy_matrix(A)
    
    # ===== Solver implementations =====
    
    def _solve_direct(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, None]:
        """Direct solver implementation using spsolve"""
        return splinalg.spsolve(A, b), None
    
    def _solve_gmres(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """GMRES solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 200)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.gmres(A, b, x0=x0, rtol=tol, maxiter=maxiter, restart=restart, M=M)
        return result[0], result[1]
    
    def _solve_lgmres(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """LGMRES solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        inner_m = options.get("inner_m", 30)
        outer_k = options.get("outer_k", 3)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.lgmres(A, b, x0=x0, rtol=tol, maxiter=maxiter, 
                                inner_m=inner_m, outer_k=outer_k, M=M)
        return result[0], result[1]
    
    def _solve_cg(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Conjugate Gradient solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_bicg(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """BiCG solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.bicg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_bicgstab(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """BiCGSTAB solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.bicgstab(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_cgs(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """CGS solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.cgs(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_qmr(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """QMR solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        # QMR supports different left/right preconditioners
        M1, M2 = M, None
        
        # Solve
        result = splinalg.qmr(A, b, x0=x0, rtol=tol, maxiter=maxiter, M1=M1, M2=M2)
        return result[0], result[1]
    
    def _solve_tfqmr(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """TFQMR solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.tfqmr(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_minres(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """MINRES solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.minres(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_gcrotmk(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """GCROT(m,k) solver implementation"""
        # Extract options with defaults
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        m = options.get("m", 20)
        k = options.get("k", 10)
        x0 = options.get("x0", np.zeros_like(b))
        
        # Get preconditioner
        M = self._create_preconditioner_operator()
        
        # Solve
        result = splinalg.gcrotmk(A, b, x0=x0, rtol=tol, maxiter=maxiter, m=m, k=k, M=M)
        return result[0], result[1]
    
    def _solve_lsqr(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """LSQR solver implementation (no preconditioner support)"""
        # Extract options with defaults
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        iter_lim = options.get("maxiter", options.get("iter_lim", None))
        
        # Solve (note: LSQR doesn't support preconditioners)
        result = splinalg.lsqr(A, b, damp=damp, atol=atol, btol=btol,
                             conlim=conlim, iter_lim=iter_lim)
        return result[0], result[2]  # results[2] contains iteration count

    def _solve_lsmr(self, A: Any, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """LSMR solver implementation (no preconditioner support)"""
        # Extract options with defaults
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        maxiter = options.get("maxiter", None)
        x0 = options.get("x0", None)
        
        # Solve (note: LSMR doesn't support preconditioners)
        result = splinalg.lsmr(A, b, damp=damp, atol=atol, btol=btol,
                             conlim=conlim, maxiter=maxiter, x0=x0)
        return result[0], result[2]  # results[2] contains iteration count