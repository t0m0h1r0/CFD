"""
Linear system solver base class for Ax=b systems

This module provides the abstract base class for all linear solvers
with common functionality for solving linear systems with various methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union, Callable, List
import numpy as np


class LinearSolver(ABC):
    """Abstract base class for solving linear systems Ax=b"""
    
    def __init__(self, 
                 A: Any, 
                 enable_dirichlet: bool = False, 
                 enable_neumann: bool = False, 
                 scaling_method: Optional[str] = None, 
                 preconditioner: Optional[Union[str, Any]] = None):
        """
        Initialize the solver with the system matrix and options
        
        Args:
            A: System matrix
            enable_dirichlet: Whether to use Dirichlet boundary conditions
            enable_neumann: Whether to use Neumann boundary conditions
            scaling_method: Name of scaling method (optional)
            preconditioner: Preconditioner name or instance (optional)
        """
        self.original_A = A
        self.A = None  # Will be set in _initialize()
        self.enable_dirichlet = enable_dirichlet
        self.enable_neumann = enable_neumann
        
        # Solver configuration
        self.solver_method = "direct"  # Default method
        self.solver_options: Dict[str, Any] = {}
        
        # Scaling configuration
        self.scaling_method = scaling_method
        self.scaler = None
        self.scaling_info = None
        
        # Preconditioner configuration
        self.preconditioner_name = preconditioner if isinstance(preconditioner, str) else None
        self.preconditioner = preconditioner if not isinstance(preconditioner, str) else None
        
        # Solver state
        self.last_iterations = None
        self.solvers: Dict[str, Callable] = {}
        
        # Initialize preconditioner and backend-specific properties
        self._initialize_preconditioner()
        self._initialize()
    
    def _initialize_preconditioner(self) -> None:
        """Initialize preconditioner from name if provided"""
        if not self.preconditioner_name:
            return
            
        try:
            from preconditioner import plugin_manager
            self.preconditioner = plugin_manager.get_plugin(self.preconditioner_name)
            print(f"Initialized preconditioner '{self.preconditioner.name}'")
        except ImportError:
            print("Warning: preconditioner module not found")
            self.preconditioner = None
        except Exception as e:
            print(f"Preconditioner initialization error: {e}")
            self.preconditioner = None
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize backend-specific properties"""
        pass
    
    def solve(self, 
              b: Any, 
              method: Optional[str] = None, 
              options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Solve the linear system Ax=b
        
        Args:
            b: Right-hand side vector
            method: Solver method (overrides self.solver_method if provided)
            options: Solver options (updates self.solver_options if provided)
            
        Returns:
            Solution vector x
        """
        # Determine method and options
        actual_method = method if method is not None else self.solver_method
        actual_options = self.solver_options.copy()
        if options:
            actual_options.update(options)
        
        # Preprocess the right-hand side vector
        try:
            b_processed = self._preprocess_vector(b)
        except Exception as e:
            print(f"Vector preprocessing error: {e}")
            b_processed = b
            
        # Apply scaling if configured
        try:
            b_scaled = self._apply_scaling_to_b(b_processed)
        except Exception as e:
            print(f"Scaling error: {e}")
            b_scaled = b_processed
        
        # Validate method
        if actual_method not in self.solvers:
            print(f"Unsupported solver: {actual_method}, falling back to direct solver")
            actual_method = "direct"
        
        # Setup preconditioner if needed
        self._setup_preconditioner()
        
        # Solve the linear system with error handling
        try:
            solver_func = self.solvers[actual_method]
            x, iterations = solver_func(self.A, b_scaled, actual_options)
            
            # Apply unscaling
            try:
                x_unscaled = self._apply_unscaling_to_x(x)
            except Exception as e:
                print(f"Unscaling error: {e}")
                x_unscaled = x
            
            # Record results
            self.last_iterations = iterations
            return x_unscaled
            
        except Exception as e:
            print(f"Solver error [{actual_method}]: {e}")
            
            # Try direct method as fallback
            if actual_method != "direct":
                try:
                    print("Falling back to direct solver...")
                    x, _ = self._direct_fallback(self.A, b_scaled)
                    
                    # Apply unscaling
                    try:
                        x_unscaled = self._apply_unscaling_to_x(x)
                    except Exception:
                        x_unscaled = x
                        
                    return x_unscaled
                except Exception as fallback_error:
                    print(f"Direct solver fallback error: {fallback_error}")
            
            # Re-raise if all fallbacks failed
            raise
    
    def _setup_preconditioner(self) -> None:
        """Set up preconditioner with matrix A if not already configured"""
        if not self.preconditioner or not hasattr(self.preconditioner, 'setup'):
            return
            
        # Skip if already set up
        if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
            return
            
        try:
            # Convert A to CPU format for preconditioner setup
            A_cpu = self._to_numpy_matrix(self.A)
            self.preconditioner.setup(A_cpu)
            print(f"Set up preconditioner: {self.preconditioner.name}")
        except Exception as e:
            print(f"Preconditioner setup error: {e}")
    
    def _create_preconditioner_operator(self) -> Optional[Any]:
        """
        Create a preconditioner operator for iterative solvers
        
        Returns:
            Preconditioner operator or None
        """
        if not self.preconditioner:
            return None
            
        # Matrix-based preconditioner
        if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
            return self.preconditioner.M
            
        # Function-based preconditioner
        if hasattr(self.preconditioner, '__call__'):
            return self.preconditioner
            
        return None
    
    def set_solver(self, 
                   method: str = "direct", 
                   options: Optional[Dict[str, Any]] = None, 
                   scaling_method: Optional[str] = None, 
                   preconditioner: Optional[Union[str, Any]] = None) -> None:
        """
        Set solver configuration
        
        Args:
            method: Solver method
            options: Solver options
            scaling_method: Scaling method name
            preconditioner: Preconditioner name or instance
        """
        self.solver_method = method
        self.solver_options = options or {}
        
        if scaling_method is not None and scaling_method != self.scaling_method:
            self.scaling_method = scaling_method
            self._initialize_scaling()
            
        if preconditioner is not None:
            if isinstance(preconditioner, str):
                if preconditioner != self.preconditioner_name:
                    self.preconditioner_name = preconditioner
                    self._initialize_preconditioner()
            else:
                self.preconditioner = preconditioner
                self.preconditioner_name = None
    
    def _initialize_scaling(self) -> None:
        """Initialize scaling method (to be implemented by subclasses)"""
        pass
    
    def _direct_fallback(self, A: Any, b: Any) -> Tuple[Any, Optional[int]]:
        """Direct method fallback implementation"""
        if "direct" in self.solvers:
            return self.solvers["direct"](A, b, {})
        raise NotImplementedError("No direct solver available for fallback")
    
    def _apply_scaling_to_b(self, b: Any) -> Any:
        """Apply scaling to right-hand side vector"""
        if self.scaler and self.scaling_info:
            return self.scaler.scale_b_only(b, self.scaling_info)
        return b
    
    def _apply_unscaling_to_x(self, x: Any) -> Any:
        """Apply unscaling to solution vector"""
        if self.scaler and self.scaling_info:
            return self.scaler.unscale(x, self.scaling_info)
        return x
    
    def _preprocess_vector(self, b: Any) -> Any:
        """Preprocess vector to appropriate format (to be implemented by subclasses)"""
        return b
    
    def _to_numpy_matrix(self, A: Any) -> Any:
        """Convert matrix to NumPy format (to be implemented by subclasses)"""
        return A
    
    def get_preconditioner(self) -> Optional[Any]:
        """Get preconditioner instance for inspection"""
        return self.preconditioner
    
    def get_available_solvers(self) -> List[str]:
        """Get list of available solver methods"""
        return list(self.solvers.keys())