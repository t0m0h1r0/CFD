"""
Base Solver for High-Precision Compact Difference (CCD) Method 

This module provides the base solver class for solving partial differential 
equations using the Compact Difference method, handling common functionality 
and dimension-independent processing.
"""

from abc import ABC, abstractmethod

# Import solver creation function from linear_solver package
from linear_solver import create_solver


class BaseCCDSolver(ABC):
    """Abstract base class for Compact Difference method solvers"""

    def __init__(self, equation_set, grid, backend="cpu"):
        """
        Initialize solver

        Args:
            equation_set: Equation set to use
            grid: Grid object
            backend: Computational backend ('cpu', 'cuda', 'jax')
        """
        self.equation_set = equation_set
        self.grid = grid
        self.backend = backend
        
        # Initialize system based on dimension and build matrix A (CPU processing)
        self.system = self._create_equation_system(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()
        
        # Set default solver and RHS builder
        self._create_rhs_builder()
        
        # Default parameters
        self.method = "direct"
        self.scaling_method = None
        self.preconditioner = None
        
        # Initialize LinearSolver
        self._create_linear_solver()
    
    def _create_linear_solver(self):
        """
        Create or recreate the linear solver with current settings
        """
        self.linear_solver = create_solver(
            self.matrix_A,
            enable_dirichlet=self.enable_dirichlet,
            enable_neumann=self.enable_neumann,
            scaling_method=self.scaling_method,
            preconditioner=self.preconditioner,
            backend=self.backend
        )
    
    @abstractmethod
    def _create_equation_system(self, grid):
        """
        Create equation system based on dimension
        
        Args:
            grid: Grid object
            
        Returns:
            Equation system object
        """
        pass
    
    @abstractmethod
    def _create_rhs_builder(self):
        """Create RHS builder based on dimension"""
        pass
    
    def set_solver(self, method="direct", options=None, scaling_method=None, preconditioner=None):
        """
        Set solver parameters
        
        Args:
            method: Solution method
            options: Solver options dictionary
            scaling_method: Scaling method name
            preconditioner: Preconditioner name or object
        """
        # Initialize options
        options = options or {}
        
        # Get backend specification
        backend = options.get("backend", self.backend)
        
        # Extract options to pass to linear solver
        solver_options = {}
        for key, value in options.items():
            # Extract only options used by LinearSolver
            if key in ["tol", "maxiter", "restart", "inner_m", "outer_k", "m", "k", "x0"]:
                solver_options[key] = value
        
        # Update class-level attributes
        self.method = method
        self.backend = backend
        
        # Handle scaling method
        if scaling_method is not None:
            self.scaling_method = scaling_method
        
        # Handle preconditioner
        if preconditioner is not None:
            self.preconditioner = preconditioner
        
        # Recreate linear solver with updated parameters
        self._create_linear_solver()
        
        # Set solver options
        self.linear_solver.solver_method = method
        self.linear_solver.solver_options = solver_options
        
        # Print x0 if included
        if "x0" in solver_options:
            print(f"x0 has been set (shape: {solver_options['x0'].shape})")
    
    @property
    def current_scaling_method(self):
        """Get current scaling method"""
        return self.scaling_method
    
    @current_scaling_method.setter
    def current_scaling_method(self, value):
        """Set scaling method"""
        # Recreate solver with new scaling method
        self.set_solver(
            method=self.method, 
            scaling_method=value, 
            preconditioner=self.preconditioner,
            options={"backend": self.backend}
        )
    
    @property
    def last_iterations(self):
        """Get last iteration count"""
        return self.linear_solver.last_iterations if hasattr(self.linear_solver, 'last_iterations') else None
    
    def get_boundary_settings(self):
        """Get boundary condition settings"""
        return self.enable_dirichlet, self.enable_neumann

    def analyze_system(self):
        """
        Analyze matrix system sparsity
        
        Returns:
            Sparsity information dictionary
        """
        A = self.matrix_A
        total_size = A.shape[0]
        nnz = A.nnz
        sparsity = 1.0 - (nnz / (total_size * total_size))
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8 bytes/double-precision float
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 8 bytes for value + 4 bytes for indices
        
        print("\nMatrix Structure Analysis:")
        print(f"  Matrix Size: {total_size} x {total_size}")
        print(f"  Non-zero Elements: {nnz}")
        print(f"  Sparsity Rate: {sparsity:.6f}")
        print(f"  Memory Usage (Dense Matrix): {memory_dense_MB:.2f} MB")
        print(f"  Memory Usage (Sparse Matrix): {memory_sparse_MB:.2f} MB")
        
        return {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }

    def solve(self, analyze_before_solve=True, f_values=None, **boundary_values):
        """
        Solve the system
        
        Args:
            analyze_before_solve: Whether to analyze matrix before solving
            f_values: Right-hand side values 
            **boundary_values: Boundary value dictionary (dimension-dependent)
            
        Returns:
            Solution components
        """
        # Analyze matrix if requested
        if analyze_before_solve:
            self.analyze_system()
            
        # Build right-hand side vector b
        b = self.rhs_builder.build_rhs_vector(f_values, **boundary_values)
        
        # Check current options (debug info)
        if hasattr(self.linear_solver, 'solver_options'):
            print(f"Solver options in solve method: {self.linear_solver.solver_options}")
        
        # Solve linear system using preset method and options
        sol = self.linear_solver.solve(b)

        # Extract solution components (dimension-dependent)
        return self._extract_solution(sol)
    
    def solve_with_options(self, analyze_before_solve=True, f_values=None, solve_options=None, **boundary_values):
        """
        Solve system with custom options
        
        Args:
            analyze_before_solve: Whether to analyze matrix before solving
            f_values: Right-hand side values
            solve_options: Direct solver options
            **boundary_values: Boundary value dictionary (dimension-dependent)
            
        Returns:
            Solution components
        """
        # Analyze matrix if requested
        if analyze_before_solve:
            self.analyze_system()
            
        # Build right-hand side vector b
        b = self.rhs_builder.build_rhs_vector(f_values, **boundary_values)
        
        # Check options (debug info)
        if solve_options and "x0" in solve_options:
            print(f"x0 in solve_with_options method: {solve_options['x0'].shape}")
        
        # Solve linear system using specified options and method
        sol = self.linear_solver.solve(b, method=self.method, options=solve_options)

        # Extract solution components (dimension-dependent)
        return self._extract_solution(sol)
    
    @abstractmethod
    def _extract_solution(self, sol):
        """Extract solution components from solution vector (dimension-specific)"""
        pass
    
    def _to_numpy(self, arr):
        """Convert CuPy array to NumPy array (if necessary)"""
        if hasattr(arr, 'get'):
            return arr.get()
        return arr