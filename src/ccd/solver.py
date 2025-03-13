from abc import ABC, abstractmethod
import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg

class BaseCCDSolver(ABC):
    """Abstract base class for Combined Compact Difference solvers"""

    def __init__(self, system, grid):
        """
        Initialize solver
        
        Args:
            system: Equation system
            grid: Grid object
        """
        self.system = system
        self.grid = grid
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.last_iterations = None
        self.sparsity_info = None

    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        Set solver method and options
        
        Args:
            method: Solution method ("direct", "gmres", "cg", "cgs")
            options: Solver options dictionary
            scaling_method: Scaling method name (None for no scaling)
        """
        valid_methods = ["direct", "gmres", "cg", "cgs"]
        if method not in valid_methods:
            method = "direct"
            
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling_method

    def _create_preconditioner(self, A):
        """
        Create preconditioner
        
        Args:
            A: System matrix
            
        Returns:
            Preconditioner operator
        """
        if not self.solver_options.get("use_preconditioner", True):
            return None
        
        # Jacobi preconditioner (inverse of diagonal elements)
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv

    def _solve_gmres(self, A, b):
        """
        Solve system using GMRES
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        restart = self.solver_options.get("restart", 100)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        try:
            x, info = splinalg.gmres(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                restart=restart
            )
            
            # Save iteration count if available
            if hasattr(info, 'iterations'):
                self.last_iterations = info.iterations
            elif isinstance(info, tuple) and len(info) > 0:
                self.last_iterations = info[0]
            else:
                self.last_iterations = maxiter if info != 0 else None
            
            if info == 0:
                return x
        except Exception as e:
            print(f"GMRESソルバーでエラーが発生しました: {e}")
            
        print("直接解法にフォールバックします")
        return splinalg.spsolve(A, b)

    def _solve_cg(self, A, b):
        """
        Solve system using CG
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        try:
            x, info = splinalg.cg(A, b, tol=tol, maxiter=maxiter, M=precond)
            
            # Save iteration count if available
            if hasattr(info, 'iterations'):
                self.last_iterations = info.iterations
            else:
                self.last_iterations = maxiter if info != 0 else None
                
            if info == 0:
                return x
        except Exception as e:
            print(f"CG解法でエラーが発生しました: {e}")
            
        print("直接解法にフォールバックします")
        return splinalg.spsolve(A, b)
        
    def _solve_cgs(self, A, b):
        """
        Solve system using CGS
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        try:
            x, info = splinalg.cgs(A, b, tol=tol, maxiter=maxiter, M=precond)
            
            # Save iteration count if available
            if hasattr(info, 'iterations'):
                self.last_iterations = info.iterations
            else:
                self.last_iterations = maxiter if info != 0 else None
                
            if info == 0:
                return x
        except Exception as e:
            print(f"CGS解法でエラーが発生しました: {e}")
            
        print("直接解法にフォールバックします")
        return splinalg.spsolve(A, b)

    def analyze_system(self):
        """
        Analyze matrix system sparsity
        
        Returns:
            Sparsity information dictionary
        """
        sparsity_info = self.system.analyze_sparsity()
        self.sparsity_info = sparsity_info
        
        print("\n行列構造分析:")
        print(f"  行列サイズ: {sparsity_info['matrix_size']} x {sparsity_info['matrix_size']}")
        print(f"  非ゼロ要素: {sparsity_info['non_zeros']}")
        print(f"  疎性率: {sparsity_info['sparsity']:.6f}")
        print(f"  メモリ使用量(密行列): {sparsity_info['memory_dense_MB']:.2f} MB")
        print(f"  メモリ使用量(疎行列): {sparsity_info['memory_sparse_MB']:.2f} MB")
        print(f"  メモリ削減率: {sparsity_info['memory_dense_MB'] / sparsity_info['memory_sparse_MB']:.1f}倍")
        
        return sparsity_info

    @abstractmethod
    def solve(self, analyze_before_solve=True):
        """
        Solve the system
        
        Args:
            analyze_before_solve: Whether to analyze the matrix before solving
            
        Returns:
            Solution components
        """
        pass


class CCDSolver1D(BaseCCDSolver):
    """1D Combined Compact Difference solver"""

    def __init__(self, system, grid):
        """
        Initialize 1D solver
        
        Args:
            system: Equation system
            grid: 1D Grid object
        """
        super().__init__(system, grid)
        if grid.is_2d:
            raise ValueError("1Dソルバーは2Dグリッドでは使用できません")

    def solve(self, analyze_before_solve=True, f_values=None, 
              left_dirichlet=None, right_dirichlet=None,
              left_neumann=None, right_neumann=None,
              enable_dirichlet=True, enable_neumann=True):
        """
        Solve the 1D system
        
        Args:
            analyze_before_solve: Whether to analyze the matrix before solving
            f_values: Right-hand side values for governing equation (shape: (n,))
            left_dirichlet: Left boundary Dirichlet value
            right_dirichlet: Right boundary Dirichlet value
            left_neumann: Left boundary Neumann value
            right_neumann: Right boundary Neumann value
            enable_dirichlet: Whether to enable Dirichlet boundary conditions
            enable_neumann: Whether to enable Neumann boundary conditions
            
        Returns:
            (psi, psi_prime, psi_second, psi_third) tuple
        """
        A, b = self.system.build_matrix_system()

        if analyze_before_solve:
            self.analyze_system()
            
        # Update right-hand side values if provided
        n = self.grid.n_points
        
        # Set governing equation values (f values)
        if f_values is not None:
            for i in range(n):
                b[i * 4] = f_values[i] if i < len(f_values) else 0.0
        
        # Set boundary condition values based on enabled flags
        if enable_dirichlet:
            if left_dirichlet is not None:
                b[1] = left_dirichlet  # Left boundary Dirichlet
            if right_dirichlet is not None:
                b[(n-1) * 4 + 1] = right_dirichlet  # Right boundary Dirichlet
        
        if enable_neumann:
            if left_neumann is not None:
                b[2] = left_neumann  # Left boundary Neumann
            if right_neumann is not None:
                b[(n-1) * 4 + 2] = right_neumann  # Right boundary Neumann
            
        # Apply scaling if requested
        scaling_info = None
        scaler = None
        if self.scaling_method is not None:
            from scaling import plugin_manager
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                A, b, scaling_info = scaler.scale(A, b)

        # Solve system
        if self.solver_method == "gmres":
            sol = self._solve_gmres(A, b)
        elif self.solver_method == "cg":
            sol = self._solve_cg(A, b)
        elif self.solver_method == "cgs":
            sol = self._solve_cgs(A, b)
        else:
            sol = splinalg.spsolve(A, b)
            
        # Unscale solution if scaling was applied
        if scaling_info is not None and scaler is not None:
            sol = scaler.unscale(sol, scaling_info)

        # Process 1D solution
        return self._process_solution(sol)

    def _process_solution(self, sol):
        """Process 1D solution vector"""
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third


class CCDSolver2D(BaseCCDSolver):
    """2D Combined Compact Difference solver"""

    def __init__(self, system, grid):
        """
        Initialize 2D solver
        
        Args:
            system: Equation system
            grid: 2D Grid object
        """
        super().__init__(system, grid)
        if not grid.is_2d:
            raise ValueError("2Dソルバーは1Dグリッドでは使用できません")

    def solve(self, analyze_before_solve=True, f_values=None,
              left_dirichlet=None, right_dirichlet=None, 
              bottom_dirichlet=None, top_dirichlet=None,
              left_neumann=None, right_neumann=None,
              bottom_neumann=None, top_neumann=None,
              enable_dirichlet=True, enable_neumann=True):
        """
        Solve the 2D system
        
        Args:
            analyze_before_solve: Whether to analyze the matrix before solving
            f_values: Right-hand side values for governing equation (shape: (nx, ny))
            left_dirichlet: Left boundary Dirichlet values (shape: (ny,) or scalar)
            right_dirichlet: Right boundary Dirichlet values (shape: (ny,) or scalar)
            bottom_dirichlet: Bottom boundary Dirichlet values (shape: (nx,) or scalar)
            top_dirichlet: Top boundary Dirichlet values (shape: (nx,) or scalar)
            left_neumann: Left boundary Neumann values (shape: (ny,) or scalar)
            right_neumann: Right boundary Neumann values (shape: (ny,) or scalar)
            bottom_neumann: Bottom boundary Neumann values (shape: (nx,) or scalar)
            top_neumann: Top boundary Neumann values (shape: (nx,) or scalar)
            enable_dirichlet: Whether to enable Dirichlet boundary conditions
            enable_neumann: Whether to enable Neumann boundary conditions
            
        Returns:
            (psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy) tuple
        """
        A, b = self.system.build_matrix_system()

        if analyze_before_solve:
            self.analyze_system()
            
        # Update right-hand side values if provided
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # Set governing equation values (f values)
        if f_values is not None:
            for j in range(ny):
                for i in range(nx):
                    idx = (j * nx + i) * n_unknowns
                    if isinstance(f_values, (list, cp.ndarray)) and len(f_values) > i and len(f_values[i]) > j:
                        b[idx] = f_values[i][j]
                    elif isinstance(f_values, (int, float)):
                        b[idx] = f_values
        
        # Set boundary condition values based on enabled flags
        if enable_dirichlet:
            # Dirichlet boundary conditions
            # Left boundary (i=0)
            if left_dirichlet is not None:
                for j in range(ny):
                    idx = (j * nx + 0) * n_unknowns + 1  # ψ_x
                    if isinstance(left_dirichlet, (list, cp.ndarray)) and len(left_dirichlet) > j:
                        b[idx] = left_dirichlet[j]
                    elif isinstance(left_dirichlet, (int, float)):
                        b[idx] = left_dirichlet
            
            # Right boundary (i=nx-1)
            if right_dirichlet is not None:
                for j in range(ny):
                    idx = (j * nx + (nx-1)) * n_unknowns + 1  # ψ_x
                    if isinstance(right_dirichlet, (list, cp.ndarray)) and len(right_dirichlet) > j:
                        b[idx] = right_dirichlet[j]
                    elif isinstance(right_dirichlet, (int, float)):
                        b[idx] = right_dirichlet
            
            # Bottom boundary (j=0)
            if bottom_dirichlet is not None:
                for i in range(nx):
                    idx = (0 * nx + i) * n_unknowns + 4  # ψ_y
                    if isinstance(bottom_dirichlet, (list, cp.ndarray)) and len(bottom_dirichlet) > i:
                        b[idx] = bottom_dirichlet[i]
                    elif isinstance(bottom_dirichlet, (int, float)):
                        b[idx] = bottom_dirichlet
            
            # Top boundary (j=ny-1)
            if top_dirichlet is not None:
                for i in range(nx):
                    idx = ((ny-1) * nx + i) * n_unknowns + 4  # ψ_y
                    if isinstance(top_dirichlet, (list, cp.ndarray)) and len(top_dirichlet) > i:
                        b[idx] = top_dirichlet[i]
                    elif isinstance(top_dirichlet, (int, float)):
                        b[idx] = top_dirichlet
        
        if enable_neumann:
            # Neumann boundary conditions
            # Left boundary (i=0)
            if left_neumann is not None:
                for j in range(ny):
                    idx = (j * nx + 0) * n_unknowns + 2  # ψ_xx
                    if isinstance(left_neumann, (list, cp.ndarray)) and len(left_neumann) > j:
                        b[idx] = left_neumann[j]
                    elif isinstance(left_neumann, (int, float)):
                        b[idx] = left_neumann
            
            # Right boundary (i=nx-1)
            if right_neumann is not None:
                for j in range(ny):
                    idx = (j * nx + (nx-1)) * n_unknowns + 2  # ψ_xx
                    if isinstance(right_neumann, (list, cp.ndarray)) and len(right_neumann) > j:
                        b[idx] = right_neumann[j]
                    elif isinstance(right_neumann, (int, float)):
                        b[idx] = right_neumann
            
            # Bottom boundary (j=0)
            if bottom_neumann is not None:
                for i in range(nx):
                    idx = (0 * nx + i) * n_unknowns + 5  # ψ_yy
                    if isinstance(bottom_neumann, (list, cp.ndarray)) and len(bottom_neumann) > i:
                        b[idx] = bottom_neumann[i]
                    elif isinstance(bottom_neumann, (int, float)):
                        b[idx] = bottom_neumann
            
            # Top boundary (j=ny-1)
            if top_neumann is not None:
                for i in range(nx):
                    idx = ((ny-1) * nx + i) * n_unknowns + 5  # ψ_yy
                    if isinstance(top_neumann, (list, cp.ndarray)) and len(top_neumann) > i:
                        b[idx] = top_neumann[i]
                    elif isinstance(top_neumann, (int, float)):
                        b[idx] = top_neumann
                
        # Apply scaling if requested
        scaling_info = None
        scaler = None
        if self.scaling_method is not None:
            from scaling import plugin_manager
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                A, b, scaling_info = scaler.scale(A, b)

        # Solve system
        if self.solver_method == "gmres":
            sol = self._solve_gmres(A, b)
        elif self.solver_method == "cg":
            sol = self._solve_cg(A, b)
        elif self.solver_method == "cgs":
            sol = self._solve_cgs(A, b)
        else:
            sol = splinalg.spsolve(A, b)
            
        # Unscale solution if scaling was applied
        if scaling_info is not None and scaler is not None:
            sol = scaler.unscale(sol, scaling_info)

        # Process 2D solution
        return self._process_solution(sol)

    def _process_solution(self, sol):
        """Process 2D solution vector"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # Initialize solution arrays
        psi = cp.zeros((nx, ny))
        psi_x = cp.zeros((nx, ny))
        psi_xx = cp.zeros((nx, ny))
        psi_xxx = cp.zeros((nx, ny))
        psi_y = cp.zeros((nx, ny))
        psi_yy = cp.zeros((nx, ny))
        psi_yyy = cp.zeros((nx, ny))
        
        # Extract values for each grid point
        for j in range(ny):
            for i in range(nx):
                idx = (j * nx + i) * n_unknowns
                psi[i, j] = sol[idx]
                psi_x[i, j] = sol[idx + 1]
                psi_xx[i, j] = sol[idx + 2]
                psi_xxx[i, j] = sol[idx + 3]
                psi_y[i, j] = sol[idx + 4]
                psi_yy[i, j] = sol[idx + 5]
                psi_yyy[i, j] = sol[idx + 6]
        
        return psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy