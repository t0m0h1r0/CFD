import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg

class CCD2DSolver:
    """Solver for 2D CCD method"""
    
    def __init__(self, system, grid):
        """
        Initialize with equation system and grid
        
        Args:
            system: EquationSystem2D object
            grid: Grid2D object
        """
        self.system = system
        self.grid = grid
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.sparsity_info = None
        self.last_iterations = None
    
    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        Set the solver method and options
        
        Args:
            method: One of "direct", "gmres", "cg", "cgs"
            options: Solver options dictionary
            scaling_method: Name of scaling method to use (None for no scaling)
        """
        valid_methods = ["direct", "gmres", "cg", "cgs"]
        if method not in valid_methods:
            method = "direct"
        
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling_method
    
    def _create_preconditioner(self, A):
        """
        Create a preconditioner for iterative solvers
        
        Args:
            A: System matrix
            
        Returns:
            Preconditioner operator
        """
        if not self.solver_options.get("use_preconditioner", True):
            return None
        
        # Simple Jacobi preconditioner
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv
    
    def _solve_gmres(self, A, b):
        """
        Solve using GMRES method
        
        Args:
            A: System matrix
            b: Right-hand side
            
        Returns:
            Solution vector
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        restart = self.solver_options.get("restart", 100)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        # 初期解にonesを使用
        x0 = cp.ones_like(b)
        
        try:
            x, info = splinalg.gmres(
                A, b, 
                x0=x0,  # 初期解を指定
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                restart=restart,
            )
            
            # 反復回数が利用可能な場合は保存
            if hasattr(info, 'iterations'):
                self.last_iterations = info.iterations
            elif isinstance(info, tuple) and len(info) > 0:
                self.last_iterations = info[0]
            else:
                self.last_iterations = maxiter if info != 0 else None
            
            if info == 0:
                return x
        except Exception as e:
            print(f"GMRES failed: {e}")
        
        print("Falling back to direct solver")
        return splinalg.spsolve(A, b)
    
    def _solve_cg(self, A, b):
        """
        Solve using Conjugate Gradient method
        
        Args:
            A: System matrix
            b: Right-hand side
            
        Returns:
            Solution vector
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        # 初期解にonesを使用
        x0 = cp.ones_like(b)
        
        try:
            x, info = splinalg.cg(
                A, b, 
                x0=x0,  # 初期解を指定
                tol=tol, 
                maxiter=maxiter, 
                M=precond
            )
            
            # 反復回数が利用可能な場合は保存
            if hasattr(info, 'iterations'):
                self.last_iterations = info.iterations
            else:
                self.last_iterations = maxiter if info != 0 else None
            
            if info == 0:
                return x
        except Exception as e:
            print(f"CG failed: {e}")
        
        print("Falling back to direct solver")
        return splinalg.spsolve(A, b)
    
    def _solve_cgs(self, A, b):
        """
        Solve using Conjugate Gradient Squared method
        
        Args:
            A: System matrix
            b: Right-hand side
            
        Returns:
            Solution vector
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        # 初期解にonesを使用
        x0 = cp.ones_like(b)
        
        try:
            x, info = splinalg.cgs(
                A, b, 
                x0=x0,  # 初期解を指定
                tol=tol, 
                maxiter=maxiter, 
                M=precond
            )
            
            # 反復回数が利用可能な場合は保存
            if hasattr(info, 'iterations'):
                self.last_iterations = info.iterations
            else:
                self.last_iterations = maxiter if info != 0 else None
            
            if info == 0:
                return x
        except Exception as e:
            print(f"CGS failed: {e}")
        
        print("Falling back to direct solver")
        return splinalg.spsolve(A, b)
    
    def solve(self, analyze_before_solve=False):
        """
        Solve the 2D CCD system
        
        Args:
            analyze_before_solve: Whether to analyze sparsity before solving
            
        Returns:
            Tuple of solution arrays:
            (psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy)
        """
        if analyze_before_solve:
            self.analyze_system()
        
        # Build the matrix system
        A, b = self.system.build_matrix_system()
        
        # スケーリングを適用（要求された場合）
        scaling_info = None
        scaler = None
        if self.scaling_method is not None:
            from scaling import plugin_manager
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                A, b, scaling_info = scaler.scale(A, b)
        
        # Solve based on selected method
        if self.solver_method == "gmres":
            sol = self._solve_gmres(A, b)
        elif self.solver_method == "cg":
            sol = self._solve_cg(A, b)
        elif self.solver_method == "cgs":
            sol = self._solve_cgs(A, b)
        else:
            sol = splinalg.spsolve(A, b)
        
        # スケーリングが適用された場合は解をアンスケール
        if scaling_info is not None and scaler is not None:
            sol = scaler.unscale(sol, scaling_info)
        
        # Extract solution components
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
    
    def analyze_system(self):
        """
        Analyze the sparsity of the matrix system
        
        Returns:
            Sparsity information dictionary
        """
        sparsity_info = self.system.analyze_sparsity()
        self.sparsity_info = sparsity_info
        
        print("\nMatrix Structure Analysis:")
        print(f"  Matrix size: {sparsity_info['matrix_size']} x {sparsity_info['matrix_size']}")
        print(f"  Non-zeros: {sparsity_info['non_zeros']}")
        print(f"  Sparsity: {sparsity_info['sparsity']:.6f}")
        print(f"  Memory (dense): {sparsity_info['memory_dense_MB']:.2f} MB")
        print(f"  Memory (sparse): {sparsity_info['memory_sparse_MB']:.2f} MB")
        print(f"  Memory reduction: {sparsity_info['memory_dense_MB'] / sparsity_info['memory_sparse_MB']:.1f}x")
        
        return sparsity_info