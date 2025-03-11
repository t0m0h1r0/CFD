import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg
from grid2d import Grid2D

class CCDSolver:
    """1D/2D 両対応のCCDソルバークラス"""

    def __init__(self, system, grid):
        """
        ソルバーを初期化
        
        Args:
            system: 方程式システム
            grid: Grid (1D) or Grid2D (2D) オブジェクト
        """
        self.system = system
        self.grid = grid
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.last_iterations = None
        self.sparsity_info = None
        
        # 1D or 2D mode判定
        self.is_2d = isinstance(grid, Grid2D)

    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        使用するソルバーを設定
        
        Args:
            method: 解法 ("direct", "gmres", "cg", "cgs")
            options: ソルバーオプション辞書
            scaling_method: 使用するスケーリング手法の名前（Noneの場合はスケーリングなし）
        """
        valid_methods = ["direct", "gmres", "cg", "cgs"]
        if method not in valid_methods:
            method = "direct"
            
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling_method

    def _create_preconditioner(self, A):
        """
        前処理器を作成
        
        Args:
            A: システム行列
            
        Returns:
            Preconditioner operator
        """
        if not self.solver_options.get("use_preconditioner", True):
            return None
        
        # Jacobi前処理（対角要素の逆数）
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv

    def _solve_gmres(self, A, b):
        """
        GMRESを使用して行列システムを解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            解ベクトル
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
            print(f"GMRES解法でエラーが発生しました: {e}")
            
        print("直接解法にフォールバックします")
        return splinalg.spsolve(A, b)

    def _solve_cg(self, A, b):
        """
        CGを使用して行列システムを解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            解ベクトル
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        try:
            x, info = splinalg.cg(A, b, tol=tol, maxiter=maxiter, M=precond)
            
            # 反復回数が利用可能な場合は保存
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
        CGSを使用して行列システムを解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            解ベクトル
        """
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        try:
            x, info = splinalg.cgs(A, b, tol=tol, maxiter=maxiter, M=precond)
            
            # 反復回数が利用可能な場合は保存
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

    def solve(self, analyze_before_solve=True):
        """
        方程式を解く
        
        Args:
            analyze_before_solve: 解く前に行列を分析するかどうか
            
        Returns:
            1D: (psi, psi_prime, psi_second, psi_third) のタプル
            2D: (psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy) のタプル
        """
        A, b = self.system.build_matrix_system()

        if analyze_before_solve:
            self.analyze_system()
            
        # スケーリングを適用（要求された場合）
        scaling_info = None
        scaler = None
        if self.scaling_method is not None:
            from scaling import plugin_manager
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                A, b, scaling_info = scaler.scale(A, b)

        # システムを解く
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

        # 1D または 2D モードに基づいて結果を返す
        if self.is_2d:
            return self._process_2d_solution(sol)
        else:
            return self._process_1d_solution(sol)

    def _process_1d_solution(self, sol):
        """1D解ベクトルを処理"""
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third

    def _process_2d_solution(self, sol):
        """2D解ベクトルを処理"""
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
        行列システムの疎性を分析
        
        Returns:
            疎性情報の辞書
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