import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as splinalg

class CCDSolver:
    """CCDソルバークラス"""

    def __init__(self, system, grid):
        self.system = system
        self.grid = grid
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.last_iterations = None

    def set_solver(self, method="direct", options=None, scaling_method=None):
        """使用するソルバーを設定
        
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
        """前処理器を作成"""
        if not self.solver_options.get("use_preconditioner", True):
            return None
            
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        D_inv = sp.diags(1.0 / diag)
        
        return splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: D_inv @ x
        )

    def _solve_gmres(self, A, b):
        """GMRESを使用して行列システムを解く"""
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
            
        return splinalg.spsolve(A, b)

    def _solve_cg(self, A, b):
        """CGを使用して行列システムを解く"""
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
            
        return splinalg.spsolve(A, b)
        
    def _solve_cgs(self, A, b):
        """CGSを使用して行列システムを解く"""
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
            
        return splinalg.spsolve(A, b)

    def solve(self, analyze_before_solve=True):
        """方程式を解く"""
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

        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third

    def analyze_system(self):
        """行列システムの疎性を分析"""
        sparsity_info = self.system.analyze_sparsity()
        self.sparsity_info = sparsity_info
        return sparsity_info