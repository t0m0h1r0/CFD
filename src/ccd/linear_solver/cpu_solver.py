"""
CPU (SciPy) を使用した線形方程式系ソルバー

This module provides solvers for linear systems Ax = b using SciPy's
sparse linear algebra solvers. It supports direct and iterative methods
with various convergence options and scaling techniques.
"""

import os
import time
import numpy as np
import scipy.sparse.linalg as splinalg
from .base import LinearSolver


class CPULinearSolver(LinearSolver):
    """CPU (SciPy) を使用した線形方程式系ソルバー
    
    This solver uses SciPy's sparse linear algebra functionality to solve
    linear systems on the CPU. It supports direct solvers and iterative 
    methods like GMRES, CG, BiCGSTAB, etc., along with various scaling 
    techniques to improve numerical stability.
    """
    
    def _initialize(self):
        """Initialize CPU solver resources and configuration"""
        # Convert matrix to NumPy/SciPy format
        self.A = self._ensure_scipy_matrix(self.original_A)
        
        # Setup scaling if requested
        if self.scaling_method:
            from scaling import plugin_manager
            self.scaler = plugin_manager.get_plugin(self.scaling_method)
            self._prepare_scaling()
        
        # Define available solvers - direct mapping to specialized methods
        self.solvers = {
            "direct": self._solve_direct,
            "gmres": self._solve_gmres,
            "cg": self._solve_iterative,
            "cgs": self._solve_iterative,
            "bicgstab": self._solve_iterative,
            "minres": self._solve_iterative,
            "lsqr": self._solve_lsqr,
            "lsmr": self._solve_lsmr
        }
    
    def _ensure_scipy_matrix(self, A):
        """Ensure matrix is in SciPy format
        
        Args:
            A: Input matrix in any supported format
            
        Returns:
            SciPy sparse matrix or ndarray
        """
        # Handle CuPy arrays
        if hasattr(A, 'get'):
            A = A.get()
        
        # Handle JAX arrays
        if 'jax' in str(type(A)):
            A = np.array(A)
            
        # Handle case where A is already in correct format
        return A
    
    def _ensure_numpy_vector(self, b):
        """Ensure vector is in NumPy format
        
        Args:
            b: Input vector in any supported format
            
        Returns:
            NumPy ndarray
        """
        # Handle CuPy arrays
        if hasattr(b, 'get'):
            return b.get()
        
        # Handle JAX arrays
        if 'jax' in str(type(b)):
            return np.array(b)
            
        # Already NumPy array
        return b
    
    def _prepare_scaling(self):
        """Initialize scaling for the linear system"""
        if not self.scaler:
            return
            
        # Create dummy vector for getting scaling information
        dummy_b = np.ones(self.A.shape[0])
        
        try:
            # Calculate and store scaling information
            _, _, self.scaling_info = self.scaler.scale(self.A, dummy_b)
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
            self.scaling_info = None
    
    def solve(self, b, method="direct", options=None):
        """Solve linear system Ax = b
        
        Args:
            b: Right-hand side vector
            method: Solution method ('direct', 'gmres', 'cg', etc.)
            options: Solver-specific options
            
        Returns:
            Solution vector x
        """
        start_time = time.time()
        options = options or {}
        
        # Convert right-hand side to NumPy
        b_np = self._ensure_numpy_vector(b)
        
        # Apply scaling if requested
        b_scaled = b_np
        if self.scaler and self.scaling_info:
            try:
                b_scaled = self.scaler.scale_b_only(b_np, self.scaling_info)
            except Exception as e:
                print(f"スケーリングエラー: {e}")
        
        # Choose solver method
        if method not in self.solvers:
            print(f"未対応の解法: {method}、directに切り替えます")
            method = "direct"
        
        # Solve the system
        try:
            solver_func = self.solvers[method]
            x, iterations = solver_func(self.A, b_scaled, options)
            
            # Apply unscaling if needed
            if self.scaler and self.scaling_info:
                x = self.scaler.unscale(x, self.scaling_info)
                
            # Record solver statistics
            self.last_iterations = iterations
            elapsed = time.time() - start_time
            print(f"CPU解法: {method}, 時間: {elapsed:.4f}秒" + 
                  (f", 反復: {iterations}" if iterations else ""))
                  
            return x
            
        except Exception as e:
            print(f"CPU解法エラー: {e}")
            # Fall back to direct solve on failure
            x = splinalg.spsolve(self.A, b_scaled)
            if self.scaler and self.scaling_info:
                x = self.scaler.unscale(x, self.scaling_info)
            return x
    
    def _solve_direct(self, A, b, options=None):
        """Direct solver using factorization
        
        Args:
            A: System matrix
            b: Right-hand side vector
            options: Solver options
            
        Returns:
            tuple: (solution vector, None)
        """
        return splinalg.spsolve(A, b), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES iterative solver
        
        Args:
            A: System matrix
            b: Right-hand side vector
            options: Solver options
            
        Returns:
            tuple: (solution vector, iteration count)
        """
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 10000)
        
        # Initial guess
        x0 = options.get("x0", np.ones_like(b))
        
        # Setup convergence monitoring
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
                residuals.append(float(residual))
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # Run GMRES
        result = splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                              restart=restart, callback=callback)
        
        # Visualize convergence history if requested
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, "gmres", options)
        
        return result[0], result[1]  # solution, iterations
    
    def _solve_iterative(self, A, b, options=None):
        """Generic iterative solver interface
        
        Args:
            A: System matrix
            b: Right-hand side vector
            options: Solver options
            
        Returns:
            tuple: (solution vector, iteration count)
        """
        options = options or {}
        method_name = options.get("method_name", "cg")
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # Initial guess
        x0 = options.get("x0", np.ones_like(b))
        
        # Setup convergence monitoring
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
                residuals.append(float(residual))
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # Get the appropriate solver function
        solver_func = getattr(splinalg, method_name)
        
        # Run the solver
        result = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        
        # Visualize convergence history if requested
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, method_name, options)
        
        return result[0], result[1]  # solution, iterations
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            options: ソルバーオプション
                - damp: 正則化パラメータ
                - atol, btol: 収束許容誤差
                - conlim: 条件数の制限
                - iter_lim, maxiter: 最大反復回数
                - show: 反復過程を表示するかどうか
        
        Returns:
            tuple: (解ベクトル, 反復回数)
        """
        options = options or {}
        
        # LSQRパラメータを抽出
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        maxiter = options.get("maxiter", options.get("iter_lim", None))
        show = options.get("show", False)
        
        # scipy.sparse.linalg.lsqrを実行
        result = splinalg.lsqr(A, b, damp=damp, atol=atol, btol=btol,
                          conlim=conlim, iter_lim=maxiter, show=show)
        
        # 結果を解析
        x = result[0]         # 解ベクトル
        istop = result[1]     # 終了理由
        itn = result[2]       # 反復回数
        
        return x, itn

    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            options: ソルバーオプション
                - x0: 初期解
                - damp: 正則化パラメータ
                - atol, btol: 収束許容誤差
                - conlim: 条件数の制限
                - maxiter: 最大反復回数
                - show: 反復過程を表示するかどうか
        
        Returns:
            tuple: (解ベクトル, 反復回数)
        """
        options = options or {}
        
        # LSMRパラメータを抽出
        x0 = options.get("x0", None)
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        maxiter = options.get("maxiter", None)
        show = options.get("show", False)
        
        # scipy.sparse.linalg.lsmrを実行
        result = splinalg.lsmr(A, b, damp=damp, atol=atol, btol=btol,
                          conlim=conlim, maxiter=maxiter, show=show, x0=x0)
        
        # 結果を解析
        x = result[0]         # 解ベクトル
        istop = result[1]     # 終了理由
        itn = result[2]       # 反復回数
        
        return x, itn
    
    def _visualize_convergence(self, residuals, method_name, options):
        """Visualize convergence history
        
        Args:
            residuals: List of residual values
            method_name: Name of the solver method
            options: Visualization options
        """
        output_dir = options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals)+1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{method_name.upper()} ソルバーの収束履歴')
        
        prefix = options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_{method_name}.png")
        plt.savefig(filename, dpi=150)
        plt.close()