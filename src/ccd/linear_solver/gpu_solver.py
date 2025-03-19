"""
GPU (CuPy) を使用した線形方程式系ソルバー

This module provides solvers for linear systems Ax = b using GPU acceleration
via CuPy. It supports direct solvers and various iterative methods.
"""

import os
import time
import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver


class GPULinearSolver(LinearSolver):
    """GPU (CuPy) を使用した線形方程式系ソルバー
    
    This solver uses GPU acceleration through CuPy to solve linear systems.
    It supports direct solvers and iterative methods like GMRES, CG, etc.
    Fallbacks to CPU implementation when CuPy is not available or when
    GPU operations fail.
    """
    
    def _initialize(self):
        """Initialize GPU solver-specific resources and state"""
        # Try to import CuPy and prepare GPU environment
        try:
            import cupy as cp
            import cupyx.scipy.sparse.linalg as splinalg
            self.cp = cp
            self.splinalg = splinalg
            self.has_cupy = True
            
            # Define solver methods dictionary - direct mapping to specialized methods
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
            
            # Convert matrix to GPU format
            self.A = self._to_cupy_matrix(self.original_A)
            
            # Initialize scaling if requested
            if self.scaling_method:
                from scaling import plugin_manager
                self.scaler = plugin_manager.get_plugin(self.scaling_method)
                self._prepare_scaling()
                
        except ImportError as e:
            print(f"警告: CuPyが利用できません: {e}")
            self.has_cupy = False
            # Fall back to CPU solver
            self.cpu_solver = CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            )
    
    def _to_cupy_matrix(self, A):
        """Convert matrix to CuPy format
        
        Args:
            A: Matrix to convert (NumPy array or SciPy sparse matrix)
            
        Returns:
            CuPy array or sparse matrix
        """
        try:
            # Check if already in CuPy format
            if hasattr(A, 'device') and str(type(A)).find('cupy') >= 0:
                return A
                
            # Handle sparse matrices
            if hasattr(A, 'tocsr'):
                A = A.tocsr()
                
            if hasattr(A, 'data') and hasattr(A, 'indices') and hasattr(A, 'indptr'):
                # Convert CSR format sparse matrix
                return self.cp.sparse.csr_matrix(
                    (self.cp.array(A.data), self.cp.array(A.indices), self.cp.array(A.indptr)),
                    shape=A.shape
                )
            elif hasattr(A, 'toarray'):
                # Convert general sparse matrix
                return self.cp.sparse.csr_matrix(A.toarray())
            else:
                # Convert dense matrix
                return self.cp.sparse.csr_matrix(A)
        except Exception as e:
            print(f"GPU行列変換エラー: {e}")
            return A
    
    def _to_cupy_vector(self, b):
        """Convert vector to CuPy format
        
        Args:
            b: Vector to convert
            
        Returns:
            CuPy array
        """
        try:
            # Check if already in CuPy format
            if hasattr(b, 'device') and str(type(b)).find('cupy') >= 0:
                return b
                
            # Convert NumPy array to CuPy array
            return self.cp.array(b)
        except Exception as e:
            print(f"GPU変換エラー: {e}")
            return b
    
    def _prepare_scaling(self):
        """Prepare scaling information for the linear system"""
        if not self.scaler or not self.has_cupy:
            return
            
        try:
            # Create dummy vector for calculating scaling information
            dummy_b = np.ones(self.A.shape[0])
            
            # Calculate scaling information in NumPy
            A_np = self._to_numpy_matrix(self.A)
            _, _, scale_info_np = self.scaler.scale(A_np, dummy_b)
            
            # Convert scaling information to CuPy
            self.scaling_info = {}
            for key, value in scale_info_np.items():
                if isinstance(value, np.ndarray):
                    self.scaling_info[key] = self.cp.array(value)
                else:
                    self.scaling_info[key] = value
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def solve(self, b, method="direct", options=None):
        """Solve linear system Ax = b using specified method
        
        Args:
            b: Right-hand side vector
            method: Solution method ('direct', 'gmres', 'cg', etc.)
            options: Solver-specific options
            
        Returns:
            Solution vector x
        """
        # Fall back to CPU if CuPy not available
        if not self.has_cupy:
            return self.cpu_solver.solve(b, method, options)
        
        start_time = time.time()
        options = options or {}
        
        try:
            # Convert vector to CuPy
            b_gpu = self._to_cupy_vector(b)
            
            # Apply scaling if configured
            b_scaled = self._apply_scaling_to_b(b_gpu)
            
            # Select and apply solver method
            if method not in self.solvers:
                print(f"未対応の解法: {method}、directに切り替えます")
                method = "direct"
            
            solver_func = self.solvers[method]
            x_gpu, iterations = solver_func(self.A, b_scaled, options)
            
            # Apply unscaling to solution
            x_gpu = self._apply_unscaling_to_x(x_gpu)
                
            # Transfer result to CPU
            x = x_gpu.get()
            
            # Record solution statistics
            self.last_iterations = iterations
            elapsed = time.time() - start_time
            print(f"GPU解法: {method}, 時間: {elapsed:.4f}秒" + 
                  (f", 反復: {iterations}" if iterations else ""))
                  
            return x
                
        except Exception as e:
            print(f"GPU解法エラー: {e}, CPUに切り替えます")
            # Fall back to CPU solver
            return CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            ).solve(b, method, options)
    
    def _apply_scaling_to_b(self, b):
        """Apply scaling to right-hand side vector
        
        Args:
            b: Vector to scale
            
        Returns:
            Scaled vector
        """
        if not self.scaler or not self.scaling_info:
            return b
            
        try:
            # Apply row scaling
            row_scale = self.scaling_info.get('row_scale')
            if row_scale is not None:
                return b * row_scale
                
            # Apply symmetric scaling
            D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
            if D_sqrt_inv is not None:
                return b * D_sqrt_inv
                
            # No scaling applied
            return b
        except Exception as e:
            print(f"スケーリングエラー: {e}")
            return b
    
    def _apply_unscaling_to_x(self, x):
        """Apply unscaling to solution vector
        
        Args:
            x: Solution vector to unscale
            
        Returns:
            Unscaled solution vector
        """
        if not self.scaler or not self.scaling_info:
            return x
            
        try:
            # Apply column scaling
            col_scale = self.scaling_info.get('col_scale')
            if col_scale is not None:
                return x / col_scale
                
            # Apply symmetric scaling
            D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
            if D_sqrt_inv is not None:
                return x * D_sqrt_inv
                
            # No unscaling needed
            return x
        except Exception as e:
            print(f"アンスケーリングエラー: {e}")
            return x
    
    def _to_numpy_matrix(self, A):
        """Convert CuPy matrix to NumPy format
        
        Args:
            A: Matrix to convert
            
        Returns:
            NumPy array or sparse matrix
        """
        if hasattr(A, 'get'):
            return A.get()
        return A
    
    def _to_numpy_vector(self, b):
        """Convert CuPy vector to NumPy format
        
        Args:
            b: Vector to convert
            
        Returns:
            NumPy array
        """
        if hasattr(b, 'get'):
            return b.get()
        return b
    
    def _solve_direct(self, A, b, options=None):
        """Direct solver using LU decomposition
        
        Args:
            A: System matrix
            b: Right-hand side vector
            options: Solver options
            
        Returns:
            tuple: (solution vector, None)
        """
        try:
            x = self.splinalg.spsolve(A, b)
            return x, None
        except Exception as e:
            print(f"GPU直接解法エラー: {e}, CPUにフォールバック")
            # Fall back to CPU direct solver
            import scipy.sparse.linalg as splinalg
            x = splinalg.spsolve(self._to_numpy_matrix(A), self._to_numpy_vector(b))
            return self.cp.array(x), None
    
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
        x0 = options.get("x0", self.cp.ones_like(b))
        
        # Set up convergence monitoring
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = float(self.cp.linalg.norm(b - A @ xk) / self.cp.linalg.norm(b))
                residuals.append(residual)
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # Run GMRES
        result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                      restart=restart, callback=callback)
        
        # Visualize convergence if requested
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
        x0 = options.get("x0", self.cp.ones_like(b))
        
        # Set up convergence monitoring
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = float(self.cp.linalg.norm(b - A @ xk) / self.cp.linalg.norm(b))
                residuals.append(residual)
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # Get the appropriate solver function
        solver_func = getattr(self.splinalg, method_name)
        
        # Run the solver
        result = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        
        # Visualize convergence if requested
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, method_name, options)
        
        return result[0], result[1]  # solution, iterations
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー（GPU版）
        
        Args:
            A: システム行列（CuPy形式）
            b: 右辺ベクトル（CuPy形式）
            options: ソルバーオプション
        
        Returns:
            tuple: (解ベクトル, 反復回数)
        """
        try:
            # CuPyのLSQR実装では多くのパラメータをサポートしていない
            # CuPyのドキュメント：The function solves ``Ax = b``. Given two-dimensional matrix ``A`` is
            # decomposed into ``Q * R``.
            result = self.splinalg.lsqr(A, b)
            
            # CuPyのLSQRは解ベクトルxのみを返す
            return result[0], None
        except Exception as e:
            print(f"GPU LSQR解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            
            # NumPy形式に変換して計算
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            
            # SciPyのLSQRを使用
            options = options or {}
            damp = options.get("damp", 0.0)
            atol = options.get("atol", 1e-6)
            btol = options.get("btol", 1e-6)
            conlim = options.get("conlim", 1e8)
            maxiter = options.get("maxiter", options.get("iter_lim", None))
            
            result = splinalg.lsqr(A_np, b_np, damp=damp, atol=atol, btol=btol,
                                  conlim=conlim, iter_lim=maxiter)
            
            # NumPy解をCuPyに変換して返す
            return self.cp.array(result[0]), result[2]

    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー（GPU版）
        
        Args:
            A: システム行列（CuPy形式）
            b: 右辺ベクトル（CuPy形式）
            options: ソルバーオプション
                - x0: 初期解
                - damp: 正則化パラメータ
                - atol, btol: 収束許容誤差
                - conlim: 条件数の制限
                - maxiter: 最大反復回数
        
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
        
        try:
            # 添付された実装に合わせてパラメータを渡す
            result = self.splinalg.lsmr(A, b, x0=x0, damp=damp, 
                                      atol=atol, btol=btol,
                                      conlim=conlim, maxiter=maxiter)
            
            # 解ベクトルと反復回数を返す
            x, istop, itn = result[0], result[1], result[2]
            return x, itn
        except Exception as e:
            print(f"GPU LSMR解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            
            # NumPy形式に変換して計算
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = None if x0 is None else self._to_numpy_vector(x0)
            
            result = splinalg.lsmr(A_np, b_np, x0=x0_np, damp=damp, 
                                  atol=atol, btol=btol,
                                  conlim=conlim, maxiter=maxiter)
            
            # 解ベクトルと反復回数を返す
            return self.cp.array(result[0]), result[2]
    
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
        plt.title(f'{method_name.upper()} ソルバーの収束履歴 (GPU)')
        
        prefix = options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_gpu_{method_name}.png")
        plt.savefig(filename, dpi=150)
        plt.close()