"""
JAX を使用した線形方程式系ソルバー

This module provides solvers for linear systems Ax = b using JAX's
numerical computing capabilities. It supports direct and iterative
methods with various convergence options and scaling techniques.
All implementations are designed to be JIT-friendly and follow
JAX's functional programming paradigm.
"""

import os
import time
import numpy as np
from .base import LinearSolver


class JAXLinearSolver(LinearSolver):
    """JAX を使用した線形方程式系ソルバー
    
    This solver leverages JAX's high-performance numerical computing
    capabilities to solve linear systems on accelerators. It supports
    direct solvers and iterative methods like CG and BiCGSTAB,
    with implementations designed for JIT compilation and automatic
    differentiation support.
    """
    
    def _initialize(self):
        """JAX固有の初期化処理"""
        try:
            import jax
            import jax.numpy as jnp
            from jax import lax
            self.jax = jax
            self.jnp = jnp
            self.lax = lax
            self.has_jax = True
            
            # 解法メソッド辞書 - 必ず最初に定義（エラー回避のため）
            self.solvers = {
                "direct": self._solve_direct,
                "cg": self._solve_cg,
                "bicgstab": self._solve_bicgstab,
                "gmres": self._solve_gmres
            }
            
            # 行列をJAX形式に変換
            self.A, self.jit_matvec = self._to_jax_matrix(self.original_A)
            
            # スケーリングの初期化
            if self.scaling_method:
                from scaling import plugin_manager
                self.scaler = plugin_manager.get_plugin(self.scaling_method)
                self._prepare_scaling()
            
        except ImportError as e:
            print(f"警告: JAXが利用できません: {e}")
            self.has_jax = False
            self.cpu_solver = self._create_cpu_fallback()
    
    def _create_cpu_fallback(self):
        """CPU fallbackソルバーを作成"""
        from .cpu_solver import CPULinearSolver
        return CPULinearSolver(
            self.original_A, 
            self.enable_dirichlet, 
            self.enable_neumann, 
            self.scaling_method
        )
    
    def _to_jax_matrix(self, A):
        """行列をJAX形式に変換"""
        try:
            # JAXですでに処理されているか確認
            if hasattr(A, 'shape') and hasattr(A, 'dtype') and str(type(A)).find('jax') >= 0:
                return A, None
                
            # JAX用にCSRデータを抽出
            if hasattr(A, 'tocsr'):
                A = A.tocsr()
                
            # 疎行列の場合
            if hasattr(A, 'data') and hasattr(A, 'indices') and hasattr(A, 'indptr'):
                from jax.experimental import sparse as jsparse
                
                # JAX形式の疎行列データを構築
                data = self.jnp.array(A.data)
                indices = self.jnp.array(A.indices)
                indptr = self.jnp.array(A.indptr)
                shape = A.shape
                
                # JAX用のマトリックス-ベクトル演算子を定義
                def matvec(x):
                    return jsparse.csr_matvec(data, indices, indptr, shape[1], x)
                
                # JITコンパイル
                jit_matvec = self.jax.jit(matvec)
                
                # JAX行列表現
                jax_matrix = {
                    'data': data, 
                    'indices': indices, 
                    'indptr': indptr, 
                    'shape': shape, 
                    'matvec': jit_matvec
                }
                
                # 密行列も準備
                if hasattr(A, 'toarray'):
                    jax_matrix['dense'] = self.jnp.array(A.toarray())
                
                return jax_matrix, jit_matvec
            else:
                # 密行列の場合
                jax_matrix = self.jnp.array(A)
                
                # 行列-ベクトル積関数
                def matvec(x):
                    return jax_matrix @ x
                jit_matvec = self.jax.jit(matvec)
                
                return jax_matrix, jit_matvec
        except Exception as e:
            print(f"JAX行列変換エラー: {e}")
            return A, None
    
    def _to_jax_vector(self, b):
        """ベクトルをJAX配列に変換"""
        try:
            # すでにJAX配列の場合
            if str(type(b)).find('jax') >= 0:
                return b
                
            # NumPy/CuPy配列からJAX配列に変換
            if hasattr(b, 'get'):  # CuPy
                return self.jnp.array(b.get())
            else:
                return self.jnp.array(b)
        except Exception as e:
            print(f"JAX変換エラー: {e}")
            return b
    
    def _prepare_scaling(self):
        """スケーリング前処理"""
        if not self.scaler or not self.has_jax:
            return
            
        # NumPy用ダミーベクトルでスケーリング情報を計算
        dummy_b = np.ones(self.A['shape'][0] if isinstance(self.A, dict) else self.A.shape[0])
        
        # スケーリング情報を保存
        try:
            # NumPy版の行列を作成
            if isinstance(self.A, dict) and 'dense' in self.A:
                A_np = np.array(self.A['dense'])
            else:
                A_np = self._to_numpy_matrix(self.A)
                
            # NumPyでスケーリング情報を計算
            _, _, scale_info_np = self.scaler.scale(A_np, dummy_b)
            
            # スケーリング情報をJAXに変換
            self.scaling_info = {}
            for key, value in scale_info_np.items():
                if isinstance(value, np.ndarray):
                    self.scaling_info[key] = self.jnp.array(value)
                else:
                    self.scaling_info[key] = value
                    
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def _to_numpy_matrix(self, A):
        """行列をNumPy形式に変換"""
        if isinstance(A, dict) and 'dense' in A:
            return np.array(A['dense'])
        elif isinstance(A, dict):
            # 適切なCSR形式に変換
            import scipy.sparse as sp
            return sp.csr_matrix(
                (np.array(A['data']), np.array(A['indices']), np.array(A['indptr'])),
                shape=A['shape']
            )
        else:
            return np.array(A)
    
    def _to_numpy_scaling_info(self):
        """スケーリング情報をNumPy形式に変換"""
        numpy_info = {}
        for key, value in self.scaling_info.items():
            if hasattr(value, 'shape') and str(type(value)).find('jax') >= 0:
                numpy_info[key] = np.array(value)
            else:
                numpy_info[key] = value
        return numpy_info
    
    def solve(self, b, method="direct", options=None):
        """JAXを使用して線形方程式系を解く"""
        # JAXが使えない場合はCPUソルバーにフォールバック
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
        start_time = time.time()
        options = options or {}
        
        try:
            # 右辺ベクトルbをJAX形式に変換
            b_jax = self._to_jax_vector(b)
            
            # スケーリングの適用
            b_scaled = b_jax
            if self.scaler and self.scaling_info:
                try:
                    # スケーリングAPIを使用して右辺ベクトルをスケーリング
                    b_np = self._to_numpy_vector(b_jax)
                    b_np_scaled = self.scaler.scale_b_only(b_np, self._to_numpy_scaling_info())
                    b_scaled = self._to_jax_vector(b_np_scaled)
                except Exception as e:
                    print(f"スケーリングエラー: {e}")
            
            # 解法メソッドの選択
            if method not in self.solvers:
                print(f"JAXで未対応の解法: {method}、directに切り替えます")
                method = "direct"
            
            # 線形システムを解く
            solver_func = self.solvers[method]
            x_jax, iterations = solver_func(self.A, b_scaled, options)
            
            # 結果のアンスケーリング
            if self.scaler and self.scaling_info:
                try:
                    # スケーリングAPIを使用して解ベクトルをアンスケーリング
                    x_np = self._to_numpy_vector(x_jax)
                    x_np_unscaled = self.scaler.unscale(x_np, self._to_numpy_scaling_info())
                    x_jax = self._to_jax_vector(x_np_unscaled)
                except Exception as e:
                    print(f"アンスケーリングエラー: {e}")
                
            # JAX結果をNumPyに変換
            x = np.array(x_jax)
            
            # 計算結果の記録
            self.last_iterations = iterations
            elapsed = time.time() - start_time
            print(f"JAX解法: {method}, 時間: {elapsed:.4f}秒" + 
                  (f", 反復: {iterations}" if iterations else ""))
                  
            return x
                
        except Exception as e:
            print(f"JAX解法エラー: {e}, CPUに切り替えます")
            # CPUソルバーにフォールバック
            return self._create_cpu_fallback().solve(b, method, options)

    def _to_numpy_vector(self, b):
        """ベクトルをNumPy形式に変換"""
        return np.array(b)
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        try:
            # 密行列使用
            if isinstance(A, dict) and 'dense' in A:
                A_dense = A['dense']
            else:
                A_dense = A
                
            x = self.jnp.linalg.solve(A_dense, b)
            return x, None
        except Exception as e:
            print(f"JAX直接解法エラー: {e}, CPUにフォールバック")
            # CPUにフォールバック
            import scipy.sparse.linalg as splinalg
            x = splinalg.spsolve(self._to_numpy_matrix(A), self._to_numpy_vector(b))
            return self.jnp.array(x), None
    
    def _solve_cg(self, A, b, options=None):
        """JAX共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期値設定
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # 行列ベクトル積関数を取得
        matvec = A['matvec'] if isinstance(A, dict) and 'matvec' in A else lambda x: A @ x
        
        # 残差と初期値
        r0 = b - matvec(x0)
        p0 = r0
        
        # 収束モニタリング
        residuals = []
        
        # JAX CG実装
        def cg_step(state):
            x_k, r_k, p_k, k = state
            Ap_k = matvec(p_k)
            alpha_k = self.jnp.dot(r_k, r_k) / self.jnp.maximum(self.jnp.dot(p_k, Ap_k), 1e-15)
            x_next = x_k + alpha_k * p_k
            r_next = r_k - alpha_k * Ap_k
            beta_k = self.jnp.dot(r_next, r_next) / self.jnp.maximum(self.jnp.dot(r_k, r_k), 1e-15)
            p_next = r_next + beta_k * p_k
            
            # 収束モニタリング
            if options.get("monitor_convergence", False):
                residual = self.jnp.linalg.norm(r_next) / self.jnp.linalg.norm(b)
                self.jax.debug.callback(lambda r, i: residuals.append(float(r)) or print(f"  反復 {i}: 残差 = {r:.6e}") if i % 10 == 0 else None, residual, k)
                
            return x_next, r_next, p_next, k + 1
            
        def cg_cond(val):
            x_k, r_k, p_k, k = val
            return (self.jnp.linalg.norm(r_k) > tol * self.jnp.linalg.norm(b)) & (k < maxiter)
        
        # JITコンパイルして実行
        cg_loop = self.jax.jit(lambda state: self.lax.while_loop(cg_cond, cg_step, state))
        
        # 初期状態でループを実行
        x_final, r_final, _, iterations = cg_loop((x0, r0, p0, 0))
        
        # 収束履歴を可視化（オプション）
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, "cg", options)
        
        return x_final, int(iterations)
    
    def _solve_bicgstab(self, A, b, options=None):
        """JAX双共役勾配法安定化版"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期値設定
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # JAX固有のBiCGSTAB実装
        matvec = A['matvec'] if isinstance(A, dict) and 'matvec' in A else lambda x: A @ x
        
        # 残差と初期値
        r0 = b - matvec(x0)
        r_hat = r0  # 影残差
        rho_prev = 1.0
        alpha = 1.0
        omega = 1.0
        p = self.jnp.zeros_like(b)
        v = self.jnp.zeros_like(b)
        
        # 収束モニタリング
        residuals = []
        
        # JAX BiCGSTAB実装
        def bicgstab_step(state):
            x_k, r_k, p_k, v_k, r_hat, rho_prev, k, omega = state
            
            rho = self.jnp.dot(r_hat, r_k)
            beta = (rho / self.jnp.maximum(rho_prev, 1e-15)) * (alpha / self.jnp.maximum(omega, 1e-15))
            p_next = r_k + beta * (p_k - omega * v_k)
            
            v_next = matvec(p_next)
            alpha = rho / self.jnp.maximum(self.jnp.dot(r_hat, v_next), 1e-15)
            
            s = r_k - alpha * v_next
            t = matvec(s)
            
            omega_next = self.jnp.dot(t, s) / self.jnp.maximum(self.jnp.dot(t, t), 1e-15)
            x_next = x_k + alpha * p_next + omega_next * s
            r_next = s - omega_next * t
            
            # 収束モニタリング
            if options.get("monitor_convergence", False):
                residual = self.jnp.linalg.norm(r_next) / self.jnp.linalg.norm(b)
                self.jax.debug.callback(lambda r, i: residuals.append(float(r)) or print(f"  反復 {i}: 残差 = {r:.6e}") if i % 10 == 0 else None, residual, k)
                
            return x_next, r_next, p_next, v_next, r_hat, rho, k + 1, omega_next
            
        def bicgstab_cond(val):
            _, r_k, _, _, _, _, k, _ = val
            return (self.jnp.linalg.norm(r_k) > tol * self.jnp.linalg.norm(b)) & (k < maxiter)
        
        # JITコンパイルして実行
        bicgstab_loop = self.jax.jit(lambda state: self.lax.while_loop(bicgstab_cond, bicgstab_step, state))
        
        # 初期状態でループを実行
        try:
            init_state = (x0, r0, p, v, r_hat, rho_prev, 0, omega)
            x_final, r_final, _, _, _, _, iterations, _ = bicgstab_loop(init_state)
            
            # 収束履歴を可視化（オプション）
            if options.get("monitor_convergence", False) and residuals:
                self._visualize_convergence(residuals, "bicgstab", options)
            
            return x_final, int(iterations)
        except Exception as e:
            print(f"JAX BiCGSTAB実装エラー: {e}, CPUにフォールバック")
            # CPUにフォールバック
            import scipy.sparse.linalg as splinalg
            x = splinalg.bicgstab(self._to_numpy_matrix(A), self._to_numpy_vector(b), tol=tol, maxiter=maxiter)[0]
            return self.jnp.array(x), None
    
    """
    JAX GMRES Implementation for CCD Solver

    This implementation properly handles CSR matrix representations and
    ensures compatibility with JAX's sparse linear algebra functions.
    """

    def _solve_gmres(self, A, b, options=None):
        """JAX-compatible GMRES solver implementation"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", min(1000, b.shape[0]))
        restart = options.get("restart", min(20, b.shape[0]))
        
        # Initial guess
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # Matrix-vector product function
        if isinstance(A, dict) and 'matvec' in A:
            # Handle CSR matrix representation
            matvec = A['matvec']
        else:
            # Handle dense matrix
            matvec = lambda x: A @ x
        
        # Residual monitoring
        residuals = []
        if options.get("monitor_convergence", False):
            def callback(x, i):
                r_norm = self.jnp.linalg.norm(b - matvec(x))
                b_norm = self.jnp.linalg.norm(b)
                residual = r_norm / b_norm
                residuals.append(float(residual))
                if i % 10 == 0:
                    print(f"  反復 {i}: 残差 = {residual:.6e}")
                return None
        else:
            callback = None
        
        try:
            # Try using JAX's scipy GMRES implementation if available
            try:
                from jax import scipy as jsp
                from jax.experimental import sparse
                
                # Use JAX's GMRES implementation
                result = jsp.sparse.linalg.gmres(
                    matvec,  # Use our prepared matvec function
                    b,
                    x0=x0,
                    atol=tol,  # Note: changed from tol to atol for compatibility
                    restart=restart,
                    maxiter=maxiter,
                    callback=callback
                )
                
                # Extract result and iteration count
                x = result[0]
                iters = result[1]
            except (ImportError, AttributeError) as e:
                raise ImportError(f"JAX scipy GMRES not available: {e}")
                
        except ImportError as e:
            print(f"Using custom JAX GMRES implementation: {e}")
            # Fall back to custom implementation
            x, iters = self._custom_gmres(
                matvec, b, x0, tol=tol, restart=restart, maxiter=maxiter, callback=callback)
        
        # Visualize convergence history if requested
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, "gmres", options)
        
        return x, int(iters)

    def _custom_gmres(self, A_op, b, x0, *, tol=1e-10, restart=20, maxiter=1000, callback=None):
        """
        Custom JAX-compatible GMRES implementation
        
        This implementation follows the restarted GMRES algorithm and is fully compatible
        with JAX's transformations.
        """
        jnp = self.jnp
        lax = self.lax
        
        # Prepare constants and initial values
        b_norm = jnp.linalg.norm(b)
        tol_threshold = tol * b_norm
        
        # Initialize solution
        x = x0
        
        # Main GMRES iterations
        def outer_loop(i_outer, state):
            x, converged = state
            
            # Compute residual
            r = b - A_op(x)
            r_norm = jnp.linalg.norm(r)
            
            # Skip Arnoldi if already converged or at max iterations
            def run_arnoldi():
                # Prepare for Arnoldi process
                Q = jnp.zeros((b.shape[0], restart+1), dtype=b.dtype)
                H = jnp.zeros((restart+1, restart), dtype=b.dtype)
                
                # First basis vector: normalized residual
                q1 = r / r_norm
                Q = Q.at[:, 0].set(q1)
                
                # Initialize Givens rotation parameters
                cs = jnp.ones(restart)
                sn = jnp.zeros(restart)
                e1 = jnp.zeros(restart+1, dtype=b.dtype)
                e1 = e1.at[0].set(r_norm)
                
                # Arnoldi process
                def arnoldi_step(j, state):
                    Q, H, cs, sn, e1, m_iters = state
                    
                    # Get new Krylov vector
                    w = A_op(Q[:, j])
                    
                    # Modified Gram-Schmidt orthogonalization
                    def gs_loop(k, w_h):
                        w, h_col = w_h
                        h_jk = jnp.vdot(Q[:, k], w)
                        w = w - h_jk * Q[:, k]
                        h_col = h_col.at[k].set(h_jk)
                        return (w, h_col)
                    
                    # Run Gram-Schmidt
                    w, h_col = lax.fori_loop(0, j+1, gs_loop, (w, jnp.zeros(j+2, dtype=b.dtype)))
                    
                    # Normalize and store the new basis vector
                    h_j1j = jnp.linalg.norm(w)
                    h_col = h_col.at[j+1].set(h_j1j)
                    H = H.at[:j+2, j].set(h_col)
                    
                    # Prevent division by zero
                    safe_norm = jnp.where(h_j1j > 1e-14, h_j1j, 1.0)
                    Q = Q.at[:, j+1].set(w / safe_norm)
                    
                    # Apply previous Givens rotations to the new column
                    def apply_rotations(k, h_col):
                        # Apply the k-th rotation to the k-th and (k+1)-th elements
                        temp = cs[k] * h_col[k] + sn[k] * h_col[k+1]
                        h_col = h_col.at[k+1].set(-sn[k] * h_col[k] + cs[k] * h_col[k+1])
                        h_col = h_col.at[k].set(temp)
                        return h_col
                    
                    h_col = lax.fori_loop(0, j, apply_rotations, H[:j+2, j])
                    H = H.at[:j+2, j].set(h_col)
                    
                    # Compute new Givens rotation
                    h1 = H[j, j]
                    h2 = H[j+1, j]
                    temp = jnp.sqrt(h1*h1 + h2*h2)
                    temp = jnp.where(temp > 1e-14, temp, 1.0)  # Prevent division by zero
                    
                    cs = cs.at[j].set(h1 / temp)
                    sn = sn.at[j].set(h2 / temp)
                    
                    # Apply new Givens rotation to H and e1
                    H = H.at[j, j].set(temp)
                    H = H.at[j+1, j].set(0.0)
                    
                    # Update residual vector
                    e1 = e1.at[j+1].set(-sn[j] * e1[j])
                    e1 = e1.at[j].set(cs[j] * e1[j])
                    
                    # Check convergence
                    resid = jnp.abs(e1[j+1])
                    
                    # Callback for monitoring
                    if callback is not None:
                        # Solve the triangular system to get the current solution
                        y = jnp.linalg.solve(H[:j+1, :j+1], e1[:j+1])
                        x_cur = x + Q[:, :j+1] @ y
                        callback(x_cur, i_outer * restart + j)
                    
                    # Return updated state
                    return Q, H, cs, sn, e1, m_iters + 1
                
                # Run Arnoldi for up to restart iterations
                init_state = (Q, H, cs, sn, e1, 0)
                
                # Define convergence condition
                def arnoldi_cond(state):
                    _, _, _, _, e1, m_iters = state
                    return (jnp.abs(e1[m_iters]) > tol_threshold) & (m_iters < restart)
                
                # Run the Arnoldi process
                Q, H, cs, sn, e1, m_iters = lax.while_loop(
                    arnoldi_cond, arnoldi_step, init_state)
                
                # Solve the least-squares problem
                y = jnp.linalg.solve(H[:m_iters, :m_iters], e1[:m_iters])
                
                # Update solution
                x_new = x + Q[:, :m_iters] @ y
                
                # Check if we've converged
                r_new = b - A_op(x_new)
                r_new_norm = jnp.linalg.norm(r_new)
                converged = r_new_norm <= tol_threshold
                
                return x_new, converged, r_new_norm
            
            # Skip Arnoldi if already converged (early return)
            already_converged = converged | (r_norm <= tol_threshold)
            x_new, new_converged, _ = lax.cond(
                already_converged,
                lambda: (x, True, r_norm),  # Keep current state
                run_arnoldi  # Run Arnoldi process
            )
            
            # Return updated state
            return x_new, new_converged
        
        # Initialize state
        state = (x0, False)
        
        # Define convergence condition for outer iterations
        def outer_cond(state_iter):
            state, iter_count = state_iter
            _, converged = state
            return (~converged) & (iter_count < jnp.ceil(maxiter / restart))
        
        # Run outer iterations
        def outer_step(state_iter):
            state, iter_count = state_iter
            new_state = outer_loop(iter_count, state)
            return (new_state, iter_count + 1)
        
        # Run the restarts
        (x, _), total_restarts = lax.while_loop(
            outer_cond, outer_step, ((x0, False), 0))
        
        # Estimate total iterations
        total_iters = jnp.minimum(total_restarts * restart, maxiter)
        
        return x, total_iters
    
    def _visualize_convergence(self, residuals, method_name, options):
        """収束履歴を可視化"""
        output_dir = options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals)+1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{method_name.upper()} ソルバーの収束履歴 (JAX)')
        
        prefix = options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_jax_{method_name}.png")
        plt.savefig(filename, dpi=150)
        plt.close()