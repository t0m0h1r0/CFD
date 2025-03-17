"""線形方程式系 Ax=b を解くためのソルバーモジュール"""

import os
import time
import numpy as np
import scipy.sparse.linalg as splinalg_cpu
from abc import ABC, abstractmethod
from scaling import plugin_manager


class LinearSolver(ABC):
    """線形方程式系 Ax=b を解くための抽象基底クラス"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        self.method = method
        self.options = options or {}
        self.scaling_method = scaling_method
        self.last_iterations = None
        self.monitor_convergence = self.options.get("monitor_convergence", False)
        
        # 継承クラスで拡張される解法辞書
        self.solvers = {"direct": self._solve_direct}
    
    @abstractmethod
    def solve(self, A, b):
        """線形方程式系を解く"""
        pass
    
    def _solve_system(self, A, b, residuals=None):
        """適切な解法で方程式系を解く"""
        if self.method not in self.solvers:
            print(f"未対応の解法: {self.method}、directに切り替えます")
            self.method = "direct"
        
        return self.solvers[self.method](A, b, residuals)
    
    def _create_callback(self, A, b, residuals):
        """収束モニタリング用コールバック関数"""
        if not self.monitor_convergence:
            return None
            
        def callback(xk):
            residual = self._calculate_residual(A, b, xk)
            residuals.append(float(residual))
            if len(residuals) % 10 == 0:
                print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        return callback
    
    def _calculate_residual(self, A, b, x):
        """残差計算"""
        return np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    
    def _solve_direct(self, A, b, *args):
        """直接解法（サブクラスでオーバーライド）"""
        raise NotImplementedError()
    
    def _visualize_convergence(self, residuals):
        """収束履歴を可視化"""
        if not residuals:
            return
            
        output_dir = self.options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals)+1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{self.method.upper()} ソルバーの収束履歴')
        
        prefix = self.options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_{self.method}.png")
        plt.savefig(filename, dpi=150)
        plt.close()


class CPULinearSolver(LinearSolver):
    """CPU (SciPy) を使用した線形方程式系ソルバー"""
    
    ITERATIVE_METHODS = ['cg', 'cgs', 'bicgstab', 'minres']
    LEAST_SQUARES_METHODS = ['lsqr', 'lsmr']
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        super().__init__(method, options, scaling_method)
        
        # 解法メソッド辞書を拡張
        self.solvers.update({
            'gmres': self._solve_gmres,
            'lsqr': self._solve_least_squares,
            'lsmr': self._solve_least_squares
        })
        
        # 反復解法を追加
        for method in self.ITERATIVE_METHODS:
            self.solvers[method] = self._solve_iterative
    
    def solve(self, A, b):
        """SciPy を使用して線形方程式系を解く"""
        start_time = time.time()
        residuals = []
        
        # 解法実行
        x, iterations = self._solve_system(A, b, residuals)
        
        # ログと可視化
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        print(f"CPU解法: {self.method}, 時間: {elapsed:.4f}秒" + (f", 反復: {iterations}" if iterations else ""))
        
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
        
        return x
    
    def _solve_direct(self, A, b, *args):
        """直接解法"""
        return splinalg_cpu.spsolve(A, b), None
    
    def _solve_gmres(self, A, b, residuals):
        """GMRES法"""
        tol = self.options.get("tol", 1e-10)
        maxiter = self.options.get("maxiter", 1000)
        restart = self.options.get("restart", 100)
        x0 = np.ones_like(b)
        
        callback = self._create_callback(A, b, residuals)
        
        try:
            return splinalg_cpu.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                      restart=restart, callback=callback)
        except TypeError:
            return splinalg_cpu.gmres(A, b, restart=restart)
    
    def _solve_iterative(self, A, b, residuals):
        """反復解法共通インターフェース"""
        tol = self.options.get("tol", 1e-10)
        maxiter = self.options.get("maxiter", 1000)
        x0 = np.ones_like(b)
        
        callback = self._create_callback(A, b, residuals)
        solver_func = getattr(splinalg_cpu, self.method)
        
        try:
            return solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        except TypeError:
            return solver_func(A, b)
    
    def _solve_least_squares(self, A, b, *args):
        """最小二乗法ソルバー"""
        maxiter = self.options.get("maxiter", 1000)
        solver_func = getattr(splinalg_cpu, self.method)
        
        try:
            x = solver_func(A, b, iter_lim=maxiter)[0]
        except TypeError:
            x = solver_func(A, b)[0]
        
        return x, None


class GPULinearSolver(LinearSolver):
    """GPU (CuPy) を使用した線形方程式系ソルバー"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        super().__init__(method, options, scaling_method)
        self.A_gpu = None
        self.A_scaled_gpu = None
        self.scaling_info = None
        self.scaler = None
        self.original_A = None
        
        # CuPyが利用可能か確認
        try:
            import cupy as cp
            import cupyx.scipy.sparse.linalg as splinalg
            self.cp = cp
            self.splinalg = splinalg
            
            # 解法メソッド辞書を拡張
            self.solvers.update({
                'gmres': self._solve_gmres,
                'cg': self._solve_iterative,
                'cgs': self._solve_iterative,
                'minres': self._solve_iterative,
                'lsqr': self._solve_least_squares,
                'lsmr': self._solve_least_squares
            })
        except ImportError:
            self.cp = None
    
    def __del__(self):
        """デストラクタ：GPU上のメモリを解放"""
        self.A_gpu = None
        self.A_scaled_gpu = None
    
    def _is_new_matrix(self, A):
        """現在のGPU行列と異なる行列かを判定"""
        return (self.A_gpu is None or self.original_A is None or 
                self.A_gpu.shape != A.shape or self.original_A is not A)
    
    def _apply_scaling(self, A, b):
        """行列と右辺ベクトルにスケーリングを適用"""
        if self.scaling_method is not None:
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング: {scaler.name}")
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
        
        return A, b, None, None
    
    def solve(self, A, b):
        """CuPyを使用して線形方程式系を解く"""
        if self.cp is None:
            print("CuPyが利用できません。CPUソルバーに切り替えます。")
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        start_time = time.time()
        residuals = []
        
        # 行列準備（新しい行列か再利用）
        is_new_matrix = self._is_new_matrix(A)
        
        if is_new_matrix:
            print("行列をGPUに転送中...")
            self.A_gpu = self.cp.sparse.csr_matrix(A)
            self.original_A = A
            self.A_scaled_gpu = None
        
        # bをGPUに転送
        b_gpu = self.cp.array(b)
        
        # スケーリング適用または再利用
        if is_new_matrix or self.A_scaled_gpu is None:
            self.A_scaled_gpu, b_scaled, self.scaling_info, self.scaler = self._apply_scaling(self.A_gpu, b_gpu)
        else:
            if self.scaling_method is not None and self.scaler is not None:
                if hasattr(self.scaler, 'scale_b_only'):
                    b_scaled = self.scaler.scale_b_only(b_gpu, self.scaling_info)
                else:
                    _, b_scaled, _, _ = self.scaler.scale(self.A_gpu, b_gpu)
            else:
                b_scaled = b_gpu
        
        # GPU で計算実行
        try:
            x_gpu, iterations = self._solve_system(self.A_scaled_gpu, b_scaled, residuals)
            
            # スケーリングを戻す
            if self.scaling_info is not None and self.scaler is not None:
                x_gpu = self.scaler.unscale(x_gpu, self.scaling_info)
                
            # 結果をCPUに転送
            x = x_gpu.get()
        except Exception as e:
            print(f"GPU処理エラー: {e}、CPUに切り替えます")
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        # ログと可視化
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        print(f"GPU解法: {self.method}, 時間: {elapsed:.4f}秒" + (f", 反復: {iterations}" if iterations else ""))
        
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
        
        return x
    
    def _calculate_residual(self, A, b, x):
        """GPU用の残差計算"""
        return self.cp.linalg.norm(b - A @ x) / self.cp.linalg.norm(b)
    
    def _solve_direct(self, A, b, *args):
        """CuPyの直接解法"""
        return self.splinalg.spsolve(A, b), None
    
    def _solve_gmres(self, A, b, residuals):
        """CuPyのGMRES法"""
        tol = self.options.get("tol", 1e-10)
        maxiter = self.options.get("maxiter", 1000)
        restart = self.options.get("restart", 100)
        x0 = self.cp.ones_like(b)
        
        callback = self._create_callback(A, b, residuals)
        
        return self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                  restart=restart)
    
    def _solve_iterative(self, A, b, residuals):
        """CuPy反復解法共通インターフェース"""
        tol = self.options.get("tol", 1e-10)
        maxiter = self.options.get("maxiter", 1000)
        x0 = self.cp.ones_like(b)
        
        callback = self._create_callback(A, b, residuals)
        solver_func = getattr(self.splinalg, self.method)
        
        return solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
    
    def _solve_least_squares(self, A, b, *args):
        """CuPy最小二乗法ソルバー"""
        maxiter = self.options.get("maxiter", 1000)
        solver_func = getattr(self.splinalg, self.method)
        
        if self.method == "lsmr":
            x0 = self.cp.ones_like(b)
            x = solver_func(A, b, x0=x0, maxiter=maxiter)[0]
        else:
            x = solver_func(A, b)[0]
            
        return x, None


class JAXLinearSolver(LinearSolver):
    """JAX を使用した線形方程式系ソルバー"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        super().__init__(method, options, scaling_method)
        self.last_matrix = None
        self.last_op = None
        
        # JAXが利用可能か確認
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            
            # 解法メソッド辞書を拡張
            self.solvers.update({
                'cg': self._solve_cg,
                'bicgstab': self._solve_bicgstab
            })
        except ImportError:
            self.jax = None
    
    def solve(self, A, b):
        """JAX を使用して線形方程式系を解く"""
        if self.jax is None:
            print("JAXが利用できません。CPUソルバーに切り替えます。")
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        start_time = time.time()
        residuals = []
        
        # JAX形式に変換
        self._prepare_jax_matrix(A, b)
        
        # 解法実行
        try:
            x, iterations = self._solve_system(A, b, residuals)
        except Exception as e:
            print(f"JAX処理エラー: {e}、CPUに切り替えます")
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        # ログと可視化
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        print(f"JAX解法: {self.method}, 時間: {elapsed:.4f}秒" + (f", 反復: {iterations}" if iterations else ""))
        
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
        
        return np.array(x)
    
    def _prepare_jax_matrix(self, A, b):
        """JAX形式に行列を準備"""
        from jax.experimental import sparse as jsparse
        
        # 行列と右辺ベクトルをJAX形式に変換
        data = self.jnp.array(A.data)
        indices = self.jnp.array(A.indices)
        indptr = self.jnp.array(A.indptr)
        self.b_jax = self.jnp.array(b)
        self.shape = A.shape
        
        # 行列-ベクトル積を定義
        def matvec(x):
            return jsparse.csr_matvec(data, indices, indptr, self.shape[1], x)
        
        # 新しい行列の場合、JAX操作をコンパイル
        if (self.last_matrix is None or self.last_matrix[0] is not A or 
            self.last_matrix[1].shape != self.shape):
            self.last_op = self.jax.jit(matvec)
            self.last_matrix = (A, A)
        
        # 密行列に変換（直接解法用）
        self.dense_A = A.toarray()
        self.dense_A_jax = self.jnp.array(self.dense_A)
    
    def _solve_direct(self, A, b, *args):
        """JAXの直接解法を使用"""
        try:
            x = self.jnp.linalg.solve(self.dense_A_jax, self.b_jax)
            return x, None
        except Exception as e:
            x = np.linalg.solve(self.dense_A, b)
            return self.jnp.array(x), None
    
    def _solve_cg(self, A, b, residuals):
        """JAX共役勾配法"""
        tol = self.options.get("tol", 1e-10)
        maxiter = self.options.get("maxiter", 1000)
        
        # 初期値設定
        x0 = self.jnp.ones_like(self.b_jax)
        r0 = self.b_jax - self.last_op(x0)
        p0 = r0
        
        # JAX CG実装
        def cg_step(state):
            x_k, r_k, p_k, k = state
            Ap_k = self.last_op(p_k)
            alpha_k = self.jnp.dot(r_k, r_k) / self.jnp.maximum(self.jnp.dot(p_k, Ap_k), 1e-15)
            x_next = x_k + alpha_k * p_k
            r_next = r_k - alpha_k * Ap_k
            beta_k = self.jnp.dot(r_next, r_next) / self.jnp.maximum(self.jnp.dot(r_k, r_k), 1e-15)
            p_next = r_next + beta_k * p_k
            
            # 収束モニタリング
            if residuals is not None and self.monitor_convergence:
                residual = self.jnp.linalg.norm(r_next) / self.jnp.linalg.norm(self.b_jax)
                self.jax.debug.callback(lambda r, i: residuals.append(float(r)) or print(f"  反復 {i}: 残差 = {r:.6e}") if i % 10 == 0 else None, residual, k)
                
            return x_next, r_next, p_next, k + 1
            
        def cg_cond(val):
            x_k, r_k, p_k, k = val
            return (self.jnp.linalg.norm(r_k) > tol) & (k < maxiter)
        
        # JITコンパイルして実行
        cg_loop = self.jax.jit(lambda state: self.jax.lax.while_loop(cg_cond, cg_step, state))
        x_final, r_final, _, iterations = cg_loop((x0, r0, p0, 0))
        
        return x_final, int(iterations)
    
    def _solve_bicgstab(self, A, b, residuals):
        """JAX双共役勾配法安定化版"""
        tol = self.options.get("tol", 1e-10)
        maxiter = self.options.get("maxiter", 1000)
        
        # 初期値設定
        x0 = self.jnp.ones_like(self.b_jax)
        r0 = self.b_jax - self.last_op(x0)
        r_hat = r0
        v0 = self.last_op(r0)
        
        # JAX BiCGSTAB実装
        def bicgstab_step(state):
            x_k, r_k, p_k, v_k, r_hat, rho_prev, k, omega = state
            
            rho = self.jnp.dot(r_hat, r_k)
            beta = (rho / self.jnp.maximum(rho_prev, 1e-15)) * (omega / self.jnp.maximum(omega, 1e-15))
            p_next = r_k + beta * (p_k - omega * v_k)
            
            v_next = self.last_op(p_next)
            alpha = rho / self.jnp.maximum(self.jnp.dot(r_hat, v_next), 1e-15)
            
            s = r_k - alpha * v_next
            t = self.last_op(s)
            
            omega_next = self.jnp.dot(t, s) / self.jnp.maximum(self.jnp.dot(t, t), 1e-15)
            x_next = x_k + alpha * p_next + omega_next * s
            r_next = s - omega_next * t
            
            # 収束モニタリング
            if residuals is not None and self.monitor_convergence:
                residual = self.jnp.linalg.norm(r_next) / self.jnp.linalg.norm(self.b_jax)
                self.jax.debug.callback(lambda r, i: residuals.append(float(r)) or print(f"  反復 {i}: 残差 = {r:.6e}") if i % 10 == 0 else None, residual, k)
                
            return x_next, r_next, p_next, v_next, r_hat, rho, k + 1, omega_next
            
        def bicgstab_cond(val):
            _, r_k, _, _, _, _, k, _ = val
            return (self.jnp.linalg.norm(r_k) > tol) & (k < maxiter)
        
        # JITコンパイルして実行
        bicgstab_loop = self.jax.jit(lambda state: self.jax.lax.while_loop(bicgstab_cond, bicgstab_step, state))
        
        try:
            init_state = (x0, r0, r0, v0, r_hat, self.jnp.dot(r_hat, r0), 0, 1.0)
            x_final, r_final, _, _, _, _, iterations, _ = bicgstab_loop(init_state)
            return x_final, int(iterations)
        except Exception as e:
            # フォールバック
            import scipy.sparse.linalg as splinalg
            x_np, _ = splinalg.bicgstab(self.dense_A, b)
            return self.jnp.array(x_np), None


def create_solver(method="direct", options=None, scaling_method=None, backend="cuda"):
    """適切な線形ソルバーを作成するファクトリ関数"""
    # バックエンド設定
    backend = options.get("backend", backend) if options else backend
    
    # ソルバーマップ
    solvers = {
        "cpu": CPULinearSolver,
        "cuda": GPULinearSolver,
        "jax": JAXLinearSolver
    }
    
    # 指定されたバックエンドのソルバー作成
    solver_class = solvers.get(backend, CPULinearSolver)
    
    try:
        return solver_class(method, options, scaling_method)
    except Exception as e:
        print(f"{backend}ソルバー初期化エラー: {e}")
        if backend != "cpu":
            return CPULinearSolver(method, options, scaling_method)
        raise