"""
JAX を使用した線形方程式系ソルバー
"""

import os
import time
import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver


class JAXLinearSolver(LinearSolver):
    """JAX を使用した線形方程式系ソルバー"""
    
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
            
            # 解法メソッド辞書 - 必ず最初に定義
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
            self.cpu_solver = CPULinearSolver(
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
    
    def _to_numpy_vector(self, b):
        """ベクトルをNumPy形式に変換"""
        return np.array(b)
    
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
            return CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            ).solve(b, method, options)
    
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
        
        # JAX固有のCG実装
        matvec = self.jit_matvec if self.jit_matvec is not None else lambda x: A @ x
        
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
        matvec = self.jit_matvec if self.jit_matvec is not None else lambda x: A @ x
        
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
            beta = (rho / self.jnp.maximum(rho_prev, 1e-15)) * (omega / self.jnp.maximum(omega, 1e-15))
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
    
    def _solve_gmres(self, A, b, options=None):
        """JAX GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 20)
        
        # 初期値設定
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # GMRES実装
        def gmres_single_restart(x, b, tol, restart, matvec):
            """単一のGMRESリスタートを実装"""
            r = b - matvec(x)
            beta = self.jnp.linalg.norm(r)
            if beta <= tol:
                return x, 0
                
            # Krylov部分空間の基底
            V = self.jnp.zeros((b.shape[0], restart+1), dtype=b.dtype)
            V = V.at[:, 0].set(r / beta)
            
            # Hessenberg行列
            H = self.jnp.zeros((restart+1, restart), dtype=b.dtype)
            
            # Givens回転パラメータ
            cs = self.jnp.zeros(restart, dtype=b.dtype)
            sn = self.jnp.zeros(restart, dtype=b.dtype)
            
            # 残差ベクトル
            g = self.jnp.zeros(restart+1, dtype=b.dtype)
            g = g.at[0].set(beta)
            
            # Arnoldiプロセス実装
            def body_fun(j, val):
                V, H, cs, sn, g, iterations_inner = val
                
                # 新しいKrylov部分空間ベクトル
                w = matvec(V[:, j])
                
                # Modified Gram-Schmidt
                for i in range(j+1):
                    H = H.at[i, j].set(self.jnp.dot(w, V[:, i]))
                    w = w - H[i, j] * V[:, i]
                
                H = H.at[j+1, j].set(self.jnp.linalg.norm(w))
                
                # 0除算防止
                w = self.jnp.where(
                    H[j+1, j] > 1e-14, 
                    w / H[j+1, j], 
                    self.jnp.zeros_like(w)
                )
                V = V.at[:, j+1].set(w)
                
                # Givens回転の適用
                # 既存の回転を適用
                for i in range(j):
                    temp = cs[i] * H[i, j] + sn[i] * H[i+1, j]
                    H = H.at[i+1, j].set(-sn[i] * H[i, j] + cs[i] * H[i+1, j])
                    H = H.at[i, j].set(temp)
                
                # 新しい回転の計算
                if H[j+1, j] == 0:
                    cs = cs.at[j].set(1.0)
                    sn = sn.at[j].set(0.0)
                else:
                    if abs(H[j+1, j]) > abs(H[j, j]):
                        t = -H[j, j] / H[j+1, j]
                        s = 1.0 / self.jnp.sqrt(1.0 + t*t)
                        c = s * t
                    else:
                        t = -H[j+1, j] / H[j, j]
                        c = 1.0 / self.jnp.sqrt(1.0 + t*t)
                        s = c * t
                    cs = cs.at[j].set(c)
                    sn = sn.at[j].set(s)
                
                # 残差ベクトルの更新
                H = H.at[j, j].set(cs[j] * H[j, j] + sn[j] * H[j+1, j])
                H = H.at[j+1, j].set(0.0)
                g = g.at[j+1].set(-sn[j] * g[j])
                g = g.at[j].set(cs[j] * g[j])
                
                # 収束判定
                error = abs(g[j+1])
                iterations_inner = iterations_inner + 1
                
                return V, H, cs, sn, g, iterations_inner
            
            iterations_inner = 0
            val = (V, H, cs, sn, g, iterations_inner)
            
            # Arnoldiプロセスの実行
            for j in range(restart):
                val = body_fun(j, val)
                V, H, cs, sn, g, iterations_inner = val
                
                # 収束判定
                error = abs(g[j+1])
                if error <= tol:
                    break
            
            # 小さい方の問題を解く (Hy = g)
            y = self.jnp.zeros(restart, dtype=b.dtype)
            for j in range(min(restart, iterations_inner) - 1, -1, -1):
                y = y.at[j].set(g[j])
                for i in range(j+1, min(restart, iterations_inner)):
                    y = y.at[j].set(y[j] - H[j, i] * y[i])
                y = y.at[j].set(y[j] / H[j, j])
            
            # 解の更新
            for j in range(min(restart, iterations_inner)):
                x = x + y[j] * V[:, j]
                
            return x, iterations_inner
        
        # メインGMRESループ
        iterations_total = 0
        
        # 効率のために関数をJITコンパイル
        jit_gmres_restart = self.jax.jit(
            lambda x, b, tol, restart, matvec: gmres_single_restart(x, b, tol, restart, matvec),
            static_argnums=(2, 3)
        )
        
        # マトリックス-ベクトル積演算のためのJIT関数
        matvec = self.jit_matvec if self.jit_matvec is not None else lambda x: A @ x
        
        # メインループ
        for iter_count in range(maxiter):
            x0, inner_iterations = jit_gmres_restart(x0, b, tol, restart, matvec)
            iterations_total += inner_iterations
            
            # 収束チェック
            r = b - matvec(x0)
            if self.jnp.linalg.norm(r) <= tol * self.jnp.linalg.norm(b):
                break
        
        return x0, iterations_total
    
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