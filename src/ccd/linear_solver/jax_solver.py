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
            self.jax = jax
            self.jnp = jnp
            self.has_jax = True
            
            # 行列をJAX形式に変換
            self.A, self.jit_matvec = self._to_jax_matrix(self.original_A)
            
            # スケーリングの初期化
            if self.scaling_method:
                from scaling import plugin_manager
                self.scaler = plugin_manager.get_plugin(self.scaling_method)
                self._prepare_scaling()
            
            # 解法メソッド辞書
            self.solvers = {
                "direct": self._solve_direct,
                "cg": self._solve_cg,
                "bicgstab": self._solve_bicgstab
            }
        except ImportError:
            print("警告: JAXが利用できません。CPUソルバーを使用します。")
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
            
        # ダミーベクトルでスケーリング情報を計算（JAX密行列を使用）
        dummy_b = self.jnp.ones(self.A['shape'][0] if isinstance(self.A, dict) else self.A.shape[0])
        
        # スケーリング情報を保存
        try:
            if isinstance(self.A, dict) and 'dense' in self.A:
                A_for_scaling = self.A['dense']
            else:
                A_for_scaling = self.A
                
            _, _, self.scaling_info = self.scaler.scale(A_for_scaling, dummy_b)
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
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
                    b_scaled = self.scaler.scale_b_only(b_jax, self.scaling_info)
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
                x_jax = self.scaler.unscale(x_jax, self.scaling_info)
                
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
        cg_loop = self.jax.jit(lambda state: self.jax.lax.while_loop(cg_cond, cg_step, state))
        
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
        bicgstab_loop = self.jax.jit(lambda state: self.jax.lax.while_loop(bicgstab_cond, bicgstab_step, state))
        
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