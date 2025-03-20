"""
JAX を使用した線形方程式系ソルバー（改善版）
"""

import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver

class JAXLinearSolver(LinearSolver):
    """JAX を使用した線形方程式系ソルバー（堅牢化）"""
    
    def _initialize(self):
        """JAX固有の初期化処理"""
        try:
            import jax
            import jax.numpy as jnp
            import jax.scipy.sparse.linalg as splinalg
            
            # JAXの初期化設定
            jax.config.update('jax_enable_x64', True)  # 倍精度演算を有効化
            jax.config.update('jax_platform_name', 'gpu')  # GPUを優先使用
            
            self.jax = jax
            self.jnp = jnp
            self.splinalg = splinalg
            self.has_jax = True
            
            # 行列をJAX形式に変換
            self.A = self._to_jax_matrix(self.original_A)
            
            # スケーリングの初期化
            if self.scaling_method:
                from scaling import plugin_manager
                self.scaler = plugin_manager.get_plugin(self.scaling_method)
                self._prepare_scaling()
            
            # 解法メソッド辞書
            self.solvers = {
                "direct": self._solve_direct,
                "gmres": self._solve_gmres,
                "cg": self._solve_cg,
                "bicgstab": self._solve_bicgstab
            }
            
        except (ImportError, Exception) as e:
            print(f"警告: JAXが利用できないか初期化エラー: {e}")
            self.has_jax = False
            self.cpu_solver = CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            )
            # CPU版のsolversをコピー
            self.solvers = self.cpu_solver.solvers if hasattr(self.cpu_solver, 'solvers') else {}
    
    def _to_jax_matrix(self, A):
        """行列をJAX形式に変換（堅牢化）"""
        try:
            # JAXですでに処理されているか確認
            if 'jax' in str(type(A)):
                return A
                
            # 密行列に変換（最も安全な方法）
            if hasattr(A, 'toarray'):
                A_dense = A.toarray()
            else:
                A_dense = np.array(A)
            
            # 小さな値を切り捨て（数値的安定性のため）
            A_dense[np.abs(A_dense) < 1e-15] = 0.0
            
            # JAX配列に変換
            try:
                # 巨大行列の場合、分割して変換
                if A_dense.size > 1e7:  # ~100MB以上
                    print("大規模行列を分割処理中...")
                    if len(A_dense.shape) == 2:
                        rows = []
                        chunk_size = min(1000, A_dense.shape[0])
                        for i in range(0, A_dense.shape[0], chunk_size):
                            end = min(i + chunk_size, A_dense.shape[0])
                            rows.append(self.jnp.array(A_dense[i:end]))
                        return self.jnp.vstack(rows)
                    else:
                        return self.jnp.array(A_dense)
                else:
                    return self.jnp.array(A_dense)
            except Exception as e2:
                print(f"JAX配列変換エラー: {e2}")
                # JITコンパイルの問題がある場合は非JIT関数を使用
                with self.jax.disable_jit():
                    return self.jnp.array(A_dense)
                
        except Exception as e:
            print(f"JAX行列変換エラー: {e}")
            print(f"行列タイプ: {type(A)}, 形状: {A.shape if hasattr(A, 'shape') else 'unknown'}")
            print("CPUソルバーにフォールバックします")
            self.has_jax = False
            return A
    
    def _to_jax_vector(self, b):
        """ベクトルをJAX配列に変換（堅牢化）"""
        try:
            # すでにJAX配列の場合
            if 'jax' in str(type(b)):
                return b
                
            # NumPy/CuPy配列からJAX配列に変換
            if hasattr(b, 'get'):  # CuPy
                return self.jnp.array(b.get())
            else:
                return self.jnp.array(b)
        except Exception as e:
            print(f"JAX変換エラー: {e}")
            self.has_jax = False
            return b
    
    def _prepare_scaling(self):
        """スケーリング前処理（堅牢化）"""
        if not self.scaler or not self.has_jax:
            return
            
        try:
            # NumPy用ダミーベクトルでスケーリング情報を計算
            dummy_b = np.ones(self.A.shape[0])
            
            # NumPy版の行列を作成
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
            print("スケーリングを無効化します")
            self.scaler = None
    
    def solve(self, b, method=None, options=None):
        """JAXを使用して線形方程式系を解く（堅牢化）"""
        # JAXが使えない場合はCPUソルバーにフォールバック
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
        # メソッドとオプションを決定（引数で上書き可能）
        actual_method = method if method is not None else self.solver_method
        actual_options = self.solver_options.copy() if hasattr(self, 'solver_options') else {}
        if options:
            actual_options.update(options)
        
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
            if actual_method not in self.solvers:
                print(f"JAXで未対応の解法: {actual_method}、directに切り替えます")
                actual_method = "direct"
            
            # 線形システムを解く
            try:
                solver_func = self.solvers[actual_method]
                x_jax, iterations = solver_func(self.A, b_scaled, actual_options)
            except Exception as solver_error:
                print(f"JAXソルバーエラー: {solver_error}")
                print("非JITモードでリトライします...")
                with self.jax.disable_jit():
                    solver_func = self.solvers[actual_method]
                    x_jax, iterations = solver_func(self.A, b_scaled, actual_options)
            
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
                  
            return x
                
        except Exception as e:
            print(f"JAX解法エラー: {e}, CPUに切り替えます")
            # 動的にCPUソルバーを作成してフォールバック
            cpu_solver = CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            )
            if hasattr(self, 'solver_method'):
                cpu_solver.solver_method = self.solver_method
            if hasattr(self, 'solver_options'):
                cpu_solver.solver_options = self.solver_options
            
            return cpu_solver.solve(b, method, options)
    
    def _to_numpy_matrix(self, A):
        """JAX行列をNumPy形式に変換"""
        return np.array(A)
    
    def _to_numpy_vector(self, b):
        """JAXベクトルをNumPy形式に変換"""
        return np.array(b)
    
    def _to_numpy_scaling_info(self):
        """スケーリング情報をNumPy形式に変換"""
        numpy_info = {}
        for key, value in self.scaling_info.items():
            if 'jax' in str(type(value)):
                numpy_info[key] = np.array(value)
            else:
                numpy_info[key] = value
        return numpy_info
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        try:
            # JAXでの直接解法
            options = options or {}
            tol = options.get("tol", 1e-12)
            
            # JAXはsparse.linalg.solveを持っていない場合がある
            if hasattr(self.jax.scipy.linalg, 'solve'):
                x = self.jax.scipy.linalg.solve(A, b, assume_a='gen')
            else:
                # 代替として線形最小二乗法や反復法を使用
                print("JAX直接解法が利用できないため、CGにフォールバックします")
                x, _ = self._solve_cg(A, b, {"tol": tol})
            
            return x, None
        except Exception as e:
            print(f"JAX直接解法エラー: {e}, CPUにフォールバック")
            # CPUにフォールバック
            import scipy.sparse.linalg as splinalg
            x = splinalg.spsolve(self._to_numpy_matrix(A), self._to_numpy_vector(b))
            return self.jnp.array(x), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 20)
        
        # 初期推定値
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        try:
            # GMRES実行
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, 
                                         maxiter=maxiter, restart=restart)
            
            return result[0], result[1]  # x, iterations
        except Exception as e:
            print(f"JAX GMRES解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = self._to_numpy_vector(x0)
            result = splinalg.gmres(A_np, b_np, x0=x0_np, tol=tol, 
                                   maxiter=maxiter, restart=restart)
            return self.jnp.array(result[0]), result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        try:
            # CG実行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]  # x, iterations
        except Exception as e:
            print(f"JAX CG解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = self._to_numpy_vector(x0)
            result = splinalg.cg(A_np, b_np, x0=x0_np, tol=tol, maxiter=maxiter)
            return self.jnp.array(result[0]), result[1]
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        try:
            # BiCGSTAB実行
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]  # x, iterations
        except Exception as e:
            print(f"JAX BiCGSTAB解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = self._to_numpy_vector(x0)
            result = splinalg.bicgstab(A_np, b_np, x0=x0_np, tol=tol, maxiter=maxiter)
            return self.jnp.array(result[0]), result[1]