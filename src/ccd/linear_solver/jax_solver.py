"""
JAX を使用した線形方程式系ソルバー
"""

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
            import jax.scipy.sparse.linalg as splinalg
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
                return A
                
            # 疎行列の場合は密行列に変換
            if hasattr(A, 'toarray'):
                A = A.toarray()
                
            # JAX配列に変換
            return self.jnp.array(A)
        except Exception as e:
            print(f"JAX行列変換エラー: {e}")
            return A
    
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
            self.scaler = None
    
    def solve(self, b, method="direct", options=None):
        """JAXを使用して線形方程式系を解く"""
        # JAXが使えない場合はCPUソルバーにフォールバック
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
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
            if hasattr(value, 'shape') and str(type(value)).find('jax') >= 0:
                numpy_info[key] = np.array(value)
            else:
                numpy_info[key] = value
        return numpy_info
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        try:
            # JAXでの直接解法
            x = self.jax.scipy.linalg.solve(A, b)
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
        atol = options.get("atol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 20)
        
        # 初期推定値
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # GMRES実行
        result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, 
                                     maxiter=maxiter, restart=restart)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # CG実行
        result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # BiCGSTAB実行
        result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations