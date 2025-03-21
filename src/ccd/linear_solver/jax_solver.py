"""
JAX を使用した線形方程式系ソルバー（リファクタリング版）
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
            self._init_cpu_fallback()
    
    def _init_cpu_fallback(self):
        """CPUソルバーにフォールバック"""
        self.cpu_solver = CPULinearSolver(
            self.original_A, 
            self.enable_dirichlet, 
            self.enable_neumann, 
            self.scaling_method
        )
        # CPU版のsolversをコピー
        self.solvers = self.cpu_solver.solvers
    
    def solve(self, b, method=None, options=None):
        """JAXが使えない場合はCPUソルバーにフォールバック"""
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
        # 通常の処理
        return super().solve(b, method, options)
    
    def _to_jax_matrix(self, A):
        """行列をJAX形式に変換"""
        if not self.has_jax:
            return A
            
        # JAX初期化に失敗した場合に備えてtry-exceptをひとつだけ残す
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
            
            # 大規模行列の場合は分割処理
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
                
        except Exception as e:
            print(f"JAX行列変換エラー: {e}")
            print("CPUソルバーにフォールバックします")
            self.has_jax = False
            self._init_cpu_fallback()
            return A
    
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
    
    def _preprocess_vector(self, b):
        """ベクトルをJAX配列に変換"""
        if not self.has_jax:
            return b
            
        # すでにJAX配列の場合
        if 'jax' in str(type(b)):
            return b
            
        # NumPy/CuPy配列からJAX配列に変換
        if hasattr(b, 'get'):  # CuPy
            return self.jnp.array(b.get())
        else:
            return self.jnp.array(b)
    
    def _apply_scaling_to_b(self, b):
        """右辺ベクトルにスケーリングを適用"""
        if not self.has_jax:
            return self.cpu_solver._apply_scaling_to_b(b)
            
        if self.scaler and self.scaling_info:
            row_scale = self.scaling_info.get('row_scale')
            if row_scale is not None:
                return b * row_scale
            else:
                D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                if D_sqrt_inv is not None:
                    return b * D_sqrt_inv
        return b
    
    def _apply_unscaling_to_x(self, x):
        """解ベクトルにアンスケーリングを適用"""
        if not self.has_jax:
            return self.cpu_solver._apply_unscaling_to_x(x)
            
        if self.scaler and self.scaling_info:
            col_scale = self.scaling_info.get('col_scale')
            if col_scale is not None:
                return x / col_scale
            else:
                D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                if D_sqrt_inv is not None:
                    return x * D_sqrt_inv
        return x
    
    def _direct_fallback(self, A, b):
        """CPUソルバーを使用した直接解法フォールバック"""
        if not self.has_jax:
            return self.cpu_solver._direct_fallback(A, b)
            
        # JAX->NumPyに変換してCPUソルバーを使用
        import scipy.sparse.linalg as splinalg
        A_np = self._to_numpy_matrix(A)
        b_np = self._to_numpy_vector(b)
        x = splinalg.spsolve(A_np, b_np)
        return self.jnp.array(x), None
    
    def _to_numpy_matrix(self, A):
        """JAX行列をNumPy形式に変換"""
        return np.array(A)
    
    def _to_numpy_vector(self, b):
        """JAXベクトルをNumPy形式に変換"""
        return np.array(b)
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        # JAXでの直接解法の実装がない場合はCG法を使用
        options = options or {}
        tol = options.get("tol", 1e-12)
        
        # JAXはsparse.linalg.solveを持っていない場合がある
        if hasattr(self.jax.scipy.linalg, 'solve'):
            x = self.jax.scipy.linalg.solve(A, b, assume_a='gen')
            return x, None
        else:
            # 代替として反復法を使用
            print("JAX直接解法が利用できないため、CGにフォールバックします")
            return self._solve_cg(A, b, {"tol": tol})
    
    def _solve_gmres(self, A, b, options=None):
        """
        JAX上でのGMRES解法の効率的な実装
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            options: 解法オプション
        
        Returns:
            解ベクトルと反復回数のタプル
        """
        options = options or {}
        
        # オプションから設定を取得（デフォルト値を最適化）
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(20, max(5, b.size // 20)))
        
        # 初期解ベクトルの安全な処理
        x0 = options.get("x0")
        x0 = (self.jnp.asarray(x0) if x0 is not None else 
            self.jnp.zeros_like(b) if hasattr(b, 'size') and hasattr(x0, 'size') and x0.size != b.size else 
            self.jnp.zeros_like(b))
        
        # GMRES実行（オプションを最適化）
        result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, restart=restart)
        return result[0], result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # CG実行
        result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
        return result[0], result[1]
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # BiCGSTAB実行
        result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
        return result[0], result[1]