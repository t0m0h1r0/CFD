"""
JAX を使用した線形方程式系ソルバー

このモジュールは、JAXを使用して線形方程式系を効率的に解くためのクラスを提供します。
JAXのJIT (Just-In-Time) コンパイルを活用し、GPUまたはTPUで高速に実行できます。
"""

import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver

class JAXLinearSolver(LinearSolver):
    """JAX を使用した線形方程式系ソルバー"""
    
    def _initialize(self):
        """JAX固有の初期化処理"""
        self.has_jax = False
        self.jax = None
        self.jnp = None
        self.splinalg = None
        
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
            
            # 解法メソッド辞書
            self.solvers = {
                "direct": self._solve_direct,
                "gmres": self._solve_gmres,
                "cg": self._solve_cg,
                "bicgstab": self._solve_bicgstab
            }
            
        except ImportError as e:
            print(f"警告: JAXが利用できません: {e}")
            self._init_cpu_fallback()
    
    def _init_cpu_fallback(self):
        """CPUソルバーにフォールバック"""
        self.cpu_solver = CPULinearSolver(
            self.original_A, 
            self.enable_dirichlet, 
            self.enable_neumann
        )
        # CPU版のsolversをコピー
        self.solvers = self.cpu_solver.solvers
        
        # 行列もCPU版に設定
        self.A = self.cpu_solver.A
    
    def solve(self, b, method=None, options=None):
        """Ax=b を解く"""
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
        # オプション処理
        if options and "x0" in options:
            # 元のオプションを変更しないようにコピー
            options = options.copy()
            try:
                # x0をJAX配列に変換
                options["x0"] = self._preprocess_vector(options["x0"])
                if hasattr(options["x0"], "shape"):
                    print(f"JAX x0を設定しました (shape: {options['x0'].shape})")
            except Exception as e:
                print(f"x0変換エラー: {e}")
                # 変換に失敗した場合は削除
                del options["x0"]
                
        return super().solve(b, method, options)
        
    def _preprocess_vector(self, b):
        """ベクトルをJAX配列に変換"""
        if not self.has_jax:
            return b
            
        # すでにJAX配列の場合
        if 'jax' in str(type(b)):
            return b
            
        try:
            # NumPy/CuPy配列からJAX配列に変換
            if hasattr(b, 'get'):  # CuPy
                return self.jnp.array(b.get())
            else:
                return self.jnp.array(b)
        except Exception as e:
            print(f"JAXベクトル変換エラー: {e}")
            # CPUソルバーにフォールバック
            self.has_jax = False
            self._init_cpu_fallback()
            return b
            
    def _to_jax_matrix(self, A):
        """行列をJAX形式に変換"""
        if not self.has_jax:
            return A
            
        try:
            # JAXですでに処理されているか確認
            if 'jax' in str(type(A)):
                return A
                
            # None の場合はそのまま返す
            if A is None:
                return None
                
            # LinearOperatorの場合
            if hasattr(A, 'matvec'):
                from jax.scipy.sparse.linalg import LinearOperator
                
                # JAX対応のmatvec関数
                def jax_matvec(x):
                    if 'jax' in str(type(x)):
                        x_cpu = np.array(x)
                    else:
                        x_cpu = x
                    
                    # CPU上でmatvecを実行
                    y_cpu = A.matvec(x_cpu)
                    
                    # 結果をJAXに変換
                    return self.jnp.array(y_cpu)
                
                return LinearOperator(A.shape, matvec=jax_matvec)
                
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
    
    # 解法メソッド
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        # JAXでの直接解法の実装がない場合はCG法を使用
        options = options or {}
        tol = options.get("tol", 1e-12)
        
        # JAXはsparse.linalg.solveを持っていない場合がある
        if hasattr(self.jax.scipy.linalg, 'solve'):
            try:
                x = self.jax.scipy.linalg.solve(A, b, assume_a='gen')
                return x, None
            except Exception as e:
                print(f"JAX直接解法エラー: {e}")
                print("代替としてCGにフォールバックします")
        else:
            print("JAX直接解法が利用できないため、CGにフォールバックします")
            
        # 代替として反復法を使用
        return self._solve_cg(A, b, {"tol": tol})
    
    def _direct_fallback(self, A, b):
        """CPUソルバーを使用した直接解法フォールバック"""
        print("CPU直接解法にフォールバックします")
        import scipy.sparse.linalg as splinalg
        A_np = np.array(A)
        b_np = np.array(b)
        x = splinalg.spsolve(A_np, b_np)
        return self.jnp.array(x), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(20, max(5, b.size // 20)))
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        try:
            # GMRES実行
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, restart=restart)
            return result[0], result[1]
        except Exception as e:
            print(f"GMRES実行エラー: {e}")
            print("GMRESを再試行します")
            # 再試行
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, restart=restart)
            return result[0], result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        try:
            # CG実行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]
        except Exception as e:
            print(f"CG実行エラー: {e}")
            print("CGを再試行します")
            # 再試行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        try:
            # BiCGSTAB実行
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]
        except Exception as e:
            print(f"BiCGSTAB実行エラー: {e}")
            print("BiCGSTABを再試行します")
            # 再試行
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]