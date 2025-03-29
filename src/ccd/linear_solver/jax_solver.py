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
            
            # スケーリングの初期化
            self._initialize_scaling()
            
            # 前処理器の初期セットアップ
            self.setup_preconditioner()
            
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
            self.enable_neumann, 
            self.scaling_method,
            self.preconditioner_name if self.preconditioner_name else self.preconditioner
        )
        # CPU版のsolversをコピー
        self.solvers = self.cpu_solver.solvers
        
        # 行列もCPU版に設定
        self.A = self.cpu_solver.A
        
        # CPUソルバーの前処理器を継承
        if hasattr(self.cpu_solver, 'preconditioner'):
            self.preconditioner = self.cpu_solver.preconditioner
    
    def solve(self, b, method=None, options=None):
        """
        Ax=b を解く（JAXが使えない場合はCPUソルバーにフォールバック）
        
        Args:
            b: 右辺ベクトル
            method: 解法メソッド名（設定済みのself.solver_methodを上書き）
            options: 解法オプション（設定済みのself.solver_optionsを上書き）
            
        Returns:
            解ベクトル x
        """
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
        # オプションがある場合、x0をJAX配列に変換
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
            
        # JAX初期化に失敗した場合に備えてtry-exceptをひとつだけ残す
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
            
    def _to_numpy_matrix(self, A):
        """JAX行列をNumPy形式に変換"""
        return np.array(A)
    
    def _to_numpy_vector(self, b):
        """JAXベクトルをNumPy形式に変換"""
        return np.array(b)
    
    def setup_preconditioner(self, A=None):
        """
        前処理行列をセットアップし、必要ならJAXに転送
        
        Args:
            A: 行列 (Noneの場合はself.Aを使用)
        """
        if not self.has_jax:
            if hasattr(self, 'cpu_solver'):
                self.cpu_solver.setup_preconditioner(A)
            return
            
        # 通常の前処理セットアップを実行
        super().setup_preconditioner(A)
        
        # 前処理行列をJAXに転送
        if self.preconditioner and hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
            self._convert_preconditioner_to_jax()
    
    def _convert_preconditioner_to_jax(self):
        """
        前処理行列をJAX形式に変換
        """
        if not self.has_jax or not hasattr(self.preconditioner, 'M') or self.preconditioner.M is None:
            return
            
        try:
            # 現在の前処理行列
            M = self.preconditioner.M
            
            # 行列形式の前処理
            if hasattr(M, 'toarray') or hasattr(M, 'todense'):
                # 疎行列をJAX用配列に変換
                M_cpu = M.toarray() if hasattr(M, 'toarray') else M.todense()
                M_jax = self.jnp.array(M_cpu)
                
                # 前処理器の行列を更新
                self.preconditioner.M = M_jax
                print(f"前処理行列をJAXに転送しました ({M_jax.shape})")
            
            # LinearOperator形式の前処理
            elif hasattr(M, 'matvec'):
                # JAX用のLinearOperatorラッパーを作成
                from jax.scipy.sparse.linalg import LinearOperator
                
                shape = M.shape
                
                # JAX対応のmatvec関数
                def jax_matvec(x):
                    if 'jax' in str(type(x)):
                        x_cpu = np.array(x)
                    else:
                        x_cpu = x
                    
                    # CPU上で前処理を適用
                    y_cpu = M.matvec(x_cpu)
                    
                    # 結果をJAXに変換
                    return self.jnp.array(y_cpu)
                
                # JAX対応のLinearOperatorを作成
                self.preconditioner.M = LinearOperator(shape, matvec=jax_matvec)
                print("LinearOperator形式の前処理をJAX対応に変換しました")
        except Exception as e:
            print(f"前処理行列のJAX変換エラー: {e}")
            # エラー時はCPU版の前処理を維持
    
    def _create_preconditioner_operator(self):
        """
        JAXでの前処理演算子を作成
        
        Returns:
            前処理演算子またはNone
        """
        if not self.has_jax or not self.preconditioner:
            return None
            
        try:
            # 前処理行列が既に設定されている場合
            if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
                # 行列としての前処理
                M = self.preconditioner.M
                
                # すでにJAX上にある場合はそのまま返す
                if 'jax' in str(type(M)):
                    return M
                
                # まだCPU上にある場合はJAXに変換
                if 'jax' not in str(type(M)):
                    print("CPU前処理行列をJAXに変換します")
                    self._convert_preconditioner_to_jax()
                    M = self.preconditioner.M
                    if 'jax' in str(type(M)):
                        return M
                
                # LinearOperatorの場合
                if hasattr(M, 'matvec'):
                    from jax.scipy.sparse.linalg import LinearOperator
                    
                    # JAX対応のmatvec
                    def jax_matvec(x):
                        try:
                            if 'jax' in str(type(x)):
                                x_cpu = np.array(x)
                            else:
                                x_cpu = x
                            
                            # CPU上で前処理を適用
                            y_cpu = M.matvec(x_cpu)
                            
                            # 結果をJAXに変換
                            return self.jnp.array(y_cpu)
                        except Exception as e:
                            print(f"前処理matvecエラー: {e}")
                            return x
                    
                    return LinearOperator(self.A.shape, matvec=jax_matvec)
            
            # __call__メソッドを持つカスタム前処理器
            elif hasattr(self.preconditioner, '__call__'):
                from jax.scipy.sparse.linalg import LinearOperator
                
                # JAX対応の前処理関数
                def preconditioner_func(x):
                    try:
                        # JAX配列をNumPyに変換
                        if 'jax' in str(type(x)):
                            x_cpu = np.array(x)
                        else:
                            x_cpu = x
                        
                        # CPU上で前処理を適用
                        y_cpu = self.preconditioner(x_cpu)
                        
                        # 結果をJAXに変換
                        return self.jnp.array(y_cpu)
                    except Exception as e:
                        print(f"前処理適用エラー: {e}")
                        return x
                
                return LinearOperator(self.A.shape, matvec=preconditioner_func)
        
        except Exception as e:
            print(f"JAX前処理演算子作成エラー: {e}")
        
        return None
        
    # JAX用ソルバーメソッド
    
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
        A_np = self._to_numpy_matrix(A)
        b_np = self._to_numpy_vector(b)
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
        
        # 前処理演算子の作成
        M = self._create_preconditioner_operator()
        
        try:
            # GMRES実行
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, restart=restart, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"GMRES実行エラー: {e}")
            print("前処理なしでGMRESを再試行します")
            # 前処理なしで再試行
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, restart=restart)
            return result[0], result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # 前処理演算子の作成
        M = self._create_preconditioner_operator()
        
        try:
            # CG実行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"CG実行エラー: {e}")
            print("前処理なしでCGを再試行します")
            # 前処理なしで再試行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # 前処理演算子の作成
        M = self._create_preconditioner_operator()
        
        try:
            # BiCGSTAB実行
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"BiCGSTAB実行エラー: {e}")
            print("前処理なしでBiCGSTABを再試行します")
            # 前処理なしで再試行
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
            return result[0], result[1]