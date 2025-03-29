"""
GPU (CuPy) を使用した線形方程式系ソルバー（簡潔版）
"""

import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver

class GPULinearSolver(LinearSolver):
    """GPU (CuPy) を使用した線形方程式系ソルバー"""
    
    def _initialize(self):
        """GPU固有の初期化処理"""
        try:
            import cupy as cp
            import cupyx.scipy.sparse.linalg as splinalg
            self.cp = cp
            self.splinalg = splinalg
            self.has_cupy = True
            
            # 行列をCuPy形式に変換
            self.A = self._to_cupy_matrix(self.original_A)
            
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
                "cgs": self._solve_cgs,
                "minres": self._solve_minres,
                "lsqr": self._solve_lsqr,
                "lsmr": self._solve_lsmr
            }
            
        except ImportError as e:
            print(f"警告: CuPyが利用できません: {e}")
            self.has_cupy = False
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
    
    def solve(self, b, method=None, options=None):
        """
        Ax=b を解く（CuPyが使えない場合はCPUソルバーにフォールバック）
        
        Args:
            b: 右辺ベクトル
            method: 解法メソッド名（設定済みのself.solver_methodを上書き）
            options: 解法オプション（設定済みのself.solver_optionsを上書き）
            
        Returns:
            解ベクトル x
        """
        if not self.has_cupy:
            return self.cpu_solver.solve(b, method, options)
        
        # オプションがある場合、x0をCuPy配列に変換
        if options and "x0" in options:
            # 元のオプションを変更しないようにコピー
            options = options.copy()
            try:
                # x0をCuPy配列に変換
                options["x0"] = self._preprocess_vector(options["x0"])
                if hasattr(options["x0"], "shape"):
                    print(f"CuPy x0を設定しました (shape: {options['x0'].shape})")
            except Exception as e:
                print(f"x0変換エラー: {e}")
                # 変換に失敗した場合は削除
                del options["x0"]
        
        # 前処理機能の設定
        if self.preconditioner and hasattr(self.preconditioner, 'setup') and not hasattr(self.preconditioner, 'M'):
            try:
                # GPU行列に対して前処理を設定
                self.preconditioner.setup(self.A)
                
                # GPU用の前処理演算子に変換（必要な場合）
                if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
                    if not hasattr(self.preconditioner.M, 'device'):
                        # CPU行列をGPUに変換
                        self.preconditioner.M = self._to_cupy_matrix(self.preconditioner.M)
            except Exception as e:
                print(f"GPU前処理設定エラー: {e}")
        
        # 通常の処理
        return super().solve(b, method, options)
    
    def _to_cupy_matrix(self, A):
        """行列をCuPy CSR形式に効率的に変換"""
        if not self.has_cupy:
            return A
            
        try:
            # すでにCuPy上にある場合は何もしない
            if hasattr(A, 'device') and str(type(A)).find('cupy') >= 0:
                return A
                
            # 疎行列変換（CSR形式が最適）
            if hasattr(A, 'format'):
                # CSR形式への変換
                if A.format != 'csr':
                    A = A.tocsr()
                
                # CSR行列を直接CuPyに転送
                return self.cp.sparse.csr_matrix(
                    (self.cp.array(A.data), 
                     self.cp.array(A.indices), 
                     self.cp.array(A.indptr)),
                    shape=A.shape
                )
            else:
                # 密行列の場合
                if hasattr(A, 'toarray'):
                    A = A.toarray()
                # 密行列をCSR形式に変換
                return self.cp.sparse.csr_matrix(self.cp.array(A))
            
        except Exception as e:
            print(f"GPU行列変換エラー: {e}")
            print("CPUソルバーにフォールバックします")
            self.has_cupy = False
            self._init_cpu_fallback()
            return A
    
    def _prepare_scaling(self):
        """
        スケーリング前処理（簡潔版）
        
        メモリ効率を維持しつつ、処理をシンプル化
        """
        if not self.scaler or not self.has_cupy:
            return
            
        try:
            # より単純なアプローチ：スケーリング係数のみを計算
            dummy_b = np.ones(self.A.shape[0])
            A_np = self._to_numpy_matrix(self.A)
            _, _, scale_info = self.scaler.scale(A_np, dummy_b)
            
            # スケーリング情報をCuPyに変換
            self.scaling_info = {}
            for key, value in scale_info.items():
                if isinstance(value, np.ndarray):
                    self.scaling_info[key] = self.cp.array(value)
                else:
                    self.scaling_info[key] = value
                    
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def _preprocess_vector(self, b):
        """ベクトルをCuPy配列に変換"""
        if not self.has_cupy:
            return b
            
        # 既にCuPy配列の場合
        if hasattr(b, 'device') and str(type(b)).find('cupy') >= 0:
            return b
        
        # NumPy配列やメモリビューからCuPy配列に変換
        try:
            # メモリビューやndarrayからの変換
            return self.cp.array(b)
        except Exception as e:
            print(f"ベクトル変換エラー: {e}")
            # CPUソルバーにフォールバック
            self.has_cupy = False
            self._init_cpu_fallback()
            return b
    
    def _apply_scaling_to_b(self, b):
        """右辺ベクトルにスケーリングを適用"""
        if not self.has_cupy:
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
        if not self.has_cupy:
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
        if not self.has_cupy:
            return self.cpu_solver._direct_fallback(A, b)
            
        # CuPy->NumPyに変換してCPUソルバーを使用
        import scipy.sparse.linalg as splinalg
        A_np = self._to_numpy_matrix(A)
        b_np = self._to_numpy_vector(b)
        x = splinalg.spsolve(A_np, b_np)
        return self.cp.array(x), None
    
    def _to_numpy_matrix(self, A):
        """行列をNumPy形式に変換"""
        if hasattr(A, 'get'):
            # CuPyの疎行列の場合
            if hasattr(A, 'format') and A.format == 'csr':
                # CSR行列の各コンポーネントをNumPyに転送
                from scipy import sparse
                return sparse.csr_matrix(
                    (A.data.get(), A.indices.get(), A.indptr.get()),
                    shape=A.shape
                )
            else:
                # その他の場合は.get()を使用
                return A.get()
        return A
    
    def _to_numpy_vector(self, b):
        """ベクトルをNumPy形式に変換"""
        if hasattr(b, 'get'):
            return b.get()
        return b
    
    # GPUでの前処理適用関数
    def _create_preconditioner_operator(self):
        """
        GPUでの前処理演算子を作成
        
        Returns:
            前処理演算子またはNone
        """
        if not self.has_cupy or not self.preconditioner:
            return None
            
        try:
            # 前処理器の実体がある場合
            if hasattr(self.preconditioner, 'matrix') and self.preconditioner.matrix is not None:
                # 行列としての前処理
                precond_matrix = self.preconditioner.matrix
                
                # GPU対応のLinearOperatorを作成
                from cupyx.scipy.sparse.linalg import LinearOperator
                
                # matvecメソッドの定義
                def preconditioner_matvec(x):
                    if hasattr(precond_matrix, '__matmul__'):
                        return precond_matrix @ x
                    elif hasattr(precond_matrix, 'dot'):
                        return precond_matrix.dot(x)
                    else:
                        # フォールバック
                        return x
                
                return LinearOperator(self.A.shape, matvec=preconditioner_matvec)
            
            # __call__メソッドを持つカスタム前処理器
            elif hasattr(self.preconditioner, '__call__'):
                from cupyx.scipy.sparse.linalg import LinearOperator
                
                # 前処理関数を定義
                def preconditioner_func(x):
                    try:
                        return self.preconditioner(x)
                    except Exception as e:
                        print(f"前処理適用エラー: {e}")
                        return x
                
                return LinearOperator(self.A.shape, matvec=preconditioner_func)
        
        except Exception as e:
            print(f"GPU前処理演算子作成エラー: {e}")
        
        return None
    
    # 各ソルバーメソッドに前処理を追加
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        x = self.splinalg.spsolve(A, b)
        return x, None
    
    def _solve_gmres(self, A, b, options=None):
        """
        GPU上でのGMRES解法の効率的な実装
        
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
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(200, max(20, b.size // 10)))
        
        # 初期解ベクトルの取得
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # 前処理演算子の作成
        M = self._create_preconditioner_operator()
        
        # GMRES実行（オプションを最適化）
        result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, M=M)
        return result[0], result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解ベクトル
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # 前処理演算子の作成
        M = self._create_preconditioner_operator()
        
        # CG実行
        result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解ベクトル
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # 前処理演算子の作成
        M = self._create_preconditioner_operator()
        
        # CGS実行
        result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解ベクトル
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # 前処理演算子の作成
        M = self._create_preconditioner_operator()
        
        # MINRES実行
        result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー"""
        options = options or {}
        options.get("tol", 1e-10)
        options.get("maxiter", 1000)
        
        # CuPy cupyx.scipy.sparse.linalg.lsqr は前処理をサポートしていない
        result = self.splinalg.lsqr(A, b)
        return result[0], result[2]
    
    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # LSMRは前処理をサポートしていない
        result = self.splinalg.lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
        return result[0], result[2]