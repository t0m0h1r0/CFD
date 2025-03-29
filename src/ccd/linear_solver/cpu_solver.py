"""
CPU (SciPy) を使用した線形方程式系ソルバー（リファクタリング版）
"""

import numpy as np
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg import LinearOperator
from .base import LinearSolver

class CPULinearSolver(LinearSolver):
    """CPU (SciPy) を使用した線形方程式系ソルバー"""
    
    def _initialize(self):
        """CPU固有の初期化処理"""
        # 行列をNumPy/SciPy形式に変換
        self.A = self._to_numpy_matrix(self.original_A)
        
        # スケーリングの初期化
        if self.scaling_method:
            from scaling import plugin_manager
            self.scaler = plugin_manager.get_plugin(self.scaling_method)
            self._prepare_scaling()
        
        # 解法メソッド辞書
        self.solvers = {
            "direct": self._solve_direct,
            "gmres": self._solve_gmres,
            "lgmres": self._solve_lgmres,
            "cg": self._solve_cg,
            "cgs": self._solve_cgs,
            "bicg": self._solve_bicg,
            "bicgstab": self._solve_bicgstab,
            "minres": self._solve_minres,
            "lsqr": self._solve_lsqr,
            "lsmr": self._solve_lsmr
        }
        
        # 前処理器の初期セットアップ
        self._setup_preconditioner()
    
    def _to_numpy_matrix(self, A):
        """行列をNumPy/SciPy形式に変換"""
        # CuPy配列からの変換
        if hasattr(A, 'get'):
            return A.get()
        
        # JAX配列からの変換
        if 'jax' in str(type(A)):
            return np.array(A)
            
        # 既にNumPy/SciPyの場合はそのまま
        return A
    
    def _prepare_scaling(self):
        """スケーリング前処理を設定"""
        if not self.scaler:
            return
            
        # ダミーベクトルでスケーリング情報を計算
        dummy_b = np.ones(self.A.shape[0])
        
        try:
            _, _, self.scaling_info = self.scaler.scale(self.A, dummy_b)
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def _setup_preconditioner(self):
        """前処理器を設定"""
        if self.preconditioner and hasattr(self.preconditioner, 'setup'):
            try:
                self.preconditioner.setup(self.A)
                print(f"前処理器をセットアップしました: {self.preconditioner.name}")
            except Exception as e:
                print(f"前処理器セットアップエラー: {e}")
    
    def _preprocess_vector(self, b):
        """ベクトルをNumPy形式に変換"""
        # CuPy配列からNumPy配列への変換
        if hasattr(b, 'get'):
            return b.get()
        
        # JAX配列からNumPy配列への変換
        if 'jax' in str(type(b)):
            return np.array(b)
        
        # 既にNumPy配列の場合はそのまま
        return b
    
    def _apply_scaling_to_b(self, b):
        """右辺ベクトルにスケーリングを適用"""
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
        if self.scaler and self.scaling_info:
            col_scale = self.scaling_info.get('col_scale')
            if col_scale is not None:
                return x / col_scale
            else:
                D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                if D_sqrt_inv is not None:
                    return x * D_sqrt_inv
        return x
    
    def _create_preconditioner_operator(self):
        """
        前処理演算子を作成
        
        Returns:
            前処理演算子またはNone
        """
        if not self.preconditioner:
            return None
            
        # 行列ベースの前処理
        if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
            return self.preconditioner.M
            
        # 関数ベースの前処理
        if hasattr(self.preconditioner, '__call__'):
            # 線形演算子として定義
            def precond_mv(v):
                return self.preconditioner(v)
                
            return LinearOperator(self.A.shape, matvec=precond_mv)
            
        return None
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        return splinalg.spsolve(A, b), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 200)
        x0 = options.get("x0", np.zeros_like(b))
        
        # 前処理演算子を取得
        M = self._create_preconditioner_operator()
        
        result = splinalg.gmres(A, b, x0=x0, rtol=tol, maxiter=maxiter, restart=restart, M=M)
        return result[0], result[1]
    
    def _solve_lgmres(self, A, b, options=None):
        """LGMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        inner_m = options.get("inner_m", 30)
        outer_k = options.get("outer_k", 3)
        x0 = options.get("x0", np.zeros_like(b))
        
        # 前処理演算子を取得
        M = self._create_preconditioner_operator()
        
        result = splinalg.lgmres(A, b, x0=x0, rtol=tol, maxiter=maxiter, 
                                inner_m=inner_m, outer_k=outer_k, M=M)
        return result[0], result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # 前処理演算子を取得
        M = self._create_preconditioner_operator()
        
        result = splinalg.cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_bicg(self, A, b, options=None):
        """BiCG法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # 前処理演算子を取得
        M = self._create_preconditioner_operator()
        
        result = splinalg.bicg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # 前処理演算子を取得
        M = self._create_preconditioner_operator()
        
        result = splinalg.bicgstab(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # 前処理演算子を取得
        M = self._create_preconditioner_operator()
        
        result = splinalg.cgs(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", np.zeros_like(b))
        
        # 前処理演算子を取得
        M = self._create_preconditioner_operator()
        
        result = splinalg.minres(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        return result[0], result[1]
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー"""
        options = options or {}
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        iter_lim = options.get("maxiter", options.get("iter_lim", None))
        
        # LSQRは前処理をサポートしていないことに注意
        result = splinalg.lsqr(A, b, damp=damp, atol=atol, btol=btol,
                             conlim=conlim, iter_lim=iter_lim)
        return result[0], result[2]
    
    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー"""
        options = options or {}
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        maxiter = options.get("maxiter", None)
        x0 = options.get("x0", None)
        
        # LSMRは前処理をサポートしていないことに注意
        result = splinalg.lsmr(A, b, damp=damp, atol=atol, btol=btol,
                             conlim=conlim, maxiter=maxiter, x0=x0)
        return result[0], result[2]