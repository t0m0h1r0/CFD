"""
CPU (SciPy) を使用した線形方程式系ソルバー
"""

import numpy as np
import scipy.sparse.linalg as splinalg
from .base import LinearSolver

class CPULinearSolver(LinearSolver):
    """CPU (SciPy) を使用した線形方程式系ソルバー"""
    
    def _initialize(self):
        """CPU固有の初期化処理"""
        # 行列をNumPy/SciPy形式に変換
        self.A = self._ensure_scipy_matrix(self.original_A)
        
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
            "qmr": self._solve_qmr,
            "tfqmr": self._solve_tfqmr,
            "minres": self._solve_minres,
            "gcrotmk": self._solve_gcrotmk,
            "lsqr": self._solve_lsqr,
            "lsmr": self._solve_lsmr
        }
    
    def _ensure_scipy_matrix(self, A):
        """行列をSciPy形式に変換"""
        # CuPy配列からの変換
        if hasattr(A, 'get'):
            return A.get()
        
        # JAX配列からの変換
        if 'jax' in str(type(A)):
            return np.array(A)
            
        # 既にNumPy/SciPyの場合はそのまま
        return A
    
    def _prepare_scaling(self):
        """スケーリング前処理"""
        if not self.scaler:
            return
            
        # ダミーベクトルでスケーリング情報を計算
        dummy_b = np.ones(self.A.shape[0])
        
        try:
            _, _, self.scaling_info = self.scaler.scale(self.A, dummy_b)
        except Exception as e:
            print(f"Scaling preprocessing error: {e}")
            self.scaler = None
    
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
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        return splinalg.spsolve(A, b), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 20)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # GMRES実行
        result = splinalg.gmres(A, b, x0=x0, rtol=tol, maxiter=maxiter, restart=restart)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_lgmres(self, A, b, options=None):
        """LGMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        inner_m = options.get("inner_m", 30)
        outer_k = options.get("outer_k", 3)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # LGMRES実行
        result = splinalg.lgmres(A, b, x0=x0, rtol=tol, maxiter=maxiter, 
                                inner_m=inner_m, outer_k=outer_k)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # CG実行
        result = splinalg.cg(A, b, x0=x0, rtol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_bicg(self, A, b, options=None):
        """BiCG法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # BiCG実行
        result = splinalg.bicg(A, b, x0=x0, rtol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # BiCGSTAB実行
        result = splinalg.bicgstab(A, b, x0=x0, rtol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # CGS実行
        result = splinalg.cgs(A, b, x0=x0, rtol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_qmr(self, A, b, options=None):
        """QMR法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # QMR実行
        result = splinalg.qmr(A, b, x0=x0, rtol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_tfqmr(self, A, b, options=None):
        """TFQMR法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # TFQMR実行
        result = splinalg.tfqmr(A, b, x0=x0, rtol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # MINRES実行
        result = splinalg.minres(A, b, x0=x0, rtol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_gcrotmk(self, A, b, options=None):
        """GCROT(m,k)法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        m = options.get("m", 20)
        k = options.get("k", 10)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # GCROT(m,k)実行
        result = splinalg.gcrotmk(A, b, x0=x0, rtol=tol, maxiter=maxiter, m=m, k=k)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー"""
        options = options or {}
        
        # LSQRパラメータを抽出
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        iter_lim = options.get("maxiter", options.get("iter_lim", None))
        
        # LSQR実行
        result = splinalg.lsqr(A, b, damp=damp, atol=atol, btol=btol,
                             conlim=conlim, iter_lim=iter_lim)
        
        return result[0], result[2]  # x, iterations (itn)

    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー"""
        options = options or {}
        
        # LSMRパラメータを抽出
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        maxiter = options.get("maxiter", None)
        x0 = options.get("x0", None)
        
        # LSMR実行
        result = splinalg.lsmr(A, b, damp=damp, atol=atol, btol=btol,
                             conlim=conlim, maxiter=maxiter, x0=x0)
        
        return result[0], result[2]  # x, iterations (itn)