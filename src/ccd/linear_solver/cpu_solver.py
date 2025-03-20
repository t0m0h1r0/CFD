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
            "cg": self._solve_cg,
            "cgs": self._solve_cgs,
            "bicgstab": self._solve_bicgstab,
            "minres": self._solve_minres,
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
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def solve(self, b, method="direct", options=None):
        """SciPy を使用して線形方程式系を解く"""
        options = options or {}
        
        # 右辺ベクトルbをNumPy形式に変換
        b_np = self._to_numpy_vector(b)
        
        # スケーリングの適用
        b_scaled = b_np
        if self.scaler and self.scaling_info:
            try:
                b_scaled = self.scaler.scale_b_only(b_np, self.scaling_info)
            except Exception as e:
                print(f"スケーリングエラー: {e}")
        
        # 解法メソッドの選択
        if method not in self.solvers:
            print(f"未対応の解法: {method}、directに切り替えます")
            method = "direct"
        
        # 線形システムを解く
        try:
            solver_func = self.solvers[method]
            x, iterations = solver_func(self.A, b_scaled, options)
            
            # 結果のアンスケーリング
            if self.scaler and self.scaling_info:
                x = self.scaler.unscale(x, self.scaling_info)
                
            # 計算結果の記録
            self.last_iterations = iterations
                  
            return x
            
        except Exception as e:
            print(f"CPU解法エラー: {e}")
            # 直接解法にフォールバック
            x = splinalg.spsolve(self.A, b_scaled)
            if self.scaler and self.scaling_info:
                x = self.scaler.unscale(x, self.scaling_info)
            return x
    
    def _to_numpy_vector(self, b):
        """ベクトルをNumPy配列に変換"""
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
        result = splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # CG実行
        result = splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # BiCGSTAB実行
        result = splinalg.bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # CGS実行
        result = splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", np.zeros_like(b))
        
        # MINRES実行
        result = splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
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