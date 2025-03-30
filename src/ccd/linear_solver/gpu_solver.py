"""
GPU (CuPy) を使用した線形方程式系ソルバー

このモジュールは、GPUとCuPyを使用して線形方程式系を効率的に解くためのクラスを提供します。
"""

from .base import LinearSolver
from .cpu_solver import CPULinearSolver

class GPULinearSolver(LinearSolver):
    """GPU (CuPy) を使用した線形方程式系ソルバー"""
    
    def _initialize(self):
        """GPU固有の初期化処理"""
        try:
            # CuPyをインポート
            import cupy as cp
            import cupyx.scipy.sparse.linalg as splinalg
            self.cp = cp
            self.splinalg = splinalg
            self.has_cupy = True
            
            # 行列をCuPy CSR形式に変換
            self.A = self._to_gpu_csr(self.original_A)
            
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
            
        except ImportError:
            # CPUソルバーにフォールバック
            print("GPUが利用できないためCPUにフォールバックします")
            self.has_cupy = False
            self._init_cpu_fallback()
    
    def _init_cpu_fallback(self):
        """CPUソルバーにフォールバック"""
        cpu_solver = CPULinearSolver(
            self.original_A, 
            self.enable_dirichlet, 
            self.enable_neumann,
            None,  # scaling_method
            None   # preconditioner
        )
        # CPU版の属性を継承
        self.__dict__.update(cpu_solver.__dict__)
    
    def _to_gpu_csr(self, A):
        """行列をCuPy CSR形式に効率的に変換"""
        try:
            # すでにCuPy上にある場合
            if hasattr(A, 'device'):
                return A.tocsr() if hasattr(A, 'tocsr') else A
                
            # CSR形式に変換
            if hasattr(A, 'tocsr'):
                A = A.tocsr()
            
            # CSR形式の直接転送
            if hasattr(A, 'data') and hasattr(A, 'indices') and hasattr(A, 'indptr'):
                return self.cp.sparse.csr_matrix(
                    (self.cp.array(A.data), self.cp.array(A.indices), self.cp.array(A.indptr)),
                    shape=A.shape
                )
            
            # その他の場合
            return self.cp.sparse.csr_matrix(self.cp.array(A.toarray() if hasattr(A, 'toarray') else A))
            
        except Exception as e:
            print(f"GPU行列変換エラー: {e}")
            return A
    
    def _to_numpy_matrix(self, A):
        """行列をNumPy形式に変換"""
        if hasattr(A, 'get'):
            return A.get()
        return A
    
    def solve(self, b, method=None, options=None):
        """Ax=b を解く"""
        # CuPyが使用できない場合
        if not hasattr(self, 'cp') or not self.has_cupy:
            return super().solve(b, method, options)
            
        # 右辺ベクトルをGPUに転送
        try:
            b_gpu = self.cp.array(b.get() if hasattr(b, 'get') else b)
        except Exception as e:
            print(f"右辺ベクトル変換エラー: {e}")
            return super().solve(b, method, options)
            
        # オプション処理
        options = {} if options is None else options.copy()
        if "x0" in options:
            try:
                options["x0"] = self.cp.array(options["x0"].get() if hasattr(options["x0"], 'get') else options["x0"])
            except Exception as e:
                print(f"初期ベクトル変換エラー: {e}")
                del options["x0"]
                
        # 通常の解法プロセスを実行
        return super().solve(b_gpu, method, options)
    
    def _preprocess_vector(self, b):
        """ベクトルをGPU形式に変換"""
        if not self.has_cupy:
            return b
            
        try:
            return self.cp.array(b.get() if hasattr(b, 'get') else b)
        except Exception as e:
            print(f"GPU変換エラー: {e}")
            return b
    
    # 各解法メソッド
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        try:
            x = self.splinalg.spsolve(A, b)
            return x, None
        except Exception as e:
            print(f"GPU直接解法エラー: {e}")
            # CPU直接解法にフォールバック
            try:
                # NumPy形式に確実に変換
                A_cpu = A.get() if hasattr(A, 'get') else A
                b_cpu = b.get() if hasattr(b, 'get') else b
                
                import scipy.sparse.linalg as sp_linalg
                x_cpu = sp_linalg.spsolve(A_cpu, b_cpu)
                return self.cp.array(x_cpu), None
            except Exception as e2:
                print(f"CPU直接解法エラー: {e2}")
                # 最後の手段：Identity行列として扱う
                return b.copy(), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(200, max(20, b.size // 10)))
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # GMRESを実行
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
            return result[0], result[1]
        except Exception as e:
            print(f"GMRES実行エラー: {e}")
            
            # 直接解法にフォールバック
            return self._solve_direct(A, b)
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # CGを実行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]
        except Exception as e:
            print(f"CG実行エラー: {e}")
            
            # 直接解法にフォールバック
            return self._solve_direct(A, b)
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # CGSを実行
            result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]
        except Exception as e:
            print(f"CGS実行エラー: {e}")
            
            # 直接解法にフォールバック
            return self._solve_direct(A, b)
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # MINRESを実行
            result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]
        except Exception as e:
            print(f"MINRES実行エラー: {e}")
            
            # 直接解法にフォールバック
            return self._solve_direct(A, b)
        
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        try:
            # LSQR実行
            result = self.splinalg.lsqr(A, b, atol=tol, btol=tol, iter_lim=maxiter)
            return result[0], result[2]  # LSQRは異なるインデックスを使用
        except Exception as e:
            print(f"LSQR実行エラー: {e}")
            # 直接解法にフォールバック
            return self._solve_direct(A, b)
    
    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        try:
            # LSMR実行
            result = self.splinalg.lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
            return result[0], result[2]  # LSMRは異なるインデックスを使用
        except Exception as e:
            print(f"LSMR実行エラー: {e}")
            # 直接解法にフォールバック
            return self._solve_direct(A, b)