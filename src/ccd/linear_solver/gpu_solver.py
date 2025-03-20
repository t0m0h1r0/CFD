"""
GPU (CuPy) を使用した線形方程式系ソルバー
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
            self.cpu_solver = CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            )
    
    def _to_cupy_matrix(self, A):
        """行列をCuPy形式に変換"""
        try:
            # 既にCuPy上にある場合
            if hasattr(A, 'device') and str(type(A)).find('cupy') >= 0:
                return A
            
            # SciPy疎行列からの変換
            if hasattr(A, 'tocsr'):
                A = A.tocsr()
                
            if hasattr(A, 'data') and hasattr(A, 'indices') and hasattr(A, 'indptr'):
                # CSR形式の疎行列として変換
                return self.cp.sparse.csr_matrix(
                    (self.cp.array(A.data), self.cp.array(A.indices), self.cp.array(A.indptr)),
                    shape=A.shape
                )
            elif hasattr(A, 'toarray'):
                # 一般の疎行列
                return self.cp.sparse.csr_matrix(A.toarray())
            else:
                # 密行列
                return self.cp.sparse.csr_matrix(A)
        except Exception as e:
            print(f"GPU行列変換エラー: {e}")
            return A
    
    def _to_cupy_vector(self, b):
        """ベクトルをCuPy配列に変換"""
        try:
            # 既にCuPy配列の場合
            if hasattr(b, 'device') and str(type(b)).find('cupy') >= 0:
                return b
            
            # NumPy配列からCuPy配列に変換
            return self.cp.array(b)
        except Exception as e:
            print(f"GPU変換エラー: {e}")
            return b
    
    def _prepare_scaling(self):
        """スケーリング前処理"""
        if not self.scaler or not self.has_cupy:
            return
            
        # ダミーベクトルでスケーリング情報を計算
        dummy_b = np.ones(self.A.shape[0])
        
        try:
            # NumPy版の行列を作成
            A_np = self._to_numpy_matrix(self.A)
            # NumPyでスケーリング情報を計算
            _, _, scale_info_np = self.scaler.scale(A_np, dummy_b)
            
            # スケーリング情報をCuPyに変換
            self.scaling_info = {}
            for key, value in scale_info_np.items():
                if isinstance(value, np.ndarray):
                    self.scaling_info[key] = self.cp.array(value)
                else:
                    self.scaling_info[key] = value
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def solve(self, b, method="direct", options=None):
        """CuPyを使用して線形方程式系を解く"""
        # CuPyが使えない場合はCPUソルバーにフォールバック
        if not self.has_cupy:
            return self.cpu_solver.solve(b, method, options)
        
        options = options or {}
        
        try:
            # 右辺ベクトルbをCuPy形式に変換
            b_gpu = self._to_cupy_vector(b)
            
            # スケーリングの適用
            b_scaled = b_gpu
            if self.scaler and self.scaling_info:
                try:
                    row_scale = self.scaling_info.get('row_scale')
                    if row_scale is not None:
                        b_scaled = b_gpu * row_scale
                    else:
                        D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                        if D_sqrt_inv is not None:
                            b_scaled = b_gpu * D_sqrt_inv
                except Exception as e:
                    print(f"スケーリングエラー: {e}")
            
            # 解法メソッドの選択
            if method not in self.solvers:
                print(f"未対応の解法: {method}、directに切り替えます")
                method = "direct"
            
            # 線形システムを解く
            solver_func = self.solvers[method]
            x_gpu, iterations = solver_func(self.A, b_scaled, options)
            
            # 結果のアンスケーリング
            if self.scaler and self.scaling_info:
                try:
                    col_scale = self.scaling_info.get('col_scale')
                    if col_scale is not None:
                        x_gpu = x_gpu / col_scale
                    else:
                        D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                        if D_sqrt_inv is not None:
                            x_gpu = x_gpu * D_sqrt_inv
                except Exception as e:
                    print(f"アンスケーリングエラー: {e}")
                
            # GPU結果をCPUに転送
            x = x_gpu.get()
            
            # 計算結果の記録
            self.last_iterations = iterations
                  
            return x
                
        except Exception as e:
            print(f"GPU解法エラー: {e}, CPUに切り替えます")
            return CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            ).solve(b, method, options)
    
    def _to_numpy_matrix(self, A):
        """行列をNumPy形式に変換"""
        if hasattr(A, 'get'):
            return A.get()
        return A
    
    def _to_numpy_vector(self, b):
        """ベクトルをNumPy形式に変換"""
        if hasattr(b, 'get'):
            return b.get()
        return b
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        try:
            x = self.splinalg.spsolve(A, b)
            return x, None
        except Exception as e:
            print(f"GPU直接解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            x = splinalg.spsolve(self._to_numpy_matrix(A), self._to_numpy_vector(b))
            return self.cp.array(x), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 20)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # GMRES実行
        result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # CG実行
        result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # CGS実行
        result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # MINRES実行
        result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー"""
        try:
            result = self.splinalg.lsqr(A, b)
            return result[0], None
        except Exception as e:
            print(f"GPU LSQR解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            result = splinalg.lsqr(A_np, b_np)
            return self.cp.array(result[0]), None
    
    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー"""
        options = options or {}
        maxiter = options.get("maxiter", None)
        
        try:
            result = self.splinalg.lsmr(A, b, maxiter=maxiter)
            return result[0], result[2]  # x, iterations
        except Exception as e:
            print(f"GPU LSMR解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            result = splinalg.lsmr(A_np, b_np, maxiter=maxiter)
            return self.cp.array(result[0]), result[2]