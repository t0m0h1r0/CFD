"""
GPU (CuPy) を使用した線形方程式系ソルバー（改善版）
"""

import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver

class GPULinearSolver(LinearSolver):
    """GPU (CuPy) を使用した線形方程式系ソルバー（堅牢性を改善）"""
    
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
            # CPU版のsolversをコピー
            self.solvers = self.cpu_solver.solvers if hasattr(self.cpu_solver, 'solvers') else {}
    
    def _to_cupy_matrix(self, A):
        """行列をCuPy形式に変換（堅牢化）"""
        try:
            # 既にCuPy上にある場合
            if hasattr(A, 'device') and str(type(A)).find('cupy') >= 0:
                return A
            
            # Numpyの密行列に変換してからCuPyに転送（最も安全だが効率が悪い）
            if hasattr(A, 'toarray'):
                A_dense = A.toarray()
            else:
                A_dense = np.array(A)
            
            # 小さな値を切り捨て
            A_dense[np.abs(A_dense) < 1e-15] = 0.0
            
            # CuPyのCSR形式に変換
            return self.cp.sparse.csr_matrix(self.cp.array(A_dense))
            
        except Exception as e:
            print(f"GPU行列変換エラー: {e}")
            print("CPUソルバーにフォールバックします")
            self.has_cupy = False
            return A
    
    def _to_cupy_vector(self, b):
        """ベクトルをCuPy配列に変換（堅牢化）"""
        try:
            # 既にCuPy配列の場合
            if hasattr(b, 'device') and str(type(b)).find('cupy') >= 0:
                return b
            
            # NumPy配列からCuPy配列に変換
            return self.cp.array(b)
        except Exception as e:
            print(f"GPU変換エラー: {e}")
            self.has_cupy = False
            return b
    
    def _prepare_scaling(self):
        """スケーリング前処理（堅牢化）"""
        if not self.scaler or not self.has_cupy:
            return
            
        # ダミーベクトルでスケーリング情報を計算
        try:
            dummy_b = np.ones(self.A.shape[0])
            
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
            print("スケーリングを無効化します")
            self.scaler = None
    
    def solve(self, b, method=None, options=None):
        """CuPyを使用して線形方程式系を解く（堅牢化）"""
        # CuPyが使えない場合はCPUソルバーにフォールバック
        if not self.has_cupy:
            return self.cpu_solver.solve(b, method, options)
        
        # メソッドとオプションを決定（引数で上書き可能）
        actual_method = method if method is not None else self.solver_method
        actual_options = self.solver_options.copy() if hasattr(self, 'solver_options') else {}
        if options:
            actual_options.update(options)
        
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
            if actual_method not in self.solvers:
                print(f"未対応の解法: {actual_method}、directに切り替えます")
                actual_method = "direct"
            
            # 線形システムを解く
            solver_func = self.solvers[actual_method]
            x_gpu, iterations = solver_func(self.A, b_scaled, actual_options)
            
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
                
            # 計算結果の記録
            self.last_iterations = iterations
            
            # GPU結果をCPUに転送
            if hasattr(x_gpu, 'get'):
                x = x_gpu.get()
            else:
                x = np.array(x_gpu)
                  
            return x
                
        except Exception as e:
            print(f"GPU解法エラー: {e}, CPUに切り替えます")
            # 動的にCPUソルバーを作成してフォールバック
            cpu_solver = CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            )
            if hasattr(self, 'solver_method'):
                cpu_solver.solver_method = self.solver_method
            if hasattr(self, 'solver_options'):
                cpu_solver.solver_options = self.solver_options
            
            return cpu_solver.solve(b, method, options)
    
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
        
        try:
            # GMRES実行
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
            return result[0], result[1]  # x, iterations
        except Exception as e:
            print(f"GPU GMRES解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = self._to_numpy_vector(x0)
            result = splinalg.gmres(A_np, b_np, x0=x0_np, tol=tol, maxiter=maxiter, restart=restart)
            return self.cp.array(result[0]), result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # CG実行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]  # x, iterations
        except Exception as e:
            print(f"GPU CG解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = self._to_numpy_vector(x0)
            result = splinalg.cg(A_np, b_np, x0=x0_np, tol=tol, maxiter=maxiter)
            return self.cp.array(result[0]), result[1]
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # CGS実行
            result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]  # x, iterations
        except Exception as e:
            print(f"GPU CGS解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = self._to_numpy_vector(x0)
            result = splinalg.cgs(A_np, b_np, x0=x0_np, tol=tol, maxiter=maxiter)
            return self.cp.array(result[0]), result[1]
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # MINRES実行
            result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter)
            return result[0], result[1]  # x, iterations
        except Exception as e:
            print(f"GPU MINRES解法エラー: {e}, CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            x0_np = self._to_numpy_vector(x0)
            result = splinalg.minres(A_np, b_np, x0=x0_np, tol=tol, maxiter=maxiter)
            return self.cp.array(result[0]), result[1]


    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー（CuPy互換性対応版）"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        try:
            # CuPy cupyx.scipy.sparse.linalg.lsqr は引数が少なく、
            # dampなどの追加パラメータはサポートしていない
            # https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.lsqr.html
            result = self.splinalg.lsqr(A, b, tol=tol, iter_lim=maxiter)
            return result[0], result[2]  # x, iterations
        except TypeError as te:
            # 引数互換性のエラー（引数が多すぎる/少なすぎる/タイプが違う）
            print(f"CuPy LSQR引数エラー - ベーシック呼び出しへフォールバック: {te}")
            try:
                # 最小限の引数だけで再試行
                result = self.splinalg.lsqr(A, b)
                return result[0], result[2]
            except Exception:
                pass
        except Exception as e:
            print(f"GPU LSQR解法エラー: {e}")
        
        # CPUにフォールバック
        print("CPUにフォールバック")
        import scipy.sparse.linalg as splinalg
        A_np = self._to_numpy_matrix(A)
        b_np = self._to_numpy_vector(b)
        result = splinalg.lsqr(A_np, b_np, damp=options.get("damp", 0.0), 
                            atol=options.get("atol", tol), 
                            btol=options.get("btol", tol), 
                            iter_lim=maxiter)
        try:
            return self.cp.array(result[0]), result[2]
        except:
            return result[0], result[2]

    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー（CuPy互換性対応版）"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # CuPyのバージョンによってはlsmrが実装されていない場合がある
        if not hasattr(self.splinalg, 'lsmr'):
            print("CuPyにLSMRがないため、CPUにフォールバック")
            import scipy.sparse.linalg as splinalg
            A_np = self._to_numpy_matrix(A)
            b_np = self._to_numpy_vector(b)
            result = splinalg.lsmr(A_np, b_np, damp=options.get("damp", 0.0), 
                                atol=options.get("atol", tol), 
                                btol=options.get("btol", tol), 
                                maxiter=maxiter)
            try:
                return self.cp.array(result[0]), result[2]
            except:
                return result[0], result[2]
        
        try:
            # 引数互換性に注意
            result = self.splinalg.lsmr(A, b, tol=tol, maxiter=maxiter)
            return result[0], result[2]
        except TypeError as te:
            # 引数互換性のエラー
            print(f"CuPy LSMR引数エラー - ベーシック呼び出しへフォールバック: {te}")
            try:
                # 最小限の引数だけで再試行
                result = self.splinalg.lsmr(A, b)
                return result[0], result[2]
            except Exception:
                pass
        except Exception as e:
            print(f"GPU LSMR解法エラー: {e}")
        
        # CPUにフォールバック
        print("CPUにフォールバック")
        import scipy.sparse.linalg as splinalg
        A_np = self._to_numpy_matrix(A)
        b_np = self._to_numpy_vector(b)
        result = splinalg.lsmr(A_np, b_np, damp=options.get("damp", 0.0), 
                            atol=options.get("atol", tol), 
                            btol=options.get("btol", tol), 
                            maxiter=maxiter)
        try:
            return self.cp.array(result[0]), result[2]
        except:
            return result[0], result[2]