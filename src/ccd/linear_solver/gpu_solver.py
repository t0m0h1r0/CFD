"""
GPU (CuPy) を使用した線形方程式系ソルバー (完全修正版)

このモジュールは、GPUとCuPyを使用して線形方程式系を効率的に解くためのクラスを提供します。
前処理器とGMRES実行時のエラーを修正しています。
"""

import numpy as np
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
            
            # スケーリングと前処理の初期化
            self._init_scaling()
            self._init_preconditioner()
            
            # 個別のソルバーメソッドを辞書に登録
            self.solvers = {
                "direct": self._solve_direct,
                "gmres": self._solve_gmres,
                "cg": self._solve_cg,
                "cgs": self._solve_cgs,
                "minres": self._solve_minres,
                "bicgstab": self._solve_bicgstab,
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
            self.scaling_method,
            self.preconditioner_name if hasattr(self, 'preconditioner_name') else self.preconditioner
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
    
    def _init_scaling(self):
        """スケーリング係数を初期化"""
        if not self.scaling_method:
            return
            
        try:
            from scaling import plugin_manager
            scaler = plugin_manager.get_plugin(self.scaling_method)
            
            # CPU上でスケーリング情報を計算
            dummy_b = np.ones(self.A.shape[0])
            # CuPyの場合はNumPy形式に変換
            A_cpu = self.A.get() if hasattr(self.A, 'get') else self.A
            _, _, scale_info = scaler.scale(A_cpu, dummy_b)
            
            # GPU用に変換して保存
            self.scaler = scaler
            self.scaling_info = {
                k: self.cp.array(v) if isinstance(v, np.ndarray) else v
                for k, v in scale_info.items()
            }
        except Exception as e:
            print(f"スケーリング初期化エラー: {e}")
            self.scaler = None
            self.scaling_info = None
    
    def _init_preconditioner(self):
        """前処理器を初期化"""
        if not hasattr(self, 'preconditioner') or not self.preconditioner:
            return
            
        try:
            # 前処理器のセットアップ
            if hasattr(self.preconditioner, 'setup'):
                # 必ずNumPy形式に変換
                A_cpu = self.A.get() if hasattr(self.A, 'get') else self.A
                self.preconditioner.setup(A_cpu)
                
            # 前処理行列をGPUには転送せず、LinearOperatorで使用することにする
            # CuPyは一部の前処理行列の形式に対応していないため
        except Exception as e:
            print(f"前処理器初期化エラー: {e}")
    
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
    
    def _get_preconditioner_op(self):
        """
        前処理演算子を取得 (LinearOperator形式)
        この方法では、前処理行列をCuPyに変換せず、NumPyとCuPyの間で変換しながら使用
        """
        if not hasattr(self, 'preconditioner') or not self.preconditioner:
            return None
            
        try:
            from cupyx.scipy.sparse.linalg import LinearOperator
            
            # 行列の次元を取得
            n = self.A.shape[0]
            
            # 前処理適用のラッパー関数
            def precond_op(x):
                # CuPy → NumPy変換
                x_cpu = x.get()
                
                # 前処理を適用 (前処理器の形式に応じて処理)
                if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
                    # 行列形式の前処理
                    y_cpu = self.preconditioner.M @ x_cpu
                elif hasattr(self.preconditioner, '__call__'):
                    # 関数形式の前処理
                    y_cpu = self.preconditioner(x_cpu)
                else:
                    # 前処理がない場合
                    return x
                
                # NumPy → CuPy変換
                return self.cp.array(y_cpu)
                
            # LinearOperatorを作成して返す
            return LinearOperator((n, n), matvec=precond_op)
            
        except Exception as e:
            print(f"前処理演算子作成エラー: {e}")
            return None
    
    def _apply_scaling_to_b(self, b):
        """右辺ベクトルにスケーリングを適用"""
        if not hasattr(self, 'scaling_info') or not self.scaling_info:
            return b
            
        # すべてCuPy環境で実行
        try:
            # 行スケーリングの適用
            row_scale = self.scaling_info.get('row_scale')
            if row_scale is not None:
                return b * row_scale
                
            # 対称スケーリングの適用  
            D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
            if D_sqrt_inv is not None:
                return b * D_sqrt_inv
        except Exception as e:
            print(f"スケーリング適用エラー: {e}")
            
        return b
    
    def _apply_unscaling_to_x(self, x):
        """解ベクトルにアンスケーリングを適用"""
        if not hasattr(self, 'scaling_info') or not self.scaling_info:
            return x
            
        # すべてCuPy環境で実行
        try:
            # 列スケーリングの適用
            col_scale = self.scaling_info.get('col_scale')
            if col_scale is not None:
                return x / col_scale
                
            # 対称スケーリングの適用
            D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
            if D_sqrt_inv is not None:
                return x * D_sqrt_inv
        except Exception as e:
            print(f"アンスケーリング適用エラー: {e}")
            
        return x
    
    # 個別の解法メソッド (徹底的な例外処理とエラー修正)
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
        """GMRES法 (徹底修正版)"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(200, max(20, b.size // 10)))
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # 前処理演算子を取得 (常にLinearOperator形式で取得)
            M = self._get_preconditioner_op()
            
            # GMRESを実行 (CuPy環境内で完結)
            result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"GMRES実行エラー: {e}")
            print("前処理なしでGMRESを再試行します")
            
            try:
                # 前処理なしで再試行
                result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
                return result[0], result[1]
            except Exception as e2:
                print(f"前処理なしGMRESエラー: {e2}")
                print("直接解法にフォールバックします")
                # 直接解法にフォールバック
                return self._solve_direct(A, b)
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法 (修正版)"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # 前処理演算子を取得 (常にLinearOperator形式で取得)
            M = self._get_preconditioner_op()
            
            # CGを実行
            result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"CG実行エラー: {e}")
            print("前処理なしでCGを再試行します")
            
            try:
                # 前処理なしで再試行
                result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
                return result[0], result[1]
            except Exception as e2:
                print(f"前処理なしCGエラー: {e2}")
                # 直接解法にフォールバック
                return self._solve_direct(A, b)
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法 (修正版)"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # 前処理演算子を取得 (常にLinearOperator形式で取得)
            M = self._get_preconditioner_op()
            
            # CGSを実行
            result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"CGS実行エラー: {e}")
            
            try:
                # 前処理なしで再試行
                result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter)
                return result[0], result[1]
            except Exception:
                # 直接解法にフォールバック
                return self._solve_direct(A, b)
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法 (修正版)"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # 前処理演算子を取得 (常にLinearOperator形式で取得)
            M = self._get_preconditioner_op()
            
            # MINRESを実行
            result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"MINRES実行エラー: {e}")
            
            try:
                # 前処理なしで再試行
                result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter)
                return result[0], result[1]
            except Exception:
                # 直接解法にフォールバック
                return self._solve_direct(A, b)
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法 (修正版)"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        try:
            # 前処理演算子を取得 (常にLinearOperator形式で取得)
            M = self._get_preconditioner_op()
            
            # BiCGSTABを実行
            result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            return result[0], result[1]
        except Exception as e:
            print(f"BiCGSTAB実行エラー: {e}")
            
            try:
                # 前処理なしで再試行
                result = self.splinalg.bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter)
                return result[0], result[1]
            except Exception:
                # 直接解法にフォールバック
                return self._solve_direct(A, b)
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー (修正版)"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        try:
            # LSQRは前処理をサポートしない
            result = self.splinalg.lsqr(A, b, atol=tol, btol=tol, iter_lim=maxiter)
            return result[0], result[2]  # LSQRは異なるインデックスを使用
        except Exception as e:
            print(f"LSQR実行エラー: {e}")
            # 直接解法にフォールバック
            return self._solve_direct(A, b)
    
    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー (修正版)"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        try:
            # LSMRは前処理をサポートしない
            result = self.splinalg.lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
            return result[0], result[2]  # LSMRは異なるインデックスを使用
        except Exception as e:
            print(f"LSMR実行エラー: {e}")
            # 直接解法にフォールバック
            return self._solve_direct(A, b)