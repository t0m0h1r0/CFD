"""
GPU (CuPy) を使用した線形方程式系ソルバー
"""

import os
import time
import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver

class GPULinearSolver(LinearSolver):
    """GPU (CuPy) を使用した線形方程式系ソルバー"""
    
    def _initialize(self):
        """GPU固有の初期化処理"""
        # CuPyが利用可能か確認
        try:
            import cupy as cp
            import cupyx.scipy.sparse.linalg as splinalg
            self.cp = cp
            self.splinalg = splinalg
            self.has_cupy = True
            
            # 解法メソッド辞書 - 必ず最初に定義（エラー回避のため）
            self.solvers = {
                "direct": self._solve_direct,
                "gmres": self._solve_gmres,
                "cg": self._solve_iterative,
                "cgs": self._solve_iterative,
                "bicgstab": self._solve_iterative,
                "minres": self._solve_iterative,
                "lsqr": self._solve_least_squares,
                "lsmr": self._solve_least_squares
            }
            
            # 行列をCuPy形式に変換
            self.A = self._to_cupy_matrix(self.original_A)
            
            # スケーリングの初期化
            if self.scaling_method:
                from scaling import plugin_manager
                self.scaler = plugin_manager.get_plugin(self.scaling_method)
                self._prepare_scaling()
            
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
            
        # NumPy用ダミーベクトルでスケーリング情報を計算
        dummy_b = np.ones(self.A.shape[0])
        
        # スケーリング情報を保存
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
        
        start_time = time.time()
        options = options or {}
        
        try:
            # 右辺ベクトルbをCuPy形式に変換
            b_gpu = self._to_cupy_vector(b)
            
            # スケーリングの適用
            b_scaled = b_gpu
            if self.scaler and self.scaling_info:
                try:
                    # 右辺ベクトルのスケーリング - スケーラーのscale_b_onlyメソッドを使用
                    b_scaled = self.scaler.scale_b_only(b_gpu, self.scaling_info)
                except Exception as e:
                    print(f"スケーリングエラー: {e}")
            
            # 解法メソッドの選択
            if method not in self.solvers:
                print(f"未対応の解法: {method}、directに切り替えます")
                method = "direct"
            
            # 線形システムを解く
            solver_func = self.solvers[method]
            x_gpu, iterations = solver_func(self.A, b_scaled, options)
            
            # 結果のアンスケーリング - 修正: スケーラーのunscaleメソッドを使用
            if self.scaler and self.scaling_info:
                try:
                    x_gpu = self.scaler.unscale(x_gpu, self.scaling_info)
                except Exception as e:
                    print(f"アンスケーリングエラー: {e}")
                
            # GPU結果をCPUに転送
            x = x_gpu.get()
            
            # 計算結果の記録
            self.last_iterations = iterations
            elapsed = time.time() - start_time
            print(f"GPU解法: {method}, 時間: {elapsed:.4f}秒" + 
                  (f", 反復: {iterations}" if iterations else ""))
                  
            return x
                
        except Exception as e:
            print(f"GPU解法エラー: {e}, CPUに切り替えます")
            # CPUソルバーにフォールバック
            return CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            ).solve(b, method, options)
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        try:
            return self.splinalg.spsolve(A, b), None
        except Exception as e:
            print(f"GPU直接解法エラー: {e}, CPUにフォールバック")
            # CPUにフォールバック
            import scipy.sparse.linalg as splinalg
            x = splinalg.spsolve(self._to_numpy_matrix(A), self._to_numpy_vector(b))
            return self.cp.array(x), None
    
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
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 10000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.ones_like(b))
        
        # コールバック関数設定
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = float(self.cp.linalg.norm(b - A @ xk) / self.cp.linalg.norm(b))
                residuals.append(residual)
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # GMRES実行
        result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                    restart=restart, callback=callback)
        
        # 収束履歴を可視化（オプション）
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, "gmres", options)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_iterative(self, A, b, options=None):
        """反復解法共通インターフェース"""
        options = options or {}
        method_name = options.get("method_name", "cg")
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期推定値
        x0 = options.get("x0", self.cp.ones_like(b))
        
        # コールバック関数設定
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = float(self.cp.linalg.norm(b - A @ xk) / self.cp.linalg.norm(b))
                residuals.append(residual)
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # 指定した反復解法を実行
        solver_func = getattr(self.splinalg, method_name)
        result = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        
        # 収束履歴を可視化（オプション）
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, method_name, options)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_least_squares(self, A, b, options=None):
        """最小二乗法ソルバー"""
        options = options or {}
        method_name = options.get("method_name", "lsqr")
        
        # 最小二乗法解法を実行
        solver_func = getattr(self.splinalg, method_name)
        result = solver_func(A, b)
        
        return result[0], None  # x, iterations
    
    def _visualize_convergence(self, residuals, method_name, options):
        """収束履歴を可視化"""
        output_dir = options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals)+1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{method_name.upper()} ソルバーの収束履歴 (GPU)')
        
        prefix = options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_gpu_{method_name}.png")
        plt.savefig(filename, dpi=150)
        plt.close()