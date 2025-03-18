"""
CPU (SciPy) を使用した線形方程式系ソルバー
"""

import os
import time
import numpy as np
import scipy.sparse.linalg as splinalg
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
            "cg": self._solve_iterative,
            "cgs": self._solve_iterative,
            "bicgstab": self._solve_iterative,
            "minres": self._solve_iterative,
            "lsqr": self._solve_least_squares,
            "lsmr": self._solve_least_squares
        }
    
    def _to_numpy_matrix(self, A):
        """行列をNumPy/SciPy形式に変換"""
        # CuPy配列からNumPy配列への変換
        if hasattr(A, 'get'):
            return A.get()
        
        # JAX配列からNumPy配列への変換
        if 'jax' in str(type(A)):
            return np.array(A)
        
        # 既にNumPy/SciPyの場合はそのまま
        return A
    
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
    
    def _prepare_scaling(self):
        """スケーリング前処理"""
        if not self.scaler:
            return
            
        # ダミーベクトルでスケーリング情報を計算
        dummy_b = np.ones(self.A.shape[0])
        
        # スケーリング情報を保存
        try:
            _, _, self.scaling_info = self.scaler.scale(self.A, dummy_b)
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def solve(self, b, method="direct", options=None):
        """SciPy を使用して線形方程式系を解く"""
        start_time = time.time()
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
            elapsed = time.time() - start_time
            print(f"CPU解法: {method}, 時間: {elapsed:.4f}秒" + 
                  (f", 反復: {iterations}" if iterations else ""))
                  
            return x
            
        except Exception as e:
            print(f"CPU解法エラー: {e}")
            # 直接解法にフォールバック
            x = splinalg.spsolve(self.A, b_scaled)
            if self.scaler and self.scaling_info:
                x = self.scaler.unscale(x, self.scaling_info)
            return x
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        return splinalg.spsolve(A, b), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 10000)
        
        # 初期推定値
        x0 = options.get("x0", np.ones_like(b))
        
        # コールバック関数設定
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
                residuals.append(float(residual))
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # GMRES実行
        result = splinalg.gmres(A, b, x0=x0, rtol=tol, maxiter=maxiter, 
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
        x0 = options.get("x0", np.ones_like(b))
        
        # コールバック関数設定
        residuals = []
        callback = None
        if options.get("monitor_convergence", False):
            def callback(xk):
                residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
                residuals.append(float(residual))
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        
        # 指定した反復解法を実行
        solver_func = getattr(splinalg, method_name)
        result = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        
        # 収束履歴を可視化（オプション）
        if options.get("monitor_convergence", False) and residuals:
            self._visualize_convergence(residuals, method_name, options)
        
        return result[0], result[1]  # x, iterations
    
    def _solve_least_squares(self, A, b, options=None):
        """最小二乗法ソルバー"""
        options = options or {}
        method_name = options.get("method_name", "lsqr")
        maxiter = options.get("maxiter", 1000)
        
        # 指定した最小二乗法解法を実行
        solver_func = getattr(splinalg, method_name)
        result = solver_func(A, b, iter_lim=maxiter)
        
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
        plt.title(f'{method_name.upper()} ソルバーの収束履歴')
        
        prefix = options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_{method_name}.png")
        plt.savefig(filename, dpi=150)
        plt.close()