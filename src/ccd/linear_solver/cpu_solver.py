"""
CPU (SciPy) を使用した線形方程式系ソルバー（改良版）
"""

import os
import time
import numpy as np
import scipy.sparse.linalg as splinalg
from .base import LinearSolver

class CPULinearSolver(LinearSolver):
    """CPU (SciPy) を使用した線形方程式系ソルバー（改良版）"""
    
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
        """
        スケーリング前処理を改良
        実際の計算に使用する行列と右辺ベクトルの特性を反映
        """
        if not self.scaler:
            return
        
        try:
            # ダミーの右辺ベクトルではなく、適切な初期化
            dummy_b = np.ones(self.A.shape[0], dtype=float)
            
            # スケーリング情報を計算
            self.scaled_A, self.scaled_b, self.scaling_info = self.scaler.scale(self.A, dummy_b)
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
            self.scaled_A = self.A
            self.scaled_b = dummy_b
            self.scaling_info = {}
    
    def solve(self, b, method="direct", options=None):
        """
        線形方程式系を解く（改良版スケーリング処理）
        
        Args:
            b: 右辺ベクトル
            method: 解法メソッド
            options: オプション設定
        
        Returns:
            解ベクトル
        """
        start_time = time.time()
        options = options or {}
        
        # 入力ベクトルをNumPy形式に変換
        b_np = self._to_numpy_vector(b)
        
        # スケーリングの適用
        b_scaled = b_np
        if self.scaler and hasattr(self, 'scaling_info'):
            try:
                # scale_b_onlyを使用して右辺ベクトルをスケーリング
                b_scaled = self.scaler.scale_b_only(b_np, self.scaling_info)
            except Exception as e:
                print(f"スケーリングエラー: {e}")
                # スケーリングに失敗した場合、元の値を使用
                b_scaled = b_np
        
        # 解法メソッドの選択
        if method not in self.solvers:
            print(f"未対応の解法: {method}、directに切り替えます")
            method = "direct"
        
        try:
            # 線形システムを解く
            solver_func = self.solvers[method]
            x, iterations = solver_func(self.A, b_scaled, options)
            
            # アンスケーリング
            if self.scaler and hasattr(self, 'scaling_info'):
                try:
                    x = self.scaler.unscale(x, self.scaling_info)
                except Exception as e:
                    print(f"アンスケーリングエラー: {e}")
            
            # 計算結果の記録
            self.last_iterations = iterations
            elapsed = time.time() - start_time
            print(f"CPU解法: {method}, 時間: {elapsed:.4f}秒" + 
                  (f", 反復: {iterations}" if iterations else ""))
            
            return x
        
        except Exception as e:
            print(f"解法エラー: {e}")
            # フォールバック: 直接法
            try:
                x = splinalg.spsolve(self.A, b_scaled)
                
                # フォールバック時のアンスケーリング
                if self.scaler and hasattr(self, 'scaling_info'):
                    try:
                        x = self.scaler.unscale(x, self.scaling_info)
                    except Exception as scaling_error:
                        print(f"フォールバック時のアンスケーリングエラー: {scaling_error}")
                
                return x
            
            except Exception as fallback_error:
                print(f"フォールバック解法エラー: {fallback_error}")
                raise
    
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
    
    def debug_scaling(self, A=None, b=None):
        """
        スケーリングプロセスのデバッグ情報を出力
        
        Args:
            A: 行列（指定しない場合は初期化時の行列を使用）
            b: 右辺ベクトル（指定しない場合はダミーベクトルを使用）
        """
        # デフォルト値の設定
        A = A if A is not None else self.A
        b = b if b is not None else np.ones(A.shape[0], dtype=float)
        
        print("スケーリングデバッグ情報:")
        print("=" * 40)
        
        # 初期行列の特性
        print("元の行列の特性:")
        try:
            A_array = A.toarray() if hasattr(A, 'toarray') else A
            print(f"  行列形状: {A.shape}")
            print(f"  条件数: {np.linalg.cond(A_array)}")
            print(f"  行列のランク: {np.linalg.matrix_rank(A_array)}")
        except Exception as e:
            print(f"  行列特性の取得中にエラー: {e}")
        
        # スケーリングの実行
        if self.scaler:
            try:
                print(f"\nスケーリング手法: {self.scaler.name}")
                scaled_A, scaled_b, scaling_info = self.scaler.scale(A, b)
                
                # スケーリング後の特性
                scaled_A_array = scaled_A.toarray() if hasattr(scaled_A, 'toarray') else scaled_A
                print("\nスケーリング後の行列特性:")
                print(f"  スケーリング後の条件数: {np.linalg.cond(scaled_A_array)}")
                print(f"  スケーリング後の行列のランク: {np.linalg.matrix_rank(scaled_A_array)}")
                
                # スケーリング情報の詳細
                print("\nスケーリング情報:")
                for key, value in scaling_info.items():
                    print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"スケーリングの実行中にエラー: {e}")
        else:
            print("スケーリング手法が設定されていません。")