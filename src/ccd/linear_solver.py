"""
線形方程式系 Ax=b を解くためのソルバーモジュール

このモジュールはCPU (SciPy)、GPU (CuPy)、およびJAX実装の線形ソルバーを提供し、
スパース行列向けの様々な解法 (直接解法・反復解法) をサポートします。
"""

import os
import time
import numpy as np
import scipy.sparse.linalg as splinalg_cpu
from abc import ABC, abstractmethod

from scaling import plugin_manager


class LinearSolver(ABC):
    """線形方程式系 Ax=b を解くための抽象基底クラス"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        """
        線形方程式系ソルバーを初期化
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        self.method = method
        self.options = options or {}
        self.scaling_method = scaling_method
        self.last_iterations = None
        self.monitor_convergence = self.options.get("monitor_convergence", False)
    
    @abstractmethod
    def solve(self, A, b):
        """
        線形方程式系を解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            解ベクトル x
        """
        pass
    
    def _visualize_convergence(self, residuals):
        """収束履歴をグラフ化（必要な場合）"""
        if not residuals:
            return
            
        # 出力ディレクトリを確保
        output_dir = self.options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 残差の推移グラフ
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals)+1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{self.method.upper()} ソルバーの収束履歴')
        
        # 保存
        prefix = self.options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_{self.method}.png")
        plt.savefig(filename, dpi=150)
        plt.close()


class CPULinearSolver(LinearSolver):
    """CPU (SciPy) を使用した線形方程式系ソルバー"""
    
    def solve(self, A, b):
        """
        SciPy を使用して線形方程式系を解く
        
        Args:
            A: システム行列 (SciPy CSR形式)
            b: 右辺ベクトル (NumPy配列)
            
        Returns:
            解ベクトル x (NumPy配列)
        """
        # 開始時間を記録
        start_time = time.time()
        residuals = []
        
        if self.method == "direct" or self.method not in ["gmres", "cg", "cgs", "minres", "lsqr", "lsmr"]:
            x = splinalg_cpu.spsolve(A, b)
            iterations = None
        elif self.method == "gmres":
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            restart = self.options.get("restart", 100)
            x0 = np.ones_like(b)
            
            # 収束モニター
            if self.monitor_convergence:
                def callback(xk):
                    residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
                    residuals.append(float(residual))
                    if len(residuals) % 10 == 0:
                        print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
            else:
                callback = None
            
            # SciPyバージョンに対応するAPI呼び出し
            try:
                # 標準のパラメータで試行
                x, iterations = splinalg_cpu.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                                  restart=restart, callback=callback)
            except TypeError:
                # パラメータ互換性問題の場合、最小限のパラメータで再試行
                print("警告: SciPyのAPI互換性問題を検出しました。最小限のパラメータで再試行します。")
                x, iterations = splinalg_cpu.gmres(A, b, restart=restart)
        elif self.method in ["cg", "cgs"]:
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg_cpu, self.method)
            x0 = np.ones_like(b)
            
            # 収束モニター
            if self.monitor_convergence:
                def callback(xk):
                    residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
                    residuals.append(float(residual))
                    if len(residuals) % 10 == 0:
                        print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
            else:
                callback = None
            
            # SciPyバージョンに対応するAPI呼び出し
            try:
                x, iterations = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
            except TypeError:
                # パラメータ互換性問題の場合、最小限のパラメータで再試行
                print("警告: SciPyのAPI互換性問題を検出しました。最小限のパラメータで再試行します。")
                x, iterations = solver_func(A, b)
        elif self.method == "minres":
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            x0 = np.ones_like(b)
            
            # 収束モニター
            if self.monitor_convergence:
                def callback(xk):
                    residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
                    residuals.append(float(residual))
                    if len(residuals) % 10 == 0:
                        print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
            else:
                callback = None
            
            # SciPyバージョンに対応するAPI呼び出し
            try:
                x, iterations = splinalg_cpu.minres(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
            except TypeError:
                # パラメータ互換性問題の場合、最小限のパラメータで再試行
                print("警告: SciPyのAPI互換性問題を検出しました。最小限のパラメータで再試行します。")
                x, iterations = splinalg_cpu.minres(A, b)
        elif self.method in ["lsqr", "lsmr"]:
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg_cpu, self.method)
            
            # SciPyバージョンに対応するAPI呼び出し
            try:
                x = solver_func(A, b, iter_lim=maxiter)[0]
            except TypeError:
                # パラメータ互換性問題の場合、最小限のパラメータで再試行
                print("警告: SciPyのAPI互換性問題を検出しました。最小限のパラメータで再試行します。")
                x = solver_func(A, b)[0]
            iterations = None
        
        # 経過時間を計算
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        
        print(f"CPU (SciPy) 解法実行: {self.method}, 経過時間: {elapsed:.4f}秒")
        if iterations:
            print(f"反復回数: {iterations}")
            
        # 収束履歴を表示
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
        
        return x


class GPULinearSolver(LinearSolver):
    """GPU (CuPy) を使用した線形方程式系ソルバー"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        """
        GPU線形方程式系ソルバーを初期化
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        super().__init__(method, options, scaling_method)
        self.A_gpu = None
        self.A_scaled_gpu = None
        self.scaling_info = None
        self.scaler = None
        self.original_A = None
    
    def __del__(self):
        """デストラクタ：インスタンス削除時にGPUメモリを解放"""
        self.clear_gpu_memory()
    
    def clear_gpu_memory(self):
        """GPU上の行列メモリを解放"""
        self.A_gpu = None
        self.A_scaled_gpu = None
        self.scaling_info = None
        self.scaler = None
        if hasattr(self, 'original_A'):
            self.original_A = None
    
    def _is_new_matrix(self, A):
        """現在のGPU行列と異なる行列かどうかを判定"""
        # GPU行列がまだないか、形状が違う場合は新しい行列
        if self.A_gpu is None or self.original_A is None or self.A_gpu.shape != A.shape:
            return True
        
        # 形状が同じ場合は、保存している参照と比較
        if self.original_A is A:
            return False
        
        # オブジェクトが異なれば新しい行列と判断
        return True
    
    def _apply_scaling(self, A, b):
        """
        行列と右辺ベクトルにスケーリングを適用
        
        Args:
            A: システム行列 (GPU)
            b: 右辺ベクトル (GPU)
            
        Returns:
            tuple: (scaled_A, scaled_b, scaling_info, scaler)
        """
        if self.scaling_method is not None:
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング: {scaler.name}")
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
        
        return A, b, None, None
    
    def _solve_gpu(self, A, b, callback=None):
        """
        GPU上で線形方程式系を解く
        
        Args:
            A: システム行列 (CuPy CSR形式)
            b: 右辺ベクトル (CuPy配列)
            callback: 収束モニタリング用コールバック関数
            
        Returns:
            tuple: (x, iterations)
        """
        import cupy as cp
        import cupyx.scipy.sparse.linalg as splinalg
        
        if self.method == "direct":
            x = splinalg.spsolve(A, b)
            iterations = None
        elif self.method == "gmres":
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            restart = self.options.get("restart", 100)
            x0 = cp.ones_like(b)
            x, iterations = splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                          restart=restart, callback=callback)
        elif self.method in ["cg", "cgs", "minres"]:
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg, self.method)
            x0 = cp.ones_like(b)
            x, iterations = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        elif self.method in ["lsqr", "lsmr"]:
            solver_func = getattr(splinalg, self.method)
            maxiter = self.options.get("maxiter", 1000)
            if self.method == "lsmr":
                x0 = cp.ones_like(b)
                x = solver_func(A, b, x0=x0, maxiter=maxiter)[0]
            else:
                x = solver_func(A, b)[0]
            iterations = None
        else:
            print(f"未知の解法 {self.method}。直接解法を使用します。")
            x = splinalg.spsolve(A, b)
            iterations = None
        
        return x, iterations
        
    def solve(self, A, b):
        """
        線形方程式系を解く
        
        Args:
            A: システム行列 (CPU/SciPy形式)
            b: 右辺ベクトル (CPU/NumPy形式)
            
        Returns:
            解ベクトル x (CPU/NumPy形式)
        """
        # GPUが利用可能かチェック
        try:
            import cupy as cp
            import cupyx.scipy.sparse as sp
        except ImportError:
            print("CuPyが利用できません。CPUソルバーに切り替えます。")
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        # スケーリング実行前のタイミング計測
        start_time = time.time()
        residuals = []
        
        # 行列が変更されたかチェック
        is_new_matrix = self._is_new_matrix(A)
        
        # 新しい行列の場合はGPUに転送
        if is_new_matrix:
            print("行列を GPU (CuPy) に転送しています...")
            self.clear_gpu_memory()
            self.A_gpu = sp.csr_matrix(A)
            self.original_A = A  # 元の行列への参照を保存
            
            # 新しい行列なのでスケーリング情報もリセット
            self.A_scaled_gpu = None
            self.scaling_info = None
            self.scaler = None
        else:
            print("GPU上の既存行列を再利用します")
        
        # bをGPUに転送
        b_gpu = cp.array(b)
        
        # スケーリングを適用または再利用
        if is_new_matrix or self.A_scaled_gpu is None:
            # 新しい行列または初めての呼び出し時
            self.A_scaled_gpu, b_scaled, self.scaling_info, self.scaler = self._apply_scaling(self.A_gpu, b_gpu)
        else:
            # 既存のスケーリング済み行列を再利用して、bだけを新たにスケーリング
            if self.scaling_method is not None and self.scaler is not None:
                # スケーリング情報からbをスケーリング
                if hasattr(self.scaler, 'scale_b_only'):
                    # scale_b_onlyメソッドがあればそれを使用
                    b_scaled = self.scaler.scale_b_only(b_gpu, self.scaling_info)
                else:
                    # なければ完全なスケーリングを実行しA_scaled_gpuは無視
                    _, b_scaled, _, _ = self.scaler.scale(self.A_gpu, b_gpu)
            else:
                b_scaled = b_gpu
        
        # モニタリングコールバック
        if self.monitor_convergence:
            def callback(xk):
                residual = cp.linalg.norm(b_scaled - self.A_scaled_gpu @ xk) / cp.linalg.norm(b_scaled)
                residuals.append(float(residual))
                if len(residuals) % 10 == 0:
                    print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
        else:
            callback = None
            
        # GPU で計算を実行
        try:
            x_gpu, iterations = self._solve_gpu(self.A_scaled_gpu, b_scaled, callback)
        except Exception as e:
            print(f"GPU処理でエラー: {e}")
            print("CPU (SciPy) に切り替えて計算を実行します...")
            # エラー時はGPUメモリを解放して再試行を容易にする
            self.clear_gpu_memory()
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        # スケーリングを戻す
        if self.scaling_info is not None and self.scaler is not None:
            x_gpu = self.scaler.unscale(x_gpu, self.scaling_info)
            
        # 結果をCPUに転送
        x = x_gpu.get()
        
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        
        print(f"GPU (CuPy) 解法実行: {self.method}, 経過時間: {elapsed:.4f}秒")
        if iterations:
            print(f"反復回数: {iterations}")
            
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
        
        return x


class JAXLinearSolver(LinearSolver):
    """JAX を使用した線形方程式系ソルバー"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        """
        JAX線形方程式系ソルバーを初期化
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        super().__init__(method, options, scaling_method)
        self.last_matrix = None
        self.last_op = None
    
    def solve(self, A, b):
        """
        JAX を使用して線形方程式系を解く
        
        Args:
            A: システム行列 (SciPy CSR形式)
            b: 右辺ベクトル (NumPy配列)
            
        Returns:
            解ベクトル x (NumPy配列)
        """
        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental import sparse as jsparse
        except ImportError:
            print("JAXがインストールされていません。CPUソルバーに切り替えます。")
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        # 開始時間を記録
        start_time = time.time()
        residuals = []
        
        # 行列と右辺ベクトルをJAXで処理できる形式に変換
        data = jnp.array(A.data)
        indices = jnp.array(A.indices)
        indptr = jnp.array(A.indptr)
        b_jax = jnp.array(b)
        shape = A.shape
        
        # 行列-ベクトル積を定義
        def matvec(x):
            return jsparse.csr_matvec(data, indices, indptr, shape[1], x)
        
        # 新しい行列かどうかチェック
        new_matrix = (self.last_matrix is None or 
                      self.last_matrix[0] is not A or 
                      self.last_matrix[1].shape != shape)
        
        # 新しい行列の場合、JAX操作をコンパイル
        if new_matrix:
            print("JAX MatVec関数をコンパイル中...")
            self.last_op = jax.jit(matvec)
            self.last_matrix = (A, A)  # 元の行列への参照を保存
        
        # 解法選択
        if self.method == "direct":
            # JAXでは密行列に変換して直接解法
            dense_A = A.toarray()
            dense_A_jax = jnp.array(dense_A)
            
            try:
                # JAXの直接解法で解く
                x = jnp.linalg.solve(dense_A_jax, b_jax)
                iterations = None
            except Exception as e:
                print(f"JAXの直接解法でエラー: {e}")
                print("NumPy直接解法にフォールバックします。")
                x = np.linalg.solve(dense_A, b)
                iterations = None
        
        elif self.method == "cg":
            # 共役勾配法のパラメータ設定
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            
            # 初期値設定
            x0 = jnp.ones_like(b_jax)
            r0 = b_jax - self.last_op(x0)
            p0 = r0
            
            # モニタリングコールバック
            if self.monitor_convergence:
                def callback(state):
                    x_k, _, _, iter_num = state
                    residual = jnp.linalg.norm(b_jax - self.last_op(x_k)) / jnp.linalg.norm(b_jax)
                    residuals.append(float(residual))
                    if len(residuals) % 10 == 0:
                        print(f"  反復 {iter_num}: 残差 = {residual:.6e}")
                    return state
            else:
                callback = None
            
            # JAX特有の効率的CG実装
            def cg_step(state):
                x_k, r_k, p_k, k = state
                Ap_k = self.last_op(p_k)
                alpha_k = jnp.dot(r_k, r_k) / jnp.dot(p_k, Ap_k)
                x_next = x_k + alpha_k * p_k
                r_next = r_k - alpha_k * Ap_k
                beta_k = jnp.dot(r_next, r_next) / jnp.dot(r_k, r_k)
                p_next = r_next + beta_k * p_k
                return x_next, r_next, p_next, k + 1
                
            def cg_while_loop(val):
                x_k, r_k, p_k, k = val
                return (jnp.linalg.norm(r_k) > tol) & (k < maxiter)
            
            # JITコンパイルされたCGループ
            cg_loop = jax.jit(lambda state: jax.lax.while_loop(cg_while_loop, cg_step, state))
            
            # 初期状態を設定して実行
            init_state = (x0, r0, p0, 0)
            x_final, r_final, _, iterations = cg_loop(init_state)
            
            x = x_final
            
        else:
            print(f"JAXでは{self.method}解法は実装されていません。CPUソルバーに切り替えます。")
            return CPULinearSolver(self.method, self.options, self.scaling_method).solve(A, b)
        
        # 計算時間とログ出力
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        
        print(f"JAX解法実行: {self.method}, 経過時間: {elapsed:.4f}秒")
        if iterations:
            print(f"反復回数: {iterations}")
            
        # 残差の可視化
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
            
        # JAX配列からNumPy配列に変換
        return np.array(x)


def create_solver(method="direct", options=None, scaling_method=None, backend="cuda"):
    """
    適切な線形ソルバーを作成するファクトリ関数
    
    Args:
        method: 解法メソッド
        options: ソルバーオプション辞書
        scaling_method: スケーリング手法名
        backend: 計算バックエンド ('cpu', 'cuda', 'jax')
        
    Returns:
        LinearSolver: 指定されたバックエンドのソルバーインスタンス
    """
    # オプションからバックエンド指定を取得（引数のbackendが優先）
    if options and "backend" in options:
        backend = options.get("backend", backend)
    
    if backend == "cpu":
        # CPU (SciPy) ソルバーを使用
        return CPULinearSolver(method, options, scaling_method)
    elif backend == "jax":
        # JAX ソルバーを使用
        try:
            import jax
            return JAXLinearSolver(method, options, scaling_method)
        except ImportError:
            print("JAXが利用できません。CUDAソルバーを試行します。")
            backend = "cuda"  # JAX利用不可の場合はCUDAにフォールバック
    
    # デフォルト: CUDAソルバーを試行
    if backend == "cuda":
        try:
            import cupy
            return GPULinearSolver(method, options, scaling_method)
        except ImportError:
            print("CuPyが利用できません。CPUソルバーに切り替えます。")
            return CPULinearSolver(method, options, scaling_method)
    
    # 不明なバックエンドの場合はCPUソルバーを返す
    print(f"未知のバックエンド '{backend}'。CPUソルバーを使用します。")
    return CPULinearSolver(method, options, scaling_method)