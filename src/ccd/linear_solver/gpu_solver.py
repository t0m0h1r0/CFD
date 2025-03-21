"""
GPU (CuPy) を使用した線形方程式系ソルバー（疎行列CSR最適化版）
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
            
            # 行列をCuPy形式に変換（CSR最適化）
            self.A = self._to_cupy_matrix(self.original_A)
            
            # メモリ使用量情報を表示（デバッグ用）
            self._print_memory_info("初期化後")
            
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
            
            # 追加のCuPy固有のソルバーがあればここに追加
            
        except ImportError as e:
            print(f"警告: CuPyが利用できません: {e}")
            self.has_cupy = False
            self._init_cpu_fallback()
    
    def _init_cpu_fallback(self):
        """CPUソルバーにフォールバック"""
        self.cpu_solver = CPULinearSolver(
            self.original_A, 
            self.enable_dirichlet, 
            self.enable_neumann, 
            self.scaling_method
        )
        # CPU版のsolversをコピー
        self.solvers = self.cpu_solver.solvers
    
    def solve(self, b, method=None, options=None):
        """CuPyが使えない場合はCPUソルバーにフォールバック"""
        if not self.has_cupy:
            return self.cpu_solver.solve(b, method, options)
        
        # 通常の処理
        return super().solve(b, method, options)
    
    def _to_cupy_matrix(self, A):
        """
        行列をCuPy CSR形式に効率的に変換
        
        SciPyのCSR形式から直接CuPyのCSR形式に変換し、
        メモリ使用量を最小限に抑える
        """
        if not self.has_cupy:
            return A
            
        try:
            # すでにCuPy上にある場合は何もしない
            if hasattr(A, 'device') and str(type(A)).find('cupy') >= 0:
                return A
                
            # 行列の特性を確認
            is_sparse = hasattr(A, 'format')
            matrix_size = A.shape[0] * A.shape[1]
            large_matrix = matrix_size > 1e7  # ~80MB以上を大きいと判断
            
            # SciPyのCSR行列から直接CuPyのCSR行列に変換（メモリ効率が良い）
            if is_sparse and hasattr(A, 'format') and A.format == 'csr':
                print(f"CSR行列を直接変換（サイズ: {A.shape}, 非ゼロ要素: {A.nnz}）")
                
                # 非ゼロ値、行インデックス、列インデックスを抽出
                data = A.data
                indices = A.indices
                indptr = A.indptr
                
                # 非常に小さい値を切り捨てて疎性を高める
                if len(data) > 0:
                    small_values_mask = np.abs(data) < 1e-15
                    if np.any(small_values_mask):
                        # 値の数を削減してCSRを再構築
                        data = data[~small_values_mask]
                        indices = indices[~small_values_mask]
                        # indptrの再計算は複雑なので、新しいCSRを構築
                        from scipy import sparse
                        A = sparse.csr_matrix((data, indices, indptr), shape=A.shape)
                        data = A.data
                        indices = A.indices
                        indptr = A.indptr
                
                # CuPyのCSR行列に変換
                cupy_data = self.cp.array(data)
                cupy_indices = self.cp.array(indices)
                cupy_indptr = self.cp.array(indptr)
                
                return self.cp.sparse.csr_matrix(
                    (cupy_data, cupy_indices, cupy_indptr),
                    shape=A.shape
                )
                
            # 大きなCOO/CSC形式の行列の場合（直接変換可能な場合）
            elif is_sparse and hasattr(A, 'format') and A.format in ['coo', 'csc']:
                print(f"{A.format.upper()}行列をCSRに変換（サイズ: {A.shape}, 非ゼロ要素: {A.nnz}）")
                # 一度SciPyのCSRに変換
                from scipy import sparse
                csr_A = A.tocsr()
                
                # CuPyのCSR行列に変換
                cupy_data = self.cp.array(csr_A.data)
                cupy_indices = self.cp.array(csr_A.indices)
                cupy_indptr = self.cp.array(csr_A.indptr)
                
                return self.cp.sparse.csr_matrix(
                    (cupy_data, cupy_indices, cupy_indptr),
                    shape=A.shape
                )
                
            # 大きな密行列の場合（分割処理が必要）
            elif large_matrix and not is_sparse:
                print(f"大きな密行列を分割処理（サイズ: {A.shape}）")
                
                # 密行列のCSR形式への変換（メモリ効率を考慮）
                A_dense = np.array(A)
                
                # 小さな値を0に設定して疎性を高める
                A_dense[np.abs(A_dense) < 1e-15] = 0.0
                
                # NumPyのCSRに変換してからCuPyのCSRに変換
                from scipy import sparse
                scipy_csr = sparse.csr_matrix(A_dense)
                
                # CuPyのCSR行列に変換
                cupy_data = self.cp.array(scipy_csr.data)
                cupy_indices = self.cp.array(scipy_csr.indices)
                cupy_indptr = self.cp.array(scipy_csr.indptr)
                
                return self.cp.sparse.csr_matrix(
                    (cupy_data, cupy_indices, cupy_indptr),
                    shape=A.shape
                )
                
            # その他の場合（小さな密行列など）
            else:
                print(f"通常変換（サイズ: {A.shape}）")
                
                # 密行列に変換
                if is_sparse:
                    A_dense = A.toarray()
                else:
                    A_dense = np.array(A)
                
                # 小さな値を0に設定して疎性を高める
                A_dense[np.abs(A_dense) < 1e-15] = 0.0
                
                # CuPyのCSR形式に変換
                return self.cp.sparse.csr_matrix(self.cp.array(A_dense))
            
        except Exception as e:
            print(f"GPU行列変換エラー: {e}")
            import traceback
            traceback.print_exc()
            print("CPUソルバーにフォールバックします")
            self.has_cupy = False
            self._init_cpu_fallback()
            return A
    
    def _print_memory_info(self, label=""):
        """CuPyのメモリ使用状況を出力（デバッグ用）"""
        if not self.has_cupy:
            return
            
        try:
            mem_pool = self.cp.get_default_memory_pool()
            mem_used = mem_pool.used_bytes()
            mem_total = mem_pool.total_bytes()
            
            print(f"[GPU メモリ情報 {label}] 使用: {mem_used/1024/1024:.2f}MB, "
                  f"合計プール: {mem_total/1024/1024:.2f}MB")
                  
            if hasattr(self, 'A') and hasattr(self.A, 'format') and self.A.format == 'csr':
                nnz = self.A.nnz
                shape = self.A.shape
                density = nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0
                print(f"[行列情報] 形状: {shape}, 非ゼロ要素: {nnz}, 密度: {density:.6f}")
        except Exception as e:
            print(f"メモリ情報取得エラー: {e}")
    
    def _prepare_scaling(self):
        """スケーリング前処理（堅牢化）"""
        if not self.scaler or not self.has_cupy:
            return
            
        try:
            # NumPy用ダミーベクトルでスケーリング情報を計算
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
                    
            # メモリ情報出力
            self._print_memory_info("スケーリング後")
        except Exception as e:
            print(f"スケーリング前処理エラー: {e}")
            self.scaler = None
    
    def _preprocess_vector(self, b):
        """ベクトルをCuPy配列に変換"""
        if not self.has_cupy:
            return b
            
        # 既にCuPy配列の場合
        if hasattr(b, 'device') and str(type(b)).find('cupy') >= 0:
            return b
        
        # NumPy配列からCuPy配列に変換
        return self.cp.array(b)
    
    def _apply_scaling_to_b(self, b):
        """右辺ベクトルにスケーリングを適用"""
        if not self.has_cupy:
            return self.cpu_solver._apply_scaling_to_b(b)
            
        if self.scaler and self.scaling_info:
            row_scale = self.scaling_info.get('row_scale')
            if row_scale is not None:
                return b * row_scale
            else:
                D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                if D_sqrt_inv is not None:
                    return b * D_sqrt_inv
        return b
    
    def _apply_unscaling_to_x(self, x):
        """解ベクトルにアンスケーリングを適用"""
        if not self.has_cupy:
            return self.cpu_solver._apply_unscaling_to_x(x)
            
        if self.scaler and self.scaling_info:
            col_scale = self.scaling_info.get('col_scale')
            if col_scale is not None:
                return x / col_scale
            else:
                D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                if D_sqrt_inv is not None:
                    return x * D_sqrt_inv
        return x
    
    def _direct_fallback(self, A, b):
        """CPUソルバーを使用した直接解法フォールバック"""
        if not self.has_cupy:
            return self.cpu_solver._direct_fallback(A, b)
            
        # CuPy->NumPyに変換してCPUソルバーを使用
        import scipy.sparse.linalg as splinalg
        A_np = self._to_numpy_matrix(A)
        b_np = self._to_numpy_vector(b)
        x = splinalg.spsolve(A_np, b_np)
        return self.cp.array(x), None
    
    def _to_numpy_matrix(self, A):
        """行列をNumPy形式に変換"""
        if hasattr(A, 'get'):
            # CuPyの疎行列の場合
            if hasattr(A, 'format') and A.format == 'csr':
                # CSR行列の各コンポーネントをNumPyに転送
                data = A.data.get()
                indices = A.indices.get()
                indptr = A.indptr.get()
                shape = A.shape
                
                # SciPyのCSR行列を構築
                from scipy import sparse
                return sparse.csr_matrix((data, indices, indptr), shape=shape)
            else:
                # その他の場合は.get()を使用
                return A.get()
        return A
    
    def _to_numpy_vector(self, b):
        """ベクトルをNumPy形式に変換"""
        if hasattr(b, 'get'):
            return b.get()
        return b
    
    # 各ソルバーメソッドからエラー処理を排除
    
    def _solve_direct(self, A, b, options=None):
        """直接解法"""
        x = self.splinalg.spsolve(A, b)
        return x, None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 200)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # GMRES実行
        result = self.splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
        return result[0], result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # CG実行
        result = self.splinalg.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
        return result[0], result[1]
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # CGS実行
        result = self.splinalg.cgs(A, b, x0=x0, tol=tol, maxiter=maxiter)
        return result[0], result[1]
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        x0 = options.get("x0", self.cp.zeros_like(b))
        
        # MINRES実行
        result = self.splinalg.minres(A, b, x0=x0, tol=tol, maxiter=maxiter)
        return result[0], result[1]
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # CuPy cupyx.scipy.sparse.linalg.lsqr は引数が少ない
        result = self.splinalg.lsqr(A, b)
        return result[0], result[2]
    
    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        result = self.splinalg.lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
        return result[0], result[2]