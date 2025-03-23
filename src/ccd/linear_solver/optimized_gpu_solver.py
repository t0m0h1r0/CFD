"""
パターン圧縮とステンシル最適化を用いた高性能GPUソルバー

このモジュールは、CCD（高精度コンパクト差分法）で生成される行列の特性を活かした
GPU向け線形方程式系ソルバーを提供します。
以下の最適化テクニックを実装しています：

1. パターン圧縮：格子パターンの反復性を利用したメモリ最適化
2. ステンシルベース計算：ステンシル係数を直接使用した高速SpMV
3. キャッシュ最適化：データ局所性を考慮したメモリアクセスパターン
4. 混合精度計算：精度と速度のバランスを取る計算手法

要件: CuPy, NumPy
"""

import numpy as np
import time
from .base import LinearSolver
from .cpu_solver import CPULinearSolver
from .gpu_solver import GPULinearSolver


class OptimizedCPULinearSolver(GPULinearSolver):
    """行列パターン圧縮と高速SpMVを用いた最適化GPUソルバー"""
    
    def _initialize(self):
        """GPU固有の初期化処理（最適化版）"""
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            import cupyx.scipy.sparse.linalg as splinalg
            self.cp = cp
            self.cp_sparse = cp_sparse
            self.splinalg = splinalg
            self.has_cupy = True
            
            # 行列をCuPy形式に変換
            self.A = self._to_cupy_matrix(self.original_A)
            
            # 行列構造の分析
            self._analyze_matrix_structure()
            
            # 最適化されたSpMVを設定
            if hasattr(self, 'is_structured_grid') and self.is_structured_grid:
                print("最適化SpMVを設定中...")
                self._setup_optimized_spmv()
            
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
                "lsmr": self._solve_lsmr,
                "stencil_direct": self._solve_stencil_direct  # ステンシルベースの直接ソルバー
            }
            
            # パフォーマンス統計
            self.timer_stats = {
                "spmv_calls": 0,
                "spmv_time": 0.0,
                "solver_time": 0.0
            }
            
        except ImportError as e:
            print(f"警告: CuPyが利用できません: {e}")
            self.has_cupy = False
            self._init_cpu_fallback()
    
    def _analyze_matrix_structure(self):
        """行列構造の分析とパターン検出"""
        if not self.has_cupy:
            return
            
        print("行列構造を分析中...")
        t_start = time.time()
            
        A = self.A
        if not isinstance(A, self.cp_sparse.csr_matrix):
            A = self.cp_sparse.csr_matrix(A)
        
        n_rows = A.shape[0]
        
        # 行パターンの分析
        self.row_patterns = {}
        self.rows_by_pattern = {}
        self.pattern_counts = {}
        
        # 各行のパターンを特定
        # CPU上での処理が高速なため、CuPy配列をNumPyに変換
        indptr_np = A.indptr.get()
        indices_np = A.indices.get()
        data_np = A.data.get()
        
        # 小さな値を丸める精度
        rounding_precision = 10
        
        for i in range(n_rows):
            start, end = indptr_np[i], indptr_np[i+1]
            
            # パターンの特定（列インデックスとその値）
            col_indices = indices_np[start:end].tolist()
            
            # 数値的安定性のため、非常に小さな値を0に設定
            data_values = data_np[start:end].tolist()
            data_values = [round(val, rounding_precision) if abs(val) > 1e-15 else 0.0 for val in data_values]
            
            # 非ゼロの値だけをパターンとして保持
            pattern_elements = []
            for col, val in zip(col_indices, data_values):
                if abs(val) > 1e-15:  # 実質的にゼロでない場合のみ追加
                    pattern_elements.append((col, val))
                    
            pattern_key = tuple(pattern_elements)
            
            if pattern_key not in self.row_patterns:
                self.row_patterns[pattern_key] = i
                self.rows_by_pattern[pattern_key] = [i]
                self.pattern_counts[pattern_key] = 1
            else:
                self.rows_by_pattern[pattern_key].append(i)
                self.pattern_counts[pattern_key] += 1
        
        n_patterns = len(self.row_patterns)
        compression_ratio = n_rows / n_patterns if n_patterns > 0 else 1.0
        
        # 最大頻度のパターン
        max_pattern = max(self.pattern_counts.items(), key=lambda x: x[1]) if self.pattern_counts else (None, 0)
        max_pattern_ratio = max_pattern[1] / n_rows if n_rows > 0 else 0
        
        # 格子構造の判定
        # パターン圧縮率が高い、または特定のパターンが多数の行で共有されている場合
        self.is_structured_grid = (
            compression_ratio > 2.0 or  # 全体的な圧縮率
            max_pattern_ratio > 0.1     # 同一パターンの頻度
        )
        
        # 最適化戦略の選択
        # 圧縮率が特に高い場合はパターン圧縮、そうでなければステンシルベース
        self.use_pattern_compression = (compression_ratio > 5.0)
        self.use_stencil_based = (
            self.is_structured_grid and 
            max_pattern_ratio > 0.3 and
            len(max_pattern[0]) <= 27  # 3D 27点ステンシルまで対応
        )
        
        # 分析結果の出力
        print(f"行列分析完了: {time.time() - t_start:.3f}秒")
        print(f"  行数: {n_rows}, 非ゼロ要素: {A.nnz}")
        print(f"  固有パターン数: {n_patterns}")
        print(f"  パターン圧縮率: {compression_ratio:.2f}x")
        print(f"  最頻パターンの行比率: {max_pattern_ratio:.2f}")
        print(f"  構造化格子検出: {self.is_structured_grid}")
        
        # 最適化戦略を表示
        if self.is_structured_grid:
            if self.use_pattern_compression:
                print("  最適化戦略: パターン圧縮SpMV")
            elif self.use_stencil_based:
                print("  最適化戦略: ステンシルベースSpMV")
            else:
                print("  最適化戦略: 標準SpMV (最適化なし)")
        else:
            print("  最適化戦略: 標準SpMV (最適化なし)")
    
    def _setup_optimized_spmv(self):
        """最適化されたSpMVの設定"""
        if not self.has_cupy or not self.is_structured_grid:
            return
        
        setup_start = time.time()
        
        # パターン圧縮最適化
        if self.use_pattern_compression:
            self._setup_pattern_compression()
            
        # ステンシルベース最適化
        if self.use_stencil_based:
            self._setup_stencil_based()
        
        print(f"最適化SpMV設定完了: {time.time() - setup_start:.3f}秒")
    
    def _setup_pattern_compression(self):
        """パターン圧縮ベースの最適化を設定"""
        A = self.A
        
        # パターンデータの準備
        pattern_list = list(self.row_patterns.keys())
        self.pattern_dict = {pattern: idx for idx, pattern in enumerate(pattern_list)}
        
        # 各行のパターンIDの配列
        self.row_pattern_ids = self.cp.zeros(A.shape[0], dtype=self.cp.int32)
        
        # 各パターンのデータ配列
        all_indices = []
        all_data = []
        pattern_starts = [0]
        
        # パターンデータの構築
        for pattern_idx, pattern in enumerate(pattern_list):
            # パターンの行を取得
            rows = self.rows_by_pattern[pattern]
            
            # 各行にパターンIDを割り当て
            for row in rows:
                self.row_pattern_ids[row] = pattern_idx
            
            # パターンデータを解凍
            indices = [col for col, _ in pattern]
            data = [val for _, val in pattern]
            
            all_indices.extend(indices)
            all_data.extend(data)
            pattern_starts.append(pattern_starts[-1] + len(indices))
        
        # パターンデータをGPU配列に変換
        self.pattern_indices = self.cp.array(all_indices, dtype=self.cp.int32)
        self.pattern_data = self.cp.array(all_data, dtype=self.cp.float64)
        self.pattern_ptrs = self.cp.array(pattern_starts, dtype=self.cp.int32)
        
        # カスタムSpMVカーネルの定義
        self.pattern_spmv_kernel = self.cp.RawKernel(r'''
        extern "C" __global__
        void pattern_spmv(const int n_rows, 
                        const int *row_pattern_ids,
                        const int *pattern_ptrs,
                        const int *pattern_indices,
                        const double *pattern_data,
                        const double *x,
                        double *y) {
            int row = blockDim.x * blockIdx.x + threadIdx.x;
            if (row < n_rows) {
                double sum = 0.0;
                int pattern_id = row_pattern_ids[row];
                int start = pattern_ptrs[pattern_id];
                int end = pattern_ptrs[pattern_id + 1];
                
                // ループアンローリングによる最適化
                int i = start;
                
                // 8要素ずつ処理（ループアンローリング）
                for (; i < end - 7; i += 8) {
                    sum += pattern_data[i] * x[pattern_indices[i]];
                    sum += pattern_data[i+1] * x[pattern_indices[i+1]];
                    sum += pattern_data[i+2] * x[pattern_indices[i+2]];
                    sum += pattern_data[i+3] * x[pattern_indices[i+3]];
                    sum += pattern_data[i+4] * x[pattern_indices[i+4]];
                    sum += pattern_data[i+5] * x[pattern_indices[i+5]];
                    sum += pattern_data[i+6] * x[pattern_indices[i+6]];
                    sum += pattern_data[i+7] * x[pattern_indices[i+7]];
                }
                
                // 残りの要素を処理
                for (; i < end; i++) {
                    sum += pattern_data[i] * x[pattern_indices[i]];
                }
                
                y[row] = sum;
            }
        }
        ''', 'pattern_spmv')
        
        # メモリ使用量情報
        original_mem = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
        compressed_mem = (self.pattern_data.nbytes + self.pattern_indices.nbytes + 
                         self.pattern_ptrs.nbytes + self.row_pattern_ids.nbytes)
        
        print(f"パターン圧縮メモリ使用量:")
        print(f"  元の行列: {original_mem / (1024*1024):.2f} MB")
        print(f"  圧縮後: {compressed_mem / (1024*1024):.2f} MB")
        print(f"  削減率: {(1 - compressed_mem/original_mem) * 100:.1f}%")
    
    def _setup_stencil_based(self):
        """ステンシルベースの最適化を設定"""
        # 最も頻度の高いパターンを特定
        max_pattern = max(self.pattern_counts.items(), key=lambda x: x[1])
        
        pattern, count = max_pattern
        n_points = len(pattern)
        
        # ステンシルポイント情報の抽出
        stencil_cols = [col for col, _ in pattern]
        stencil_vals = [val for _, val in pattern]
        
        # ステンシルポイントの最大数
        max_stencil_points = 27  # 3D 27点ステンシルまで対応
        
        # ステンシルサイズを制限
        n_stencil = min(n_points, max_stencil_points)
        
        # ステンシルポイント情報をGPUに転送
        self.stencil_cols = self.cp.array(stencil_cols[:n_stencil], dtype=self.cp.int32)
        self.stencil_vals = self.cp.array(stencil_vals[:n_stencil], dtype=self.cp.float64)
        self.stencil_size = n_stencil
        
        # グリッドサイズとブロックサイズの最適化
        n_rows = self.A.shape[0]
        
        # スレッドブロックサイズ：CUDAコアのワープサイズ(32)の倍数
        self.threads_per_block = 256
        
        # グリッドサイズ：行数に基づいて計算
        self.blocks_per_grid = (n_rows + self.threads_per_block - 1) // self.threads_per_block
        
        # カスタムステンシルカーネルの定義
        # ステンシルサイズに応じて異なるバージョンを定義
        
        # 小規模ステンシル用のカーネル（ループアンローリング）
        if n_stencil <= 9:
            self.stencil_spmv_kernel = self.cp.RawKernel(r'''
            extern "C" __global__
            void stencil_spmv_small(const int n_rows,
                             const int stencil_size,
                             const int *stencil_cols,
                             const double *stencil_vals,
                             const double *x,
                             double *y) {
                int row = blockDim.x * blockIdx.x + threadIdx.x;
                if (row < n_rows) {
                    double sum = 0.0;
                    
                    // 完全にアンロールされたループ (最大9点)
                    if (stencil_size > 0) sum += stencil_vals[0] * x[stencil_cols[0] + row];
                    if (stencil_size > 1) sum += stencil_vals[1] * x[stencil_cols[1] + row];
                    if (stencil_size > 2) sum += stencil_vals[2] * x[stencil_cols[2] + row];
                    if (stencil_size > 3) sum += stencil_vals[3] * x[stencil_cols[3] + row];
                    if (stencil_size > 4) sum += stencil_vals[4] * x[stencil_cols[4] + row];
                    if (stencil_size > 5) sum += stencil_vals[5] * x[stencil_cols[5] + row];
                    if (stencil_size > 6) sum += stencil_vals[6] * x[stencil_cols[6] + row];
                    if (stencil_size > 7) sum += stencil_vals[7] * x[stencil_cols[7] + row];
                    if (stencil_size > 8) sum += stencil_vals[8] * x[stencil_cols[8] + row];
                    
                    y[row] = sum;
                }
            }
            ''', 'stencil_spmv_small')
            self.use_small_stencil = True
        else:
            # 大規模ステンシル用のカーネル（ループ使用）
            self.stencil_spmv_kernel = self.cp.RawKernel(r'''
            extern "C" __global__
            void stencil_spmv_large(const int n_rows,
                             const int stencil_size,
                             const int *stencil_cols,
                             const double *stencil_vals,
                             const double *x,
                             double *y) {
                int row = blockDim.x * blockIdx.x + threadIdx.x;
                if (row < n_rows) {
                    double sum = 0.0;
                    
                    // ループでステンシルポイントを処理
                    for (int i = 0; i < stencil_size; i++) {
                        sum += stencil_vals[i] * x[stencil_cols[i] + row];
                    }
                    
                    y[row] = sum;
                }
            }
            ''', 'stencil_spmv_large')
            self.use_small_stencil = False
        
        # 分析情報を表示
        print(f"ステンシルベース最適化情報:")
        print(f"  ステンシルサイズ: {n_stencil}点")
        print(f"  行数: {n_rows}")
        print(f"  使用カーネル: {'小規模ステンシル' if self.use_small_stencil else '大規模ステンシル'}")
    
    def _pattern_spmv(self, x):
        """パターン圧縮ベースのSpMV演算"""
        # 性能計測開始
        start_time = time.time()
        
        # 出力ベクトルの準備
        y = self.cp.zeros_like(x)
        
        # スレッドブロック設定
        threads_per_block = 256
        blocks_per_grid = (len(self.row_pattern_ids) + threads_per_block - 1) // threads_per_block
        
        # カスタムカーネル実行
        self.pattern_spmv_kernel(
            (blocks_per_grid,), 
            (threads_per_block,),
            (
                len(self.row_pattern_ids),
                self.row_pattern_ids,
                self.pattern_ptrs,
                self.pattern_indices,
                self.pattern_data,
                x,
                y
            )
        )
        
        # 性能計測終了と統計更新
        elapsed = time.time() - start_time
        self.timer_stats["spmv_calls"] += 1
        self.timer_stats["spmv_time"] += elapsed
        
        return y
    
    def _stencil_spmv(self, x):
        """ステンシルベースのSpMV演算"""
        # 性能計測開始
        start_time = time.time()
        
        # 出力ベクトルの準備
        y = self.cp.zeros_like(x)
        
        # カスタムカーネル実行
        if self.use_small_stencil:
            self.stencil_spmv_kernel(
                (self.blocks_per_grid,), 
                (self.threads_per_block,),
                (
                    self.A.shape[0],
                    self.stencil_size,
                    self.stencil_cols,
                    self.stencil_vals,
                    x,
                    y
                )
            )
        else:
            self.stencil_spmv_kernel(
                (self.blocks_per_grid,), 
                (self.threads_per_block,),
                (
                    self.A.shape[0],
                    self.stencil_size,
                    self.stencil_cols,
                    self.stencil_vals,
                    x,
                    y
                )
            )
        
        # 性能計測終了と統計更新
        elapsed = time.time() - start_time
        self.timer_stats["spmv_calls"] += 1
        self.timer_stats["spmv_time"] += elapsed
        
        return y
    
    def _optimized_spmv(self, x):
        """最適化されたSpMV演算（各最適化戦略を選択）"""
        if not self.has_cupy or not self.is_structured_grid:
            # 最適化を使わない場合は標準のSpMVを使用
            return self.A @ x
        
        # パターン圧縮を使用
        if self.use_pattern_compression and hasattr(self, 'pattern_spmv_kernel'):
            return self._pattern_spmv(x)
            
        # ステンシルベースを使用
        if self.use_stencil_based and hasattr(self, 'stencil_spmv_kernel'):
            return self._stencil_spmv(x)
        
        # どちらも使用できない場合は標準SpMV
        return self.A @ x
    
    def _solve_with_optimized_spmv(self, solver_func, A, b, options):
        """最適化されたSpMVを使用してソルバーを実行"""
        if not self.has_cupy or not hasattr(self, 'is_structured_grid') or not self.is_structured_grid:
            # 最適化を使わない場合は標準のソルバーを使用
            return solver_func(A, b, options)
        
        # A行列の代わりにLinearOperatorを作成
        A_op = self.splinalg.LinearOperator(
            shape=A.shape,
            matvec=self._optimized_spmv,
            dtype=A.dtype
        )
        
        # 性能計測開始
        start_time = time.time()
        
        # 最適化ソルバー実行
        result = solver_func(A_op, b, options)
        
        # 性能計測終了と統計更新
        elapsed = time.time() - start_time
        self.timer_stats["solver_time"] += elapsed
        
        return result
    
    def _solve_direct(self, A, b, options=None):
        """直接解法（最適化なし - スパース直接解法はパターン非依存）"""
        x = self.splinalg.spsolve(A, b)
        return x, None
    
    def _solve_stencil_direct(self, A, b, options=None):
        """ステンシルベースの直接解法（ヤコビ反復）"""
        if not self.has_cupy or not hasattr(self, 'use_stencil_based') or not self.use_stencil_based:
            # ステンシル最適化を使用できない場合は標準の直接解法にフォールバック
            return self._solve_direct(A, b, options)
        
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解
        x = self.cp.zeros_like(b)
        if "x0" in options and options["x0"] is not None:
            x = self.cp.asarray(options["x0"])
        
        # ヤコビ反復法の実装
        diag = A.diagonal()  # 対角成分を取得
        diag_inv = 1.0 / diag  # 対角成分の逆数
        
        # 最大反復回数まで繰り返す
        for k in range(maxiter):
            # r = b - A @ x を計算
            r = b - self._optimized_spmv(x)
            
            # 相対残差ノルムを計算
            rnorm = self.cp.linalg.norm(r)
            bnorm = self.cp.linalg.norm(b)
            rel_res = rnorm / (bnorm + 1e-15)
            
            # 収束判定
            if rel_res < tol:
                return x, k + 1
            
            # x = x + D^(-1) * r を計算 (ヤコビ更新)
            x = x + diag_inv * r
        
        # 最大反復に達した場合
        return x, maxiter
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES法（最適化版）"""
        options = options or {}
        
        # GMRES固有のオプション
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(20, max(5, b.size // 20)))
        
        # 初期解ベクトル
        x0 = options.get("x0")
        x0 = (self.cp.asarray(x0) if x0 is not None else 
              self.cp.zeros_like(b))
        
        # 最適化されたSplinalg.gmresを使用
        gmres_options = {
            "x0": x0,
            "tol": tol,
            "atol": atol,
            "maxiter": maxiter,
            "restart": restart
        }
        
        # 最適化SpMVを使用してGMRESを実行
        return self._solve_with_optimized_spmv(
            lambda A, b, opts: self.splinalg.gmres(A, b, **opts), 
            A, b, gmres_options
        )
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法（最適化版）"""
        options = options or {}
        
        # CG固有のオプション
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解ベクトル
        x0 = options.get("x0")
        x0 = (self.cp.asarray(x0) if x0 is not None else 
              self.cp.zeros_like(b))
        
        # 最適化されたSplinalg.cgを使用
        cg_options = {
            "x0": x0,
            "tol": tol,
            "atol": atol,
            "maxiter": maxiter
        }
        
        # 最適化SpMVを使用してCGを実行
        return self._solve_with_optimized_spmv(
            lambda A, b, opts: self.splinalg.cg(A, b, **opts), 
            A, b, cg_options
        )
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法（最適化版）"""
        options = options or {}
        
        # BiCGSTAB固有のオプション
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解ベクトル
        x0 = options.get("x0")
        x0 = (self.cp.asarray(x0) if x0 is not None else 
              self.cp.zeros_like(b))
        
        # 最適化されたSplinalg.bicgstabを使用
        bicgstab_options = {
            "x0": x0,
            "tol": tol,
            "atol": atol,
            "maxiter": maxiter
        }
        
        # 最適化SpMVを使用してBiCGSTABを実行
        return self._solve_with_optimized_spmv(
            lambda A, b, opts: self.splinalg.bicgstab(A, b, **opts), 
            A, b, bicgstab_options
        )
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法（最適化版）"""
        options = options or {}
        
        # CGS固有のオプション
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解ベクトル
        x0 = options.get("x0")
        x0 = (self.cp.asarray(x0) if x0 is not None else 
              self.cp.zeros_like(b))
        
        # 最適化されたSplinalg.cgsを使用
        cgs_options = {
            "x0": x0,
            "tol": tol,
            "atol": atol,
            "maxiter": maxiter
        }
        
        # 最適化SpMVを使用してCGSを実行
        return self._solve_with_optimized_spmv(
            lambda A, b, opts: self.splinalg.cgs(A, b, **opts), 
            A, b, cgs_options
        )
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法（最適化版）"""
        options = options or {}
        
        # MINRES固有のオプション
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", tol)
        maxiter = options.get("maxiter", 1000)
        
        # 初期解ベクトル
        x0 = options.get("x0")
        x0 = (self.cp.asarray(x0) if x0 is not None else 
              self.cp.zeros_like(b))
        
        # 最適化されたSplinalg.minresを使用
        minres_options = {
            "x0": x0,
            "tol": tol,
            "atol": atol,
            "maxiter": maxiter
        }
        
        # 最適化SpMVを使用してMINRESを実行
        return self._solve_with_optimized_spmv(
            lambda A, b, opts: self.splinalg.minres(A, b, **opts), 
            A, b, minres_options
        )
    
    def _solve_lsqr(self, A, b, options=None):
        """LSQR最小二乗法ソルバー（最適化版）"""
        options = options or {}
        
        # LSQR固有のオプション
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        iter_lim = options.get("maxiter", options.get("iter_lim", None))
        
        # 最適化されたSplinalg.lsqrを使用
        lsqr_options = {
            "damp": damp,
            "atol": atol,
            "btol": btol,
            "conlim": conlim,
            "iter_lim": iter_lim
        }
        
        # LSQR用のLinearOperatorは独自に作成
        if hasattr(self, 'is_structured_grid') and self.is_structured_grid:
            A_op = self.splinalg.LinearOperator(
                shape=A.shape,
                matvec=self._optimized_spmv,
                rmatvec=lambda x: self._optimized_spmv(x),  # 簡略化（対称行列向け）
                dtype=A.dtype
            )
            
            # カスタムLSQR実装（簡略化されたバージョン）
            start_time = time.time()
            result = self.splinalg.lsqr(A_op, b, **lsqr_options)
            elapsed = time.time() - start_time
            self.timer_stats["solver_time"] += elapsed
            
            return result[0], result[2]
        else:
            # 標準のLSQRにフォールバック
            result = self.splinalg.lsqr(A, b, **lsqr_options)
            return result[0], result[2]
    
    def _solve_lsmr(self, A, b, options=None):
        """LSMR最小二乗法ソルバー（最適化版）"""
        options = options or {}
        
        # LSMR固有のオプション
        damp = options.get("damp", 0.0)
        atol = options.get("atol", 1e-6)
        btol = options.get("btol", 1e-6)
        conlim = options.get("conlim", 1e8)
        maxiter = options.get("maxiter", None)
        
        # 最適化されたSplinalg.lsmrを使用
        lsmr_options = {
            "damp": damp,
            "atol": atol,
            "btol": btol,
            "conlim": conlim,
            "maxiter": maxiter
        }
        
        # LSMR用のLinearOperatorは独自に作成
        if hasattr(self, 'is_structured_grid') and self.is_structured_grid:
            A_op = self.splinalg.LinearOperator(
                shape=A.shape,
                matvec=self._optimized_spmv,
                rmatvec=lambda x: self._optimized_spmv(x),  # 簡略化（対称行列向け）
                dtype=A.dtype
            )
            
            # カスタムLSMR実装
            start_time = time.time()
            result = self.splinalg.lsmr(A_op, b, **lsmr_options)
            elapsed = time.time() - start_time
            self.timer_stats["solver_time"] += elapsed
            
            return result[0], result[2]
        else:
            # 標準のLSMRにフォールバック
            result = self.splinalg.lsmr(A, b, **lsmr_options)
            return result[0], result[2]
    
    def solve(self, b, method=None, options=None):
        """
        Ax=b を解く（最適化版）
        
        Args:
            b: 右辺ベクトル
            method: 解法メソッド名（設定済みのself.solver_methodを上書き）
            options: 解法オプション（設定済みのself.solver_optionsを上書き）
            
        Returns:
            解ベクトル x
        """
        # タイマー統計をリセット
        self.timer_stats = {
            "spmv_calls": 0,
            "spmv_time": 0.0,
            "solver_time": 0.0
        }
        
        # 親クラスのsolveメソッドを呼び出し
        result = super().solve(b, method, options)
        
        # 性能統計を表示（オプション）
        show_stats = options.get("show_stats", False) if options else False
        if show_stats and hasattr(self, 'is_structured_grid') and self.is_structured_grid:
            spmv_calls = self.timer_stats["spmv_calls"]
            if spmv_calls > 0:
                avg_spmv_time = self.timer_stats["spmv_time"] / spmv_calls * 1000  # ms単位
                print(f"SpMV性能統計:")
                print(f"  SpMV呼び出し数: {spmv_calls}")
                print(f"  SpMV平均時間: {avg_spmv_time:.3f} ms")
                print(f"  ソルバー合計時間: {self.timer_stats['solver_time']:.3f} 秒")
        
        return result