"""
パターン圧縮を利用した最適化GPU (CuPy) 線形方程式系ソルバー
繰り返しブロック構造を活用してGPUメモリを大幅に削減します
"""

import numpy as np
from .gpu_solver import GPULinearSolver


class OptimizedGPULinearSolver(GPULinearSolver):
    """
    パターン圧縮を利用した最適化GPU線形ソルバー
    1D: 4x4, 2D: 7x7, 3D: 10x10のブロックパターンを圧縮
    """
    
    def _initialize(self):
        """GPU固有の初期化処理"""
        # 親クラスの初期化メソッドを呼び出し
        super()._initialize()
        
        # cupyが使用できる場合のみ、最適化用の属性を初期化
        if self.has_cupy:
            self.use_optimized_spmv = True
            self.optimized_A = None
            self.memory_savings = 0.0
            self.pattern_stats = None
            
            # 次元を特定する
            self.dimension = self._detect_dimension()
            
            # 最適化行列の作成
            self._prepare_pattern_compressed_matrix()
    
    def _detect_dimension(self):
        """
        行列サイズから問題の次元を検出
        
        Returns:
            int: 検出された次元 (1, 2, 3)
        """
        # 行サイズから推定
        n = self.A.shape[0]
        
        # 4の倍数は1D、7の倍数は2D、10の倍数は3D
        if n % 10 == 0:
            print("3次元問題を検出しました")
            return 3
        elif n % 7 == 0:
            print("2次元問題を検出しました")
            return 2
        elif n % 4 == 0:
            print("1次元問題を検出しました")
            return 1
        else:
            print(f"不明な次元です: 行数 = {n}")
            return 0
    
    def _prepare_pattern_compressed_matrix(self):
        """
        ブロックパターンに基づいて圧縮された行列を準備
        """
        if not self.has_cupy or self.dimension == 0:
            return
            
        try:
            # 次元に応じたブロックサイズを設定
            if self.dimension == 1:
                block_size = 4
            elif self.dimension == 2:
                block_size = 7
            else:  # 3次元
                block_size = 10
                
            # CSR形式の行列を取得
            if not hasattr(self.A, 'format') or self.A.format != 'csr':
                # CSR形式でない場合は変換
                self.A = self.A.tocsr()
            
            # 元の行列サイズを保存
            n = self.A.shape[0]
            nnz = self.A.nnz
            
            # 行列からブロックパターンを抽出
            patterns, pattern_indices, unique_count = self._extract_patterns(self.A, block_size)
            
            # パターン統計を保存
            self.pattern_stats = {
                'original_size': n,
                'block_size': block_size,
                'total_blocks': n // block_size,
                'unique_patterns': unique_count,
                'compression_ratio': unique_count * block_size * block_size / nnz if nnz > 0 else 1.0
            }
            
            # 圧縮された行列構造を作成
            self.optimized_A = {
                'patterns': self.cp.array(patterns),
                'pattern_indices': self.cp.array(pattern_indices),
                'shape': self.A.shape,
                'block_size': block_size
            }
            
            # メモリ節約量を計算
            matrix_size = n * n * 8  # 倍精度浮動小数点(8バイト)の稠密行列
            original_sparse_size = (nnz * 8 +  # data配列(float64)
                                   nnz * 4 +  # indices配列(int32)
                                   (n + 1) * 4)  # indptr配列(int32)
            
            # パターン圧縮後のサイズ
            pattern_size = (unique_count * block_size * block_size * 8 +  # パターン辞書(float64)
                           (n // block_size) * 4)  # パターンインデックス(int32)
            
            self.memory_savings = (original_sparse_size - pattern_size) / (1024 * 1024)  # MB単位
            
            print("パターン圧縮を適用しました:")
            print(f"  元の行列: {n}x{n}, 非ゼロ要素: {nnz}")
            print(f"  ブロックサイズ: {block_size}x{block_size}")
            print(f"  検出されたユニークパターン: {unique_count}/{n//block_size}")
            print(f"  圧縮率: {unique_count*block_size*block_size/nnz:.2f}")
            print(f"  推定メモリ節約量: {self.memory_savings:.2f} MB")
            
        except Exception as e:
            print(f"パターン圧縮中にエラーが発生しました: {e}")
            import traceback
            print(traceback.format_exc())
            self.use_optimized_spmv = False
    
    def _extract_patterns(self, A, block_size):
        """
        CSR行列からブロックパターンを抽出
        
        Args:
            A: CSR形式の行列
            block_size: ブロックサイズ
            
        Returns:
            tuple: (パターン配列, パターンインデックス配列, ユニークパターン数)
        """
        n = A.shape[0]
        num_blocks = n // block_size
        
        # NumPy配列に変換（パターン抽出のため）
        if hasattr(A, 'get'):
            A_data = A.data.get()
            A_indices = A.indices.get()
            A_indptr = A.indptr.get()
        else:
            A_data = A.data
            A_indices = A.indices
            A_indptr = A.indptr
        
        # パターン抽出
        patterns = {}
        pattern_indices = np.zeros(num_blocks, dtype=np.int32)
        
        for blk_idx in range(num_blocks):
            # ブロックの行・列インデックス範囲
            row_start = blk_idx * block_size
            row_end = row_start + block_size
            
            # このブロックの非ゼロ要素を抽出
            block_data = []
            block_rows = []
            block_cols = []
            
            for r in range(row_start, row_end):
                if r < n:  # 行が範囲内かチェック
                    for j in range(A_indptr[r], A_indptr[r+1]):
                        c = A_indices[j]
                        col_blk = c // block_size
                        
                        # ブロック内の相対位置を計算
                        local_row = r - row_start
                        local_col = c % block_size
                        
                        # このブロック内の要素として追加
                        block_data.append(A_data[j])
                        block_rows.append(local_row)
                        block_cols.append(local_col + col_blk * block_size)
            
            # パターンキーを生成（相対位置と値のタプル）
            pattern_key = tuple(zip(block_rows, block_cols, block_data))
            
            # 既存のパターンと照合
            if pattern_key in patterns:
                pattern_indices[blk_idx] = patterns[pattern_key]['id']
            else:
                # 新しいパターンを登録
                pattern_id = len(patterns)
                patterns[pattern_key] = {
                    'id': pattern_id,
                    'rows': block_rows,
                    'cols': block_cols,
                    'data': block_data
                }
                pattern_indices[blk_idx] = pattern_id
        
        # パターン辞書からNumPy配列に変換
        unique_count = len(patterns)
        pattern_array = np.zeros((unique_count, block_size, block_size), dtype=np.float64)
        
        for pattern in patterns.values():
            pid = pattern['id']
            for i, (r, c, v) in enumerate(zip(pattern['rows'], pattern['cols'], pattern['data'])):
                if r < block_size and c < block_size:
                    pattern_array[pid, r, c] = v
        
        return pattern_array, pattern_indices, unique_count
    
    def _pattern_spmv(self, x):
        """
        パターン圧縮を活用した最適化行列ベクトル積を計算
        
        Args:
            x: 乗算するベクトル
            
        Returns:
            行列ベクトル積の結果
        """
        if not self.has_cupy or not self.use_optimized_spmv or self.optimized_A is None:
            # 最適化が使用できない場合は通常の方法で計算
            return self.A @ x
        
        # パターンデータを取得
        patterns = self.optimized_A['patterns']
        pattern_indices = self.optimized_A['pattern_indices']
        block_size = self.optimized_A['block_size']
        n = self.optimized_A['shape'][0]
        
        # 入力・出力ベクトルが正しい形状かチェック
        if x.shape[0] != n:
            raise ValueError(f"入力ベクトルの形状が不正です: {x.shape}、期待値: ({n},)")
        
        # 結果を格納するベクトルを初期化
        y = self.cp.zeros(n, dtype=x.dtype)
        
        # カスタムカーネルを定義（パターン最適化SpMV）
        pattern_spmv_kernel = r'''
        extern "C" __global__
        void pattern_spmv(const double* patterns, const int* pattern_indices, 
                          const double* x, double* y, int n, int block_size, int num_patterns) {
            int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (block_idx >= n / block_size) return;
            
            int pattern_id = pattern_indices[block_idx];
            if (pattern_id >= num_patterns) return;
            
            const double* pattern = patterns + pattern_id * block_size * block_size;
            
            for (int i = 0; i < block_size; i++) {
                double sum = 0.0;
                int row = block_idx * block_size + i;
                if (row >= n) continue;
                
                // パターン内の行ごとに全列について総和
                for (int j = 0; j < block_size; j++) {
                    for (int blk = 0; blk < n / block_size; blk++) {
                        int col = blk * block_size + j;
                        if (col >= n) continue;
                        
                        double val = pattern[i * block_size + j];
                        if (val != 0.0) {
                            sum += val * x[col];
                        }
                    }
                }
                
                y[row] = sum;
            }
        }
        '''
        
        # CUDAモジュールをコンパイル
        try:
            pattern_spmv_mod = self.cp.RawModule(code=pattern_spmv_kernel)
            pattern_spmv_ker = pattern_spmv_mod.get_function('pattern_spmv')
            
            # カーネルを起動
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            
            pattern_spmv_ker(
                (blocks_per_grid,), (threads_per_block,),
                (patterns, pattern_indices, x, y, n, block_size, patterns.shape[0])
            )
            
            return y
            
        except Exception as e:
            print(f"パターンSpMVカーネル実行中にエラーが発生しました: {e}")
            # フォールバック: 通常の行列積を使用
            return self.A @ x
    
    # カスタム行列ベクトル積演算子を作成
    def _create_pattern_operator(self):
        """
        パターン圧縮された行列演算子を作成
        """
        # 行列ベクトル積のラッパー関数
        def pattern_matvec(x):
            return self._pattern_spmv(x)
        
        # カスタム行列ベクトル積演算子
        class PatternOperator:
            def __init__(self, shape, matvec):
                self.shape = shape
                self.matvec = matvec
                
            def __matmul__(self, x):
                return self.matvec(x)
                
            def dot(self, x):
                return self.matvec(x)
        
        return PatternOperator(self.optimized_A['shape'], pattern_matvec)
    
    # 各ソルバーメソッドを最適化版にオーバーライド
    def _solve_gmres(self, A, b, options=None):
        """GMRES法（パターン圧縮最適化版）"""
        if not self.has_cupy or not self.use_optimized_spmv:
            return super()._solve_gmres(A, b, options)
            
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", min(200, max(20, b.size // 10)))
        
        x0 = options.get("x0")
        if x0 is not None:
            x0 = self.cp.asarray(x0)
            print(f"CuPy x0を設定しました (shape: {x0.shape})")
        else:
            x0 = self.cp.zeros_like(b)
        
        # パターン圧縮された演算子を作成
        op = self._create_pattern_operator()
        
        # GMRES実行
        result = self.splinalg.gmres(op, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
        return result[0], result[1]
    
    def _solve_cg(self, A, b, options=None):
        """共役勾配法（パターン圧縮最適化版）"""
        if not self.has_cupy or not self.use_optimized_spmv:
            return super()._solve_cg(A, b, options)
            
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        x0 = options.get("x0")
        if x0 is not None:
            x0 = self.cp.asarray(x0)
            print(f"CuPy x0を設定しました (shape: {x0.shape})")
        else:
            x0 = self.cp.zeros_like(b)
        
        # パターン圧縮された演算子を作成
        op = self._create_pattern_operator()
        
        # CG実行
        result = self.splinalg.cg(op, b, x0=x0, tol=tol, maxiter=maxiter)
        return result[0], result[1]
    
    def _solve_cgs(self, A, b, options=None):
        """CGS法（パターン圧縮最適化版）"""
        if not self.has_cupy or not self.use_optimized_spmv:
            return super()._solve_cgs(A, b, options)
            
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        x0 = options.get("x0")
        if x0 is not None:
            x0 = self.cp.asarray(x0)
            print(f"CuPy x0を設定しました (shape: {x0.shape})")
        else:
            x0 = self.cp.zeros_like(b)
        
        # パターン圧縮された演算子を作成
        op = self._create_pattern_operator()
        
        # CGS実行
        result = self.splinalg.cgs(op, b, x0=x0, tol=tol, maxiter=maxiter)
        return result[0], result[1]
    
    def _solve_minres(self, A, b, options=None):
        """MINRES法（パターン圧縮最適化版）"""
        if not self.has_cupy or not self.use_optimized_spmv:
            return super()._solve_minres(A, b, options)
            
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        x0 = options.get("x0")
        if x0 is not None:
            x0 = self.cp.asarray(x0)
            print(f"CuPy x0を設定しました (shape: {x0.shape})")
        else:
            x0 = self.cp.zeros_like(b)
        
        # パターン圧縮された演算子を作成
        op = self._create_pattern_operator()
        
        # MINRES実行
        result = self.splinalg.minres(op, b, x0=x0, tol=tol, maxiter=maxiter)
        return result[0], result[1]
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB法（パターン圧縮最適化版）"""
        if not self.has_cupy or not self.use_optimized_spmv:
            return super()._solve_bicgstab(A, b, options)
            
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        x0 = options.get("x0")
        if x0 is not None:
            x0 = self.cp.asarray(x0)
            print(f"CuPy x0を設定しました (shape: {x0.shape})")
        else:
            x0 = self.cp.zeros_like(b)
        
        # パターン圧縮された演算子を作成
        op = self._create_pattern_operator()
        
        # BiCGSTAB実行
        result = self.splinalg.bicgstab(op, b, x0=x0, tol=tol, maxiter=maxiter)
        return result[0], result[1]
        
    def _solve_lsqr(self, A, b, options=None):
        """LSQR法（パターン圧縮最適化版）"""
        if not self.has_cupy or not self.use_optimized_spmv:
            return super()._solve_lsqr(A, b, options)
            
        options = options or {}
        
        # lsqrはx0をサポートしていないかもしれないため、注意が必要
        # CuPyのlsqrの引数をチェック
        try:
            import inspect
            sig = inspect.signature(self.splinalg.lsqr)
            
            # 引数に応じてlsqrを呼び出し
            if 'x0' in sig.parameters:
                x0 = options.get("x0")
                if x0 is not None:
                    x0 = self.cp.asarray(x0)
                    print(f"CuPy x0を設定しました (shape: {x0.shape})")
                    
                # パターン圧縮された演算子を作成
                op = self._create_pattern_operator()
                
                # LSQR実行 (x0あり)
                result = self.splinalg.lsqr(op, b, x0=x0)
            else:
                # パターン圧縮された演算子を作成
                op = self._create_pattern_operator()
                
                # LSQR実行 (x0なし)
                result = self.splinalg.lsqr(op, b)
                
            return result[0], result[2]  # lsqrは[x, istop, itn, ...]を返す
            
        except Exception as e:
            print(f"LSQR実行エラー: {e}")
            return super()._solve_lsqr(A, b, options)