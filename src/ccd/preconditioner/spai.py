"""
疎行列近似逆行列 (SPAI) 前処理

このモジュールは、フロベニウスノルム最小化に基づいて
システム行列の疎行列近似逆行列を計算する前処理手法を提供します。
"""

import numpy as np
import scipy.sparse as sparse
from scipy.linalg import lstsq
from .base import BasePreconditioner

class SPAIPreconditioner(BasePreconditioner):
    """疎行列近似逆行列前処理"""
    
    def __init__(self, epsilon=1e-10, k=5, max_density=0.1):
        """
        初期化
        
        Args:
            epsilon: 小さい要素を切り捨てる閾値
            k: 列あたりの非ゼロ要素の最大数
            max_density: 近似逆行列の最大密度
        """
        super().__init__()
        self.epsilon = epsilon
        self.k = k
        self.max_density = max_density
        self.M = None
    
    def setup(self, A):
        """
        SPAI前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            # CSR形式に変換（実装を簡素化）
            if hasattr(A, 'tocsr'):
                A_csr = A.tocsr()
            elif hasattr(A, 'toarray'):
                A_csr = sparse.csr_matrix(A.toarray())
            else:
                A_csr = sparse.csr_matrix(A)
            
            # CuPy/JAX配列をNumPyに変換
            if hasattr(A_csr, 'get'):
                A_csr = A_csr.get()
            
            n = A_csr.shape[0]
            
            # A^T の CSR形式も用意（列アクセスを行で効率的に行うため）
            A_csc = A_csr.tocsc()
            
            # 結果格納用の行列を初期化（対角要素だけのパターンから開始）
            M_data = []
            M_indices = []
            M_indptr = [0]
            
            # 各列に対して最小二乗問題を解く
            print(f"SPAI前処理器: {n}x{n}行列の近似逆行列を計算中...")
            
            # 進捗表示のために処理した列数を追跡
            progress_step = max(1, n // 10)
            
            for col in range(n):
                # 進捗表示
                if col % progress_step == 0:
                    print(f"  列 {col}/{n} 処理中 ({col/n*100:.1f}%)")
                
                # 単位ベクトル e_i (標準基底)
                e_i = np.zeros(n)
                e_i[col] = 1.0
                
                # パターン選択: A^T のパターンに基づく簡略版
                # 実際には、より洗練された戦略（例：補強、拡張など）がある
                pattern = A_csc[:, col].nonzero()[0]
                
                # パターンが空の場合、対角要素のみ使用
                if len(pattern) == 0:
                    pattern = np.array([col])
                
                # パターンが多すぎる場合は制限
                if len(pattern) > self.k:
                    # 重要な行（例：対角要素を含む行とその近傍）を優先
                    diag_idx = np.where(pattern == col)[0]
                    if len(diag_idx) > 0:
                        diag_pos = diag_idx[0]
                        # 対角要素とその周りのkつの要素を選択
                        start = max(0, diag_pos - self.k // 2)
                        end = min(len(pattern), start + self.k)
                        start = max(0, end - self.k)  # 再調整
                        pattern = pattern[start:end]
                    else:
                        # 対角要素がない場合はランダムに選択
                        pattern = np.random.choice(pattern, size=self.k, replace=False)
                
                # 部分行列を抽出
                A_local = A_csr[pattern, :].toarray()
                
                # 最小二乗問題を解く: min ||A_local * m - e_i||
                try:
                    m, residuals, rank, s = lstsq(A_local, e_i[pattern])
                except Exception:
                    # 対角スケーリングにフォールバック
                    m = np.zeros(len(pattern))
                    diag_idx = np.where(pattern == col)[0]
                    if len(diag_idx) > 0:
                        aii = A_csr[col, col]
                        if abs(aii) > self.epsilon:
                            m[diag_idx[0]] = 1.0 / aii
                        else:
                            m[diag_idx[0]] = 1.0
                
                # 小さい値を切り捨て
                m[np.abs(m) < self.epsilon] = 0.0
                
                # 非ゼロ要素のみを保存
                for idx, val in zip(pattern, m):
                    if val != 0:
                        M_data.append(val)
                        M_indices.append(idx)
                
                M_indptr.append(len(M_indices))
            
            # 疎行列として格納
            self.M = sparse.csr_matrix((M_data, M_indices, M_indptr), shape=(n, n))
            
            # 密度チェックと必要に応じたさらなる制限
            density = self.M.nnz / (n * n)
            if density > self.max_density:
                print(f"SPAI密度 ({density:.4f}) が閾値 ({self.max_density:.4f}) を超えています。さらに疎性を強制します。")
                # 各行で絶対値が大きい要素だけを保持
                data = []
                indices = []
                indptr = [0]
                
                for i in range(n):
                    row_start = self.M.indptr[i]
                    row_end = self.M.indptr[i+1]
                    row_data = self.M.data[row_start:row_end]
                    row_indices = self.M.indices[row_start:row_end]
                    
                    # 行の非ゼロ要素が多すぎる場合、上位k個だけ保持
                    if len(row_data) > self.k:
                        idx = np.argsort(-np.abs(row_data))[:self.k]
                        row_data = row_data[idx]
                        row_indices = row_indices[idx]
                    
                    # 追加
                    data.extend(row_data)
                    indices.extend(row_indices)
                    indptr.append(len(indices))
                
                self.M = sparse.csr_matrix((data, indices, indptr), shape=(n, n))
            
            # 情報表示
            nnz = self.M.nnz
            density = nnz / (n * n)
            memory_MB = (nnz * 12) / (1024 * 1024)  # 12バイト/要素 (8バイト値 + 4バイト位置)
            print(f"SPAI前処理行列: サイズ={n}x{n}, 非ゼロ要素={nnz}, 密度={density:.6f}, メモリ={memory_MB:.2f}MB")
            
            return self
            
        except Exception as e:
            print(f"SPAI前処理計算エラー: {e}")
            import traceback
            traceback.print_exc()
            self.M = None
            return self
    
    def __call__(self, b):
        """
        前処理を適用 (M*b)
        
        Args:
            b: ベクトル
            
        Returns:
            前処理したベクトル
        """
        if self.M is None:
            return b
            
        # ベクトル形式に応じて処理
        if hasattr(b, 'get'):  # CuPy
            # CPUで計算
            b_cpu = b.get()
            result = self.M @ b_cpu
            
            # CuPyに戻す
            import cupy as cp
            return cp.array(result)
            
        elif 'jax' in str(type(b)):  # JAX
            # CPUで計算
            b_cpu = np.array(b)
            result = self.M @ b_cpu
            
            # JAXに戻す
            import jax.numpy as jnp
            return jnp.array(result)
            
        else:  # NumPy
            return self.M @ b
    
    @property
    def description(self):
        """前処理器の説明"""
        return f"疎行列近似逆行列前処理 (epsilon={self.epsilon}, k={self.k}, max_density={self.max_density})"