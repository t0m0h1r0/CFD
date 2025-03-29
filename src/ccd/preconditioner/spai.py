"""
疎行列近似逆行列 (SPAI) 前処理

このモジュールは、フロベニウスノルム最小化に基づく
システム行列の疎行列近似逆行列を計算する前処理手法を提供します。
"""

from .base import BasePreconditioner

class SPAIPreconditioner(BasePreconditioner):
    """疎行列近似逆行列前処理"""
    
    def __init__(self, epsilon=1e-10, k=5, frobenius_norm=True):
        """
        初期化
        
        Args:
            epsilon: 小さい要素を切り捨てる閾値
            k: 行あたりの非ゼロ要素の最大数
            frobenius_norm: フロベニウスノルム最小化方式を使用
        """
        super().__init__()
        self.epsilon = epsilon
        self.k = k
        self.frobenius_norm = frobenius_norm
    
    def setup(self, A):
        """
        SPAI前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            from scipy import sparse
            import numpy as np
            
            # CSR形式に変換（効率的な行操作のため）
            if hasattr(A, 'format'):
                if A.format != 'csr':
                    A_csr = sparse.csr_matrix(A)
                else:
                    A_csr = A
            else:
                # 密行列の場合
                A_csr = sparse.csr_matrix(A)
            
            n = A.shape[0]
            
            # 行列Aと類似構造の疎行列を初期化
            data = []
            indices = []
            indptr = [0]
            
            if self.frobenius_norm:
                # ||AM - I||_Fを最小化して近似逆行列を計算
                for i in range(n):
                    # i行目のパターンを抽出
                    row_pattern = A_csr[i, :].nonzero()[1]
                    
                    # 行が空の場合、対角成分を使用
                    if len(row_pattern) == 0:
                        row_pattern = np.array([i])
                    
                    # 最小二乗問題用のローカル密行列を作成
                    local_A = A_csr[:, row_pattern].toarray()
                    local_e = np.zeros(n)
                    local_e[i] = 1.0
                    
                    # 最小二乗問題を解いて ||A*m - e|| を最小化
                    try:
                        from scipy.linalg import lstsq
                        m, residuals, rank, s = lstsq(local_A, local_e)
                    except Exception:
                        # フォールバック: 対角成分を使用
                        m = np.zeros(len(row_pattern))
                        diag_idx = np.where(row_pattern == i)[0]
                        if len(diag_idx) > 0:
                            m[diag_idx[0]] = 1.0 / A_csr[i, i] if A_csr[i, i] != 0 else 1.0
                    
                    # 最大k個の要素のみ保持
                    if len(m) > self.k:
                        # 絶対値でソート
                        sorted_idx = np.argsort(-np.abs(m))
                        k_idx = sorted_idx[:self.k]
                        m_filtered = np.zeros_like(m)
                        m_filtered[k_idx] = m[k_idx]
                        m = m_filtered
                    
                    # 小さい要素を除去
                    m[np.abs(m) < self.epsilon] = 0.0
                    
                    # 非ゼロ要素を疎行列に追加
                    for j, val in zip(row_pattern, m):
                        if val != 0:
                            data.append(val)
                            indices.append(j)
                    
                    indptr.append(len(indices))
            else:
                # 簡易近似: 各要素を逆数化
                for i in range(n):
                    row_start = A_csr.indptr[i]
                    row_end = A_csr.indptr[i+1]
                    row_indices = A_csr.indices[row_start:row_end]
                    row_data = A_csr.data[row_start:row_end]
                    
                    # 対角成分を検索
                    diag_idx = np.where(row_indices == i)[0]
                    if len(diag_idx) > 0:
                        diag_val = row_data[diag_idx[0]]
                        inv_diag = 1.0 / diag_val if np.abs(diag_val) > self.epsilon else 1.0
                    else:
                        inv_diag = 1.0
                    
                    # 対角成分を疎行列に追加
                    data.append(inv_diag)
                    indices.append(i)
                    indptr.append(len(indices))
            
            # 疎行列近似逆行列を作成
            self.M = sparse.csr_matrix((data, indices, indptr), shape=A.shape)
            
        except Exception as e:
            print(f"SPAI前処理設定エラー: {e}")
            # 対角行列にフォールバック
            try:
                if hasattr(A, 'diagonal'):
                    diag = A.diagonal()
                    inv_diag = 1.0 / (diag + (np.abs(diag) < self.epsilon) * self.epsilon)
                    self.M = sparse.diags(inv_diag)
                else:
                    self.M = None
            except Exception:
                self.M = None
        
        return self
    
    @property
    def name(self):
        """前処理名を返す"""
        return "SPAIPreconditioner"
    
    @property
    def description(self):
        """前処理の説明を返す"""
        method = "フロベニウスノルム最小化" if self.frobenius_norm else "単純逆数化"
        return f"疎行列近似逆行列前処理 ({method}, epsilon={self.epsilon}, k={self.k})"